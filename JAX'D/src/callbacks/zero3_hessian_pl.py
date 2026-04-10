from contextlib import contextmanager
from typing import Any, Callable, Optional

import gc
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from src.spectral.robust_estimator import RobustSpectralEstimator
from src.tolerance.dynamic_ledger import DynamicToleranceLedger


class Zero3CheckpointedHessianPLCallback(pl.Callback):
    """Distributed Hessian/PL monitoring for ZeRO-3 or FSDP wrapped training."""

    def __init__(
        self,
        pl_tol: float = 2.4e-3,
        k_lanczos: int = 3,
        reg_lambda: float = 1e-4,
        monitor_every: int = 50,
        warmup_steps: int = 100,
        use_checkpointing: bool = True,
        gather_sync: bool = True,
        power_iters: int = 3,
        safety_factor: float = 1.15,
        ema_alpha: float = 0.9,
        ledger: Optional[DynamicToleranceLedger] = None,
    ):
        self.pl_tol = pl_tol
        self.k = k_lanczos
        self.reg_lambda = reg_lambda
        self.monitor_every = monitor_every
        self.warmup_steps = warmup_steps
        self.use_checkpointing = use_checkpointing
        self.gather_sync = gather_sync
        self.loss_star = None
        self.mu_global = None
        self.L_global = None
        self.ledger = ledger
        self.spectral_est = RobustSpectralEstimator(
            k_lanczos=k_lanczos,
            power_iters=power_iters,
            safety_factor=safety_factor,
            ema_alpha=ema_alpha,
            reg_floor=reg_lambda,
        )

    @staticmethod
    def _dist_ready() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _all_reduce_tensor(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        if not Zero3CheckpointedHessianPLCallback._dist_ready():
            return tensor
        dist.all_reduce(tensor, op=op)
        return tensor

    @contextmanager
    def _gather_parameters(self, model: torch.nn.Module):
        if hasattr(model, "_fsdp_wrapped_module"):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(model, writeback=False, rank0_only=False):
                yield
        elif hasattr(model, "zero_gathered_parameters"):
            with model.zero_gathered_parameters():
                yield
        else:
            yield

    def _forward_loss(self, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> torch.Tensor:
        if not self.use_checkpointing:
            return pl_module.training_step(batch, batch_idx)

        def forward_fn(*flat_batch: torch.Tensor) -> torch.Tensor:
            actual_batch: Any
            if isinstance(batch, (tuple, list)):
                actual_batch = type(batch)(flat_batch)
            else:
                actual_batch = flat_batch[0]
            return pl_module.training_step(actual_batch, batch_idx)

        if isinstance(batch, (tuple, list)):
            return checkpoint(forward_fn, *batch, use_reentrant=False)
        return checkpoint(forward_fn, batch, use_reentrant=False)

    def _hvp_gathered(
        self,
        model: torch.nn.Module,
        pl_module: pl.LightningModule,
        params: list[torch.Tensor],
        batch: Any,
        batch_idx: int,
        v: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        del model
        with self._gather_parameters(pl_module.model if hasattr(pl_module, "model") else pl_module):
            loss = self._forward_loss(pl_module, batch, batch_idx)
            grad = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            grad_v = sum(torch.sum(g * v_elem) for g, v_elem in zip(grad, v))
            hvp = torch.autograd.grad(grad_v, params, retain_graph=False)
        return hvp

    def _build_hvp_fn(
        self,
        model: torch.nn.Module,
        pl_module: pl.LightningModule,
        params: list[torch.Tensor],
        batch: Any,
        batch_idx: int,
    ) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
        def hvp_fn(v: list[torch.Tensor]) -> list[torch.Tensor]:
            hvp = self._hvp_gathered(model, pl_module, params, batch, batch_idx, v)
            if self.gather_sync and torch.cuda.is_available() and params[0].device.type == "cuda":
                torch.cuda.synchronize(params[0].device)
                gc.collect()
                torch.cuda.empty_cache()
            return hvp

        return hvp_fn

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs
        if batch_idx % self.monitor_every != 0 or trainer.global_step < self.warmup_steps:
            return

        is_rank0 = trainer.global_rank == 0
        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        params = [p for p in model.parameters() if p.requires_grad]

        with torch.no_grad():
            loss_eval = self._forward_loss(pl_module, batch, batch_idx)

        if self.loss_star is None:
            self.loss_star = float(loss_eval.detach())

        loss_bp = self._forward_loss(pl_module, batch, batch_idx)
        grad = torch.autograd.grad(loss_bp, params, retain_graph=True)
        grad_sq = sum(torch.sum(g**2) for g in grad)
        grad_sq = self._all_reduce_tensor(grad_sq)
        grad_norm_sq = float(grad_sq.detach())

        hvp_fn = self._build_hvp_fn(model, pl_module, params, batch, batch_idx)
        v_sample = [torch.randn_like(p) for p in params]
        self.L_global, self.mu_global = self.spectral_est.estimate(hvp_fn, v_sample)
        cond_num = self.L_global / max(self.mu_global, 1e-10)

        loss_gap = float(loss_eval.detach()) - self.loss_star
        pl_metric = 0.5 * grad_norm_sq / max(loss_gap, 1e-12)
        pl_satisfied = pl_metric >= self.pl_tol
        new_lr = self.spectral_est.compute_adaptive_lr(self.L_global, self.mu_global, min_lr=1e-5)

        if self.ledger is not None and is_rank0:
            pl_residual = abs(pl_metric - 1.0)
            adapted_tol = self.ledger.update_from_residual("hessian_pl", pl_residual, "hessian_callback")
            if adapted_tol < self.pl_tol * 0.8:
                self.pl_tol = adapted_tol

        if self._dist_ready():
            lr_tensor = torch.tensor(new_lr, device=loss_bp.device)
            dist.broadcast(lr_tensor, src=0)
            new_lr = float(lr_tensor.item())

        if trainer.optimizers:
            for pg in trainer.optimizers[0].param_groups:
                pg["lr"] = new_lr

        if is_rank0:
            pl_module.log_dict(
                {
                    "dist/hessian/mu_safe": self.mu_global,
                    "dist/hessian/L_safe": self.L_global,
                    "dist/hessian/condition_number": cond_num,
                    "dist/optimizer/pl_satisfied": float(pl_satisfied),
                    "dist/optimizer/adaptive_lr": new_lr,
                },
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        if self._dist_ready():
            dist.barrier()

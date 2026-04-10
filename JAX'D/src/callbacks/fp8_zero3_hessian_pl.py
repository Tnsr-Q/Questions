import logging
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from .precision_controller import PrecisionController

log = logging.getLogger("QUFT_FP8Hessian")


class FP8Zero3HessianPLCallback(pl.Callback):
    """ZeRO-3/FSDP compatible Hessian-PL callback with negotiated precision storage."""

    def __init__(
        self,
        pl_tol: float = 2.4e-3,
        k_lanczos: int = 4,
        reg_lambda: float = 1e-4,
        monitor_every: int = 50,
        warmup_steps: int = 100,
        use_checkpointing: bool = True,
        precision_target: str = "float8_e4m3fn",
    ):
        self.pl_tol = pl_tol
        self.k = k_lanczos
        self.reg_lambda = reg_lambda
        self.monitor_every = monitor_every
        self.warmup_steps = warmup_steps
        self.use_checkpointing = use_checkpointing
        self.prec_ctrl = PrecisionController(target_dtype_str=precision_target)
        self.loss_star = None
        self.mu_global = None
        self.L_global = None

    @staticmethod
    def _dist_ready() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _all_reduce_tensor(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        if not FP8Zero3HessianPLCallback._dist_ready():
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

    def _checkpointed_forward(self, model: pl.LightningModule, batch: Any, batch_idx: int) -> torch.Tensor:
        if not self.use_checkpointing:
            return model.training_step(batch, batch_idx)

        def fn(*flat_batch: torch.Tensor):
            actual_batch: Any
            if isinstance(batch, (tuple, list)):
                actual_batch = type(batch)(flat_batch)
            else:
                actual_batch = flat_batch[0]
            return model.training_step(actual_batch, batch_idx)

        if isinstance(batch, (tuple, list)):
            return checkpoint(fn, *batch, use_reentrant=False)
        return checkpoint(fn, batch, use_reentrant=False)

    def _hvp_quantized(
        self,
        pl_module: pl.LightningModule,
        model: torch.nn.Module,
        params: list[torch.Tensor],
        batch: Any,
        batch_idx: int,
        vec: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        with self._gather_parameters(model):
            loss = self._checkpointed_forward(pl_module, batch, batch_idx)
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

            grad_v = sum(torch.sum(g.to(torch.float32) * v.to(torch.float32)) for g, v in zip(grads, vec))
            hvp = torch.autograd.grad(grad_v, params, retain_graph=False)

            for g in grads:
                quantized, scale = self.prec_ctrl.quantize(g.detach())
                _ = self.prec_ctrl.track_quantization_error(g.detach(), quantized, scale)

        return hvp

    def _distributed_lanczos(
        self,
        pl_module: pl.LightningModule,
        model: torch.nn.Module,
        params: list[torch.Tensor],
        batch: Any,
        batch_idx: int,
    ) -> torch.Tensor:
        q_prev = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        q = [torch.randn_like(p, dtype=torch.float32) for p in params]
        q_norm = torch.sqrt(sum(torch.sum(x * x) for x in q))
        q = [x / (q_norm + 1e-12) for x in q]

        alphas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        device = params[0].device
        beta_prev = torch.tensor(0.0, device=device)

        for _ in range(self.k):
            hvp = self._hvp_quantized(pl_module, model, params, batch, batch_idx, q)
            hvp32 = [h.to(torch.float32) for h in hvp]

            alpha = sum(torch.sum(qi * hi) for qi, hi in zip(q, hvp32))
            alphas.append(alpha)

            r = [hi - alpha * qi - beta_prev * qpi for hi, qi, qpi in zip(hvp32, q, q_prev)]

            dot_q = sum(torch.sum(ri * qi) for ri, qi in zip(r, q))
            r = [ri - dot_q * qi for ri, qi in zip(r, q)]
            dot_prev = sum(torch.sum(ri * qpi) for ri, qpi in zip(r, q_prev))
            r = [ri - dot_prev * qpi for ri, qpi in zip(r, q_prev)]

            beta = torch.sqrt(sum(torch.sum(ri * ri) for ri in r))
            if beta < 1e-10:
                break

            betas.append(beta)
            q_prev = q
            q = [ri / (beta + 1e-12) for ri in r]
            beta_prev = beta

        n = len(alphas)
        if n == 0:
            return torch.tensor([self.reg_lambda, self.reg_lambda], device=device)

        T = torch.zeros((n, n), device=device)
        for i in range(n):
            T[i, i] = alphas[i]
            if i < n - 1 and i < len(betas):
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]

        return torch.linalg.eigvalsh(T)

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
            loss_eval = self._checkpointed_forward(pl_module, batch, batch_idx)
        if self.loss_star is None:
            self.loss_star = float(loss_eval.detach())

        loss_bp = self._checkpointed_forward(pl_module, batch, batch_idx)
        grad = torch.autograd.grad(loss_bp, params, retain_graph=True)
        grad_sq = sum(torch.sum(g.to(torch.float32) ** 2) for g in grad)
        grad_norm_sq = float(self._all_reduce_tensor(grad_sq).detach())

        local_eigs = self._distributed_lanczos(pl_module, model, params, batch, batch_idx)
        if self._dist_ready():
            gathered = [torch.zeros_like(local_eigs) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local_eigs)
            all_eigs = torch.cat(gathered).detach().cpu().numpy()
        else:
            all_eigs = local_eigs.detach().cpu().numpy()

        self.mu_global = float(np.min(all_eigs)) + self.reg_lambda
        self.L_global = float(np.max(all_eigs)) + self.reg_lambda
        cond_num = self.L_global / max(self.mu_global, 1e-10)

        avg_q_err = self.prec_ctrl.telemetry["precision/quantization_error"]
        if not np.isnan(avg_q_err) and avg_q_err > 1e-3 and self.mu_global < self.pl_tol:
            log.warning("Quantization may degrade PL constant; BF16 fallback is recommended.")

        if is_rank0:
            loss_gap = float(loss_eval.detach()) - self.loss_star
            pl_satisfied = (0.5 * grad_norm_sq / max(loss_gap, 1e-12)) >= self.pl_tol
            eta_opt = 1.0 / max(self.L_global + self.mu_global, 1e-12)
            eta_bar = 2.0 / max(self.L_global, 1e-12)
            new_lr = float(np.clip(eta_opt, 1e-5, eta_bar * 0.9))

            if self._dist_ready():
                lr_tensor = torch.tensor(new_lr, device=loss_bp.device)
                dist.broadcast(lr_tensor, src=0)
                new_lr = float(lr_tensor.item())

            if trainer.optimizers:
                for pg in trainer.optimizers[0].param_groups:
                    pg["lr"] = new_lr

            telemetry = self.prec_ctrl.telemetry
            metrics = {
                "dist/hessian/mu": self.mu_global,
                "dist/hessian/L": self.L_global,
                "dist/hessian/condition_number": cond_num,
                "dist/optimizer/pl_satisfied": float(pl_satisfied),
                "dist/optimizer/adaptive_lr": new_lr,
                "dist/precision/fp8_available": float(telemetry["precision/fp8_available"]),
                "dist/precision/compile_compatible": float(telemetry["precision/compile_compatible"]),
                "dist/precision/quantization_error": float(telemetry["precision/quantization_error"]),
            }

            pl_module.log_dict(
                metrics,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        if self._dist_ready():
            dist.barrier()

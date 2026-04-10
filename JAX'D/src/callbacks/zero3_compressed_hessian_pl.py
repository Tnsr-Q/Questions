import logging
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

log = logging.getLogger("QUFT_ZeRO3Compressed")


class OneBitCompressor:
    """1-bit sign compression with decayed error feedback."""

    def __init__(self, error_decay: float = 0.9):
        self.error_decay = error_decay
        self.buffers: dict[str, torch.Tensor] = {}

    def compress(self, name: str, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if name not in self.buffers:
            self.buffers[name] = torch.zeros_like(grad, dtype=torch.float32)

        corrected = grad + self.buffers[name]
        scale = torch.max(torch.abs(corrected)) + 1e-8
        quantized = torch.sign(corrected)
        self.buffers[name] = self.error_decay * (corrected - quantized * scale)
        return quantized, scale

    def all_reduce_compressed(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(scale, op=dist.ReduceOp.SUM)
        scale /= dist.get_world_size()
        dist.all_reduce(quantized, op=dist.ReduceOp.SUM)
        return quantized * scale


class CompressedZero3HessianPLCallback(pl.Callback):
    """ZeRO-3/FSDP Hessian-PL callback with optional 1-bit gradient compression."""

    def __init__(
        self,
        pl_tol: float = 2.4e-3,
        k_lanczos: int = 3,
        reg_lambda: float = 1e-4,
        monitor_every: int = 50,
        warmup_steps: int = 100,
        use_checkpointing: bool = True,
        compress_gradients: bool = True,
        error_decay: float = 0.9,
    ):
        self.pl_tol = pl_tol
        self.k = k_lanczos
        self.reg_lambda = reg_lambda
        self.monitor_every = monitor_every
        self.warmup_steps = warmup_steps
        self.use_checkpointing = use_checkpointing
        self.compressor = OneBitCompressor(error_decay) if compress_gradients else None
        self.loss_star = None
        self.mu_global = None
        self.L_global = None

    @staticmethod
    def _dist_ready() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _all_reduce_tensor(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        if not CompressedZero3HessianPLCallback._dist_ready():
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

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        del trainer, optimizer
        if self.compressor is None or not self._dist_ready():
            return

        for p in pl_module.parameters():
            if p.requires_grad and p.grad is not None:
                pname = getattr(p, "_name", None) or str(id(p))
                q, s = self.compressor.compress(pname, p.grad)
                p.grad.copy_(self.compressor.all_reduce_compressed(q, s))

    def _checkpointed_forward(self, model: torch.nn.Module, batch: Any, batch_idx: int) -> torch.Tensor:
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

    def _distributed_lanczos(self, model: pl.LightningModule, params: list[torch.Tensor], batch: Any, batch_idx: int) -> torch.Tensor:
        q_prev = [torch.zeros_like(p) for p in params]
        q = [torch.randn_like(p) for p in params]
        q_norm = torch.sqrt(sum(torch.sum(x * x) for x in q))
        q = [x / (q_norm + 1e-12) for x in q]

        alphas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        device = params[0].device
        beta_prev = torch.tensor(0.0, device=device)

        for _ in range(self.k):
            with self._gather_parameters(model.model if hasattr(model, "model") else model):
                loss = self._checkpointed_forward(model, batch, batch_idx)
                grad = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                grad_v = sum(torch.sum(g * v_elem) for g, v_elem in zip(grad, q))
                hvp = torch.autograd.grad(grad_v, params, retain_graph=False)

            alpha = sum(torch.sum(v_elem * h_elem) for v_elem, h_elem in zip(q, hvp))
            alphas.append(alpha)

            r = [h_elem - alpha * v_elem - beta_prev * qp for h_elem, v_elem, qp in zip(hvp, q, q_prev)]
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
        grad_sq = sum(torch.sum(g**2) for g in grad)
        grad_norm_sq = float(self._all_reduce_tensor(grad_sq).detach())

        local_eigs = self._distributed_lanczos(pl_module, params, batch, batch_idx)
        if self._dist_ready():
            gathered = [torch.zeros_like(local_eigs) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local_eigs)
            all_eigs = torch.cat(gathered).detach().cpu().numpy()
        else:
            all_eigs = local_eigs.detach().cpu().numpy()

        self.mu_global = float(np.min(all_eigs)) + self.reg_lambda
        self.L_global = float(np.max(all_eigs)) + self.reg_lambda
        cond_num = self.L_global / max(self.mu_global, 1e-10)

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

            pl_module.log_dict(
                {
                    "dist/hessian/mu": self.mu_global,
                    "dist/hessian/L": self.L_global,
                    "dist/hessian/condition_number": cond_num,
                    "dist/optimizer/pl_satisfied": float(pl_satisfied),
                    "dist/optimizer/adaptive_lr": new_lr,
                    "dist/compression/1bit_active": float(self.compressor is not None),
                },
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        if self._dist_ready():
            dist.barrier()

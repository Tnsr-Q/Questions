from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

from src.proto.orbax_atomic import OrbaxAtomicStateIO

logger = logging.getLogger(__name__)


class CheckpointedDistributedHessianPLCallback(pl.Callback):
    """Distributed Hessian/PL monitoring with activation checkpointing support."""

    def __init__(
        self,
        pl_tol: float = 2.4e-3,
        k_lanczos: int = 3,
        reg_lambda: float = 1e-4,
        monitor_every: int = 50,
        warmup_steps: int = 100,
        use_checkpointing: bool = True,
        state_save_path: str | None = None,
    ):
        self.pl_tol = pl_tol
        self.k = k_lanczos
        self.reg_lambda = reg_lambda
        self.monitor_every = monitor_every
        self.warmup_steps = warmup_steps
        self.use_checkpointing = use_checkpointing
        self.state_io = OrbaxAtomicStateIO(state_save_path) if state_save_path else None
        self.loss_star = None
        self.mu_global = None
        self.L_global = None

    @staticmethod
    def _dist_ready() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _all_reduce_tensor(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        if not CheckpointedDistributedHessianPLCallback._dist_ready():
            return tensor
        dist.all_reduce(tensor, op=op)
        return tensor

    def _checkpointed_loss(
        self,
        pl_module: pl.LightningModule,
        model: torch.nn.Module,
        batch: Any,
        batch_idx: int,
    ) -> torch.Tensor:
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

    def _distributed_lanczos(
        self,
        params: list[torch.Tensor],
        loss: torch.Tensor,
    ) -> torch.Tensor:
        q_prev = [torch.zeros_like(p) for p in params]
        q = [torch.randn_like(p) for p in params]
        q_norm = torch.sqrt(sum(torch.sum(x * x) for x in q))
        q = [x / (q_norm + 1e-12) for x in q]

        alphas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        beta_prev = torch.tensor(0.0, device=loss.device)

        for _ in range(self.k):
            grad = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            grad_v = sum(torch.sum(g * v_elem) for g, v_elem in zip(grad, q))
            hvp = torch.autograd.grad(grad_v, params, retain_graph=True)

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

            del grad, hvp

        n = len(alphas)
        if n == 0:
            return torch.tensor([self.reg_lambda, self.reg_lambda], device=loss.device)

        T = torch.zeros((n, n), device=loss.device)
        for i in range(n):
            T[i, i] = alphas[i]
            if i < n - 1 and i < len(betas):
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]

        return torch.linalg.eigvalsh(T)


    def _export_state(self) -> dict[str, float | None]:
        return {
            "mu_global": self.mu_global,
            "L_global": self.L_global,
            "loss_star": self.loss_star,
            "pl_tol": self.pl_tol,
            "k_lanczos": self.k,
            "reg_lambda": self.reg_lambda,
        }

    def save_state(self) -> None:
        if self.state_io is None:
            return
        try:
            self.state_io.save(self._export_state())
        except Exception as exc:
            logger.warning(
                "Failed to save Hessian checkpoint state; continuing without persistence.",
                exc_info=True,
            )

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
            loss_eval = self._checkpointed_loss(pl_module, model, batch, batch_idx)

        if self.loss_star is None:
            self.loss_star = float(loss_eval.detach())

        loss_bp = self._checkpointed_loss(pl_module, model, batch, batch_idx)
        grad = torch.autograd.grad(loss_bp, params, retain_graph=True)
        grad_sq = sum(torch.sum(g**2) for g in grad)
        grad_sq = self._all_reduce_tensor(grad_sq)
        grad_norm_sq = float(grad_sq.detach())

        local_eigs = self._distributed_lanczos(params, loss_bp)
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
                },
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        if is_rank0:
            self.save_state()

        if self._dist_ready():
            dist.barrier()

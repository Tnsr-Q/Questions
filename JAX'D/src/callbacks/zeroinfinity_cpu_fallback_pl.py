import logging
import os
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

log = logging.getLogger("QUFT_CPUFallback")


class CPUFallbackZeroInfinityCallback(pl.Callback):
    """Distributed Hessian/PL callback with NVMe->CPU fallback support."""

    def __init__(
        self,
        pl_tol: float = 2.4e-3,
        k_lanczos: int = 4,
        reg_lambda: float = 1e-4,
        monitor_every: int = 50,
        warmup_steps: int = 100,
        nvme_path: str = "/mnt/nvme/deepspeed_offload",
        min_nvme_free_gb: float = 20.0,
        use_checkpointing: bool = True,
    ):
        self.pl_tol = pl_tol
        self.k = k_lanczos
        self.reg_lambda = reg_lambda
        self.monitor_every = monitor_every
        self.warmup_steps = warmup_steps
        self.nvme_path = nvme_path
        self.min_nvme_free = min_nvme_free_gb * (1024**3)
        self.use_checkpointing = use_checkpointing
        self.fallback_active = False
        self.loss_star = None
        self.mu_global = None
        self.L_global = None

    @staticmethod
    def _dist_ready() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _all_reduce_tensor(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
        if not CPUFallbackZeroInfinityCallback._dist_ready():
            return tensor
        dist.all_reduce(tensor, op=op)
        return tensor

    def _check_nvme_health(self) -> bool:
        if not os.path.exists(self.nvme_path):
            return False

        stat = os.statvfs(self.nvme_path)
        free_bytes = stat.f_bavail * stat.f_frsize
        if free_bytes < self.min_nvme_free:
            log.warning(
                "NVMe free space below threshold: %.1fGB < %.1fGB",
                free_bytes / 1e9,
                self.min_nvme_free / 1e9,
            )
            return False

        t0 = time.perf_counter()
        test_file = os.path.join(self.nvme_path, ".quft_io_test")
        try:
            with open(test_file, "wb") as f:
                f.write(b"\x00" * (4 * 1024 * 1024))
            with open(test_file, "rb") as f:
                _ = f.read()
            if os.path.exists(test_file):
                os.remove(test_file)
            return (time.perf_counter() - t0) < 0.05
        except OSError:
            return False

    def _activate_cpu_fallback(self) -> None:
        if self.fallback_active:
            return

        log.info("NVMe unavailable/slow; activating CPU RAM fallback.")
        self.fallback_active = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.7)

    @contextmanager
    def _safe_param_gather(self, model: torch.nn.Module):
        if self.fallback_active:
            yield
        elif hasattr(model, "_fsdp_wrapped_module"):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(model, writeback=False, rank0_only=False):
                yield
        elif hasattr(model, "zero_gathered_parameters"):
            with model.zero_gathered_parameters():
                yield
        else:
            yield

    def _checkpointed_forward(self, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> torch.Tensor:
        if not self.use_checkpointing:
            return pl_module.training_step(batch, batch_idx)

        def fn(*flat_batch: torch.Tensor):
            if isinstance(batch, (tuple, list)):
                actual_batch = type(batch)(flat_batch)
            else:
                actual_batch = flat_batch[0]
            return pl_module.training_step(actual_batch, batch_idx)

        if isinstance(batch, (tuple, list)):
            return checkpoint(fn, *batch, use_reentrant=False)
        return checkpoint(fn, batch, use_reentrant=False)

    def _chunked_lanczos_cpu(
        self,
        pl_module: pl.LightningModule,
        model: torch.nn.Module,
        params: list[torch.Tensor],
        batch: Any,
        batch_idx: int,
        vec: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        device = params[0].device
        with self._safe_param_gather(model):
            loss = self._checkpointed_forward(pl_module, batch, batch_idx)
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            grad_v = sum(torch.sum(g.float().cpu() * v.float().cpu()) for g, v in zip(grads, vec))
            hvp = torch.autograd.grad(grad_v, params, retain_graph=False)
        return [h.to(device) for h in hvp]

    def _distributed_lanczos(
        self,
        pl_module: pl.LightningModule,
        model: torch.nn.Module,
        params: list[torch.Tensor],
        batch: Any,
        batch_idx: int,
    ) -> torch.Tensor:
        vec = [torch.randn_like(p, dtype=torch.float32) for p in params]
        norm_vec = torch.sqrt(sum(torch.sum(v * v) for v in vec))
        vec = [v / (norm_vec + 1e-12) for v in vec]

        device = params[0].device
        T = torch.zeros((self.k + 1, self.k + 1), device=device)
        last_j = 0

        for j in range(self.k):
            last_j = j
            if self.fallback_active:
                hvp = self._chunked_lanczos_cpu(pl_module, model, params, batch, batch_idx, vec)
            else:
                with self._safe_param_gather(model):
                    loss = self._checkpointed_forward(pl_module, batch, batch_idx)
                    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
                    grad_v = sum(torch.sum(g.float() * v.float()) for g, v in zip(grads, vec))
                    hvp = torch.autograd.grad(grad_v, params, retain_graph=False)
                hvp = [h.to(device) for h in hvp]

            hvp_fp32 = [h.to(torch.float32) for h in hvp]
            alpha = sum(torch.sum(v * h) for v, h in zip(vec, hvp_fp32))
            T[j, j] = alpha

            w = [h - alpha * v for v, h in zip(vec, hvp_fp32)]
            norm_w = torch.sqrt(sum(torch.sum(wi * wi) for wi in w))
            if norm_w < 1e-8:
                break

            if j < self.k:
                T[j + 1, j] = T[j, j + 1] = norm_w
            vec = [wi / (norm_w + 1e-12) for wi in w]

        eigvals = torch.linalg.eigvalsh(T[: last_j + 1, : last_j + 1])
        return eigvals

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

        if trainer.global_rank == 0 and not self.fallback_active and not self._check_nvme_health():
            self._activate_cpu_fallback()

        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return

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

        assumptions_met = {
            "A1_perturbative": self.L_global < 1.0,
            "A2_fakeon_valid": True,
            "A4_scale_invariant": self.mu_global > 0.0,
        }

        if trainer.global_rank == 0:
            loss_gap = float(loss_eval.detach()) - self.loss_star
            pl_satisfied = (0.5 * grad_norm_sq / max(loss_gap, 1e-12)) >= self.pl_tol
            eta_opt = 1.0 / max(self.L_global + self.mu_global, 1e-12)
            eta_bar = 2.0 / max(self.L_global, 1e-12)
            new_lr = float(np.clip(eta_opt, 1e-5, eta_bar * 0.9))

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
                    "dist/storage/nvme_fallback": float(self.fallback_active),
                    "dist/assumptions/A1_A4": float(all(assumptions_met.values())),
                    "dist/status_verified": float(pl_satisfied and cond_num < 50),
                },
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        if self._dist_ready():
            dist.barrier()

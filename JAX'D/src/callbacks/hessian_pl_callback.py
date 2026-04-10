import logging
from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch

log = logging.getLogger("QUFT_Lightning")


class HessianPLCallback(pl.Callback):
    """Lightning callback for stochastic Hessian estimates and PL-condition checks."""

    def __init__(
        self,
        reg_lambda: float = 1e-4,
        k_lanczos: int = 3,
        pl_tol: float = 2.4e-3,
        monitor_every_n_steps: int = 50,
        warmup_steps: int = 100,
        max_lr: float = 3.0,
        min_lr: float = 1e-5,
    ):
        super().__init__()
        self.reg_lambda = reg_lambda
        self.k = k_lanczos
        self.pl_tol = pl_tol
        self.monitor_every = monitor_every_n_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.mu_est = None
        self.L_est = None
        self.loss_star = None

    @staticmethod
    def _hvp(loss: torch.Tensor, params: List[torch.Tensor], vec: List[torch.Tensor]) -> List[torch.Tensor]:
        grad = torch.autograd.grad(loss, params, create_graph=True)
        grad_v = sum(torch.sum(g * v_elem) for g, v_elem in zip(grad, vec))
        return list(torch.autograd.grad(grad_v, params, retain_graph=False))

    def _stochastic_lanczos(self, params: List[torch.Tensor], loss: torch.Tensor) -> np.ndarray:
        """Minimal Lanczos tridiagonalization with HVP queries."""
        q_prev = [torch.zeros_like(p) for p in params]
        q = [torch.randn_like(p) for p in params]
        q_norm = torch.sqrt(sum(torch.sum(x * x) for x in q))
        q = [x / (q_norm + 1e-12) for x in q]

        alphas: list[float] = []
        betas: list[float] = []
        beta_prev = torch.tensor(0.0, device=loss.device)

        for _ in range(self.k):
            z = self._hvp(loss, params, q)
            alpha = sum(torch.sum(qi * zi) for qi, zi in zip(q, z))
            alphas.append(float(alpha.detach().cpu()))

            r = [zi - alpha * qi - beta_prev * qpi for zi, qi, qpi in zip(z, q, q_prev)]
            beta = torch.sqrt(sum(torch.sum(ri * ri) for ri in r))
            beta_val = float(beta.detach().cpu())
            if beta_val < 1e-10:
                break
            betas.append(beta_val)

            q_prev = q
            q = [ri / (beta + 1e-12) for ri in r]
            beta_prev = beta

        n = len(alphas)
        T = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            T[i, i] = alphas[i]
            if i < n - 1 and i < len(betas):
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]

        eigvals = np.linalg.eigvalsh(T) if n > 0 else np.array([self.reg_lambda, self.reg_lambda])
        return eigvals

    def _verify_pl_condition(self, grad_norm_sq: float, loss_gap: float) -> bool:
        if loss_gap < 1e-12:
            return True
        return (0.5 * grad_norm_sq / loss_gap) >= self.pl_tol

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

        model = pl_module.model if hasattr(pl_module, "model") else pl_module
        params = [p for p in model.parameters() if p.requires_grad]
        loss = pl_module.training_step(batch, batch_idx)

        if self.loss_star is None:
            self.loss_star = float(loss.detach())

        grad = torch.autograd.grad(loss, params, retain_graph=True)
        grad_norm_sq = sum(torch.sum(g**2) for g in grad).item()

        eigvals = self._stochastic_lanczos(params, loss)
        self.mu_est = float(eigvals[0]) + self.reg_lambda
        self.L_est = float(eigvals[-1]) + self.reg_lambda
        cond_num = self.L_est / max(self.mu_est, 1e-10)

        loss_gap = float(loss.detach()) - self.loss_star
        pl_satisfied = self._verify_pl_condition(grad_norm_sq, loss_gap)

        eta_opt = 1.0 / max(self.L_est + self.mu_est, 1e-10)
        eta_bar = 2.0 / max(self.L_est, 1e-10)
        max_allowed = min(self.max_lr, eta_bar * 0.9)
        new_lr = float(np.clip(eta_opt, self.min_lr, max_allowed))

        pl_module.log_dict(
            {
                "hessian/mu": self.mu_est,
                "hessian/L": self.L_est,
                "hessian/condition_number": cond_num,
                "optimizer/pl_satisfied": float(pl_satisfied),
                "optimizer/adaptive_lr": new_lr,
            },
            prog_bar=True,
            logger=True,
        )

        if pl_satisfied and cond_num < 50 and trainer.optimizers:
            for pg in trainer.optimizers[0].param_groups:
                pg["lr"] = new_lr
        elif not pl_satisfied:
            log.warning("PL violation at step %s: μ=%.2e, gap=%.2e", trainer.global_step, self.mu_est, loss_gap)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer
        pl_module.log("certification/mu_final", self.mu_est or 0.0)
        pl_module.log("certification/L_final", self.L_est or 0.0)
        if self.mu_est and self.mu_est >= self.pl_tol:
            log.info("✅ Hessian PL Certificate: μ ≥ threshold verified")
        else:
            log.warning("⚠️ Hessian PL Certificate: μ below threshold. Convergence not certified.")

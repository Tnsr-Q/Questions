from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist

VectorList = list[torch.Tensor]
HvFunction = Callable[[VectorList], VectorList]


class RobustSpectralEstimator:
    """Hybrid spectral estimator for distributed Hessian/Gauss-Newton monitoring."""

    def __init__(
        self,
        k_lanczos: int = 8,
        power_iters: int = 3,
        safety_factor: float = 1.15,
        ema_alpha: float = 0.9,
        reg_floor: float = 1e-6,
        tol_ritz: float = 1e-4,
    ) -> None:
        self.k = k_lanczos
        self.p_iters = power_iters
        self.safety = safety_factor
        self.ema_alpha = ema_alpha
        self.reg = reg_floor
        self.tol = tol_ritz

        self.L_ema: Optional[float] = None
        self.mu_ema: Optional[float] = None

    @staticmethod
    def _dist_ready() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _all_reduce_scalar_max(val: float) -> float:
        if not RobustSpectralEstimator._dist_ready():
            return val
        tensor = torch.tensor(val, dtype=torch.float64, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    @staticmethod
    def _dot(x: VectorList, y: VectorList) -> torch.Tensor:
        return sum(torch.sum(x_i * y_i) for x_i, y_i in zip(x, y))

    @staticmethod
    def _norm(x: VectorList) -> torch.Tensor:
        return torch.sqrt(RobustSpectralEstimator._dot(x, x))

    @staticmethod
    def _scale(x: VectorList, scalar: torch.Tensor) -> VectorList:
        return [xi / (scalar + 1e-15) for xi in x]

    def _power_iteration_L(self, Hv_fn: HvFunction, v: VectorList) -> float:
        v = self._scale(v, self._norm(v))
        for _ in range(self.p_iters):
            w = Hv_fn(v)
            v = self._scale(w, self._norm(w))
        rayleigh = self._dot(v, Hv_fn(v))
        return float(rayleigh.detach().item())

    def _adaptive_lanczos_mu(self, Hv_fn: HvFunction, v0: VectorList) -> float:
        base_vec = v0[0]
        device = base_vec.device
        dtype = base_vec.dtype
        v_prev = [torch.zeros_like(x) for x in v0]
        v_curr = self._scale(v0, self._norm(v0))

        alphas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        beta_prev = torch.tensor(0.0, device=device, dtype=dtype)
        mu_prev: Optional[float] = None
        mu_curr = self.reg

        for j in range(self.k):
            w = Hv_fn(v_curr)
            alpha = self._dot(v_curr, w)
            alphas.append(alpha)

            r = [w_i - alpha * v_i - beta_prev * vp_i for w_i, v_i, vp_i in zip(w, v_curr, v_prev)]
            beta = self._norm(r)

            m = len(alphas)
            T = torch.zeros((m, m), device=device, dtype=dtype)
            for i in range(m):
                T[i, i] = alphas[i]
                if i < m - 1 and i < len(betas):
                    T[i, i + 1] = betas[i]
                    T[i + 1, i] = betas[i]

            eigvals = torch.linalg.eigvalsh(T)
            mu_curr = float(eigvals[0].detach().item())
            if mu_prev is not None and j >= 3 and abs(mu_curr - mu_prev) < self.tol * max(abs(mu_curr), 1e-6):
                break
            mu_prev = mu_curr

            if beta < 1e-12:
                break

            betas.append(beta)
            v_prev = v_curr
            v_curr = self._scale(r, beta)
            beta_prev = beta

        return max(mu_curr + self.reg, 1e-8)

    def estimate(self, Hv_fn: HvFunction, v: VectorList) -> tuple[float, float]:
        L_local = self._power_iteration_L(Hv_fn, v)
        mu_local = self._adaptive_lanczos_mu(Hv_fn, v)

        L_global = self._all_reduce_scalar_max(L_local)
        mu_global = self._all_reduce_scalar_max(mu_local)

        if self.L_ema is None or self.mu_ema is None:
            self.L_ema = L_global
            self.mu_ema = mu_global
        else:
            self.L_ema = self.ema_alpha * self.L_ema + (1.0 - self.ema_alpha) * L_global
            self.mu_ema = self.ema_alpha * self.mu_ema + (1.0 - self.ema_alpha) * mu_global

        L_safe = max(self.L_ema, L_global) * self.safety
        mu_safe = max(self.mu_ema, mu_global)
        return L_safe, mu_safe

    def compute_adaptive_lr(self, L: float, mu: float, max_lr: float = 3.0, min_lr: float = 1e-6) -> float:
        eta_opt = 1.0 / (L + mu + self.reg)
        eta_bar = 1.9 / max(L, 1e-10)
        return float(np.clip(eta_opt, min_lr, min(max_lr, eta_bar)))

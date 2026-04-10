import numpy as np
from typing import Callable, Tuple


class OptimizerCertificate:
    """
    Gauss-Newton Hessian, PL condition, and Lyapunov stability monitors.
    Matches Hessian.txt §4 & Lyapunov f.txt §2.
    """

    def __init__(self, mu_lb: float = 2.4e-2, L_ub: float = 5.3e-1):
        self.mu_lb = mu_lb  # PL constant lower bound
        self.L_ub = L_ub  # Lipschitz upper bound

    @staticmethod
    def gauss_newton_hessian(J: np.ndarray, W: np.ndarray, Lambda: float = 0.0) -> np.ndarray:
        """H_GN = J^T W J + Λ I"""
        return J.T @ W @ J + Lambda * np.eye(J.shape[1])

    @staticmethod
    def hessian_vector_product(grad_fn: Callable, theta: np.ndarray, v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Numerical H*v via finite differences"""
        g_plus = grad_fn(theta + eps * v)
        g = grad_fn(theta)
        return (g_plus - g) / eps

    def check_pl_condition(self, grad_norm_sq: float, loss_gap: float) -> Tuple[bool, float]:
        """Verify 0.5||∇L||² ≥ μ(L - L*)"""
        if loss_gap < 1e-12:
            return True, 0.0
        mu_est = 0.5 * grad_norm_sq / loss_gap
        return mu_est >= self.mu_lb, mu_est

    def lyapunov_decay_check(self, V_opt: float, gamma: float, dt: float) -> float:
        """V_opt(t) ≤ V_opt(0) exp(-γ t)"""
        return V_opt * np.exp(-gamma * dt)

    def adaptive_stepsize(self, mu: float, L: float) -> float:
        """η_opt = 1/(L + μ) clipped to stability bound 2/L"""
        eta_bar = 2.0 / max(L, 1e-10)
        eta_opt = 1.0 / (L + max(mu, 1e-10))
        return np.clip(eta_opt, 1e-5, eta_bar * 0.9)

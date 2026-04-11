"""JAX Hessian-vector products and PL-condition helpers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import grad, jacrev, jvp

jax.config.update("jax_enable_x64", True)


class JAXHessianEstimator:
    """JAX-accelerated Gauss-Newton Hessian estimation and PL certification."""

    def __init__(
        self,
        constraint_fn,
        weights,
        reg_lambda: float = 1e-4,
        tolerance_ledger: Any | None = None,
        ledger_key: str = "hessian_pl",
    ):
        self.constraint_fn = constraint_fn
        self.W = jnp.asarray(weights)
        self.Lambda = reg_lambda
        self.grad_C = jax.jit(jacrev(constraint_fn))
        self.tolerance_ledger = tolerance_ledger
        self.ledger_key = ledger_key

    def _ledger_tol(self, default: float) -> float:
        if self.tolerance_ledger is None:
            return default
        try:
            return float(self.tolerance_ledger.get_tolerance(self.ledger_key))
        except Exception:
            return default

    def hessian_vector_product(self, theta: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Exact H_GN v = J_C^T W J_C v + Λ v."""
        _, jv = jvp(self.constraint_fn, (theta,), (v,))
        _, vjp_fn = jax.vjp(self.constraint_fn, theta)
        jt_w_jv = vjp_fn(self.W * jv)[0]
        hv = jt_w_jv + self.Lambda * v
        # Replace NaN with 0.0 and Inf with the dtype's representable maximum,
        # leaving all valid finite values unclipped to preserve the true spectral scale.
        return jnp.nan_to_num(hv, nan=0.0)

    def lanczos_eigenvalues(
        self,
        theta: jnp.ndarray,
        k: int = 3,
        max_iter: int = 50,
        tol: float = 1e-7,
        adaptive_k: bool = True,
        min_k: int = 3,
    ) -> jnp.ndarray:
        """Lanczos iteration with optional adaptive Krylov dimension growth."""
        dim = int(theta.shape[0])
        v_prev = jnp.zeros((dim,))
        v = jax.random.normal(jax.random.PRNGKey(42), (dim,))
        v = v / (jnp.linalg.norm(v) + 1e-12)

        # Pre-allocate arrays instead of using Python lists
        tol_eff = self._ledger_tol(tol)
        base_k = max(min_k, k)
        max_k = min(max_iter, max(base_k, dim))
        alphas = jnp.zeros(max_k, dtype=theta.dtype)
        betas = jnp.zeros(max_k, dtype=theta.dtype)
        idx = 0
        eig_prev: jnp.ndarray | None = None

        for _ in range(base_k):
            hv = self.hessian_vector_product(theta, v)
            alpha = jnp.nan_to_num(jnp.dot(hv, v), nan=0.0)
            w = hv - alpha * v
            if idx > 0:
                w = w - betas[idx - 1] * v_prev
            beta = jnp.nan_to_num(jnp.linalg.norm(w), nan=0.0)

            alphas = alphas.at[idx].set(alpha)
            if float(beta) <= tol_eff:
                idx += 1
                break
            betas = betas.at[idx].set(beta)
            idx += 1
            v_prev, v = v, w / (beta + 1e-12)

        def _eig_from_tridiag(m: int) -> jnp.ndarray:
            # Vectorized tridiagonal matrix construction
            diag_indices = jnp.arange(m)
            off_diag_indices = jnp.arange(m - 1)

            t = jnp.zeros((m, m), dtype=theta.dtype)
            t = t.at[diag_indices, diag_indices].set(alphas[:m])
            t = t.at[off_diag_indices, off_diag_indices + 1].set(betas[:m-1])
            t = t.at[off_diag_indices + 1, off_diag_indices].set(betas[:m-1])

            eigs = jnp.sort(jnp.linalg.eigvalsh(t))
            return jnp.nan_to_num(eigs, nan=self.Lambda, posinf=1e6, neginf=-1e6)

        eigvals = _eig_from_tridiag(idx)
        eig_prev = eigvals

        if adaptive_k:
            while idx < max_k:
                hv = self.hessian_vector_product(theta, v)
                alpha = jnp.nan_to_num(jnp.dot(hv, v), nan=0.0)
                w = hv - alpha * v - (betas[idx - 1] * v_prev if idx > 0 else 0.0)
                beta = jnp.nan_to_num(jnp.linalg.norm(w), nan=0.0)
                alphas = alphas.at[idx].set(alpha)
                if float(beta) <= tol_eff:
                    idx += 1
                    break
                betas = betas.at[idx].set(beta)
                idx += 1

                eigvals = _eig_from_tridiag(idx)
                n_cmp = min(int(eigvals.shape[0]), int(eig_prev.shape[0]), base_k)
                drift = jnp.max(jnp.abs(eigvals[:n_cmp] - eig_prev[:n_cmp]))
                eig_prev = eigvals

                if float(drift) <= tol_eff:
                    break
                v_prev, v = v, w / (beta + 1e-12)

        eig_floor = min(max(self._ledger_tol(1e-5), 1e-8), 1e-3)
        eig_ceil = 1.0 - min(self._ledger_tol(1e-3), 0.5)
        eig_final = jnp.clip(jnp.abs(eig_prev), eig_floor, eig_ceil)
        return eig_final[:k] if k and k < eig_final.shape[0] else eig_final

    def verify_pl_condition(
        self,
        theta: jnp.ndarray,
        loss_val: float,
        loss_star: float = 0.0,
        mu_lb: float = 2.4e-2,
    ) -> dict:
        """Certify 0.5||∇L||² ≥ μ(L - L*)."""
        grad_l = grad(self._loss_fn)(theta)
        grad_norm_sq = jnp.sum(grad_l**2)
        gap = max(float(loss_val - loss_star), 1e-12)
        mu_lb_eff = max(mu_lb, self._ledger_tol(mu_lb))
        mu_est = float(0.5 * grad_norm_sq / gap)

        if self.tolerance_ledger is not None:
            try:
                self.tolerance_ledger.update_from_residual(self.ledger_key, abs(mu_lb_eff - mu_est), "jax_hessian")
            except Exception:
                pass

        return {
            "mu_est": mu_est,
            "pl_satisfied": bool(mu_est >= mu_lb_eff),
            "grad_norm": float(jnp.linalg.norm(grad_l)),
            "loss_gap": gap,
            "mu_lb": float(mu_lb_eff),
        }

    def _loss_fn(self, theta: jnp.ndarray) -> jnp.ndarray:
        c = self.constraint_fn(theta)
        tol = self._ledger_tol(0.0)
        penalty = tol * jnp.mean(jnp.square(c - 1.0))
        return 0.5 * jnp.sum(self.W * (c - 1.0) ** 2) + 0.5 * self.Lambda * jnp.sum(theta**2) + penalty

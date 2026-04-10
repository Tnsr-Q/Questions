"""Quantum-accelerated Hessian estimation interfaces.

This module keeps the qjax dependency optional so the engine can run in
pure-classical environments while still exposing a stable integration contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Optional

import jax.numpy as jnp


class QJAXUnavailableError(RuntimeError):
    """Raised when quantum Hessian estimation is requested without a qjax runtime."""


@dataclass
class QJAXHessianEstimator:
    """Estimate dominant Hessian eigen-structure through a qjax backend.

    The implementation intentionally uses capability checks instead of assuming a
    specific qjax API version so that simulator and hardware providers can plug
    into the same contract.
    """

    mesh: Any
    runtime: Optional[ModuleType] = None

    def __post_init__(self) -> None:
        if self.runtime is None:
            try:
                import qjax as runtime  # type: ignore
            except ModuleNotFoundError as exc:
                raise QJAXUnavailableError(
                    "qjax is not installed; quantum Hessian estimation is unavailable"
                ) from exc
            self.runtime = runtime

    def estimate_mu_quantum(
        self,
        loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
        theta: jnp.ndarray,
        v_vector: jnp.ndarray,
        precision_bits: int = 8,
    ) -> float:
        """Estimate a PL lower-bound proxy (mu) from quantum phase estimation.

        If the runtime does not expose the expected primitives we raise a clear
        error so callers can fall back to classical Lanczos.
        """

        runtime = self.runtime
        if runtime is None:
            raise QJAXUnavailableError("Missing qjax runtime")

        if not hasattr(runtime, "phase_estimation") or not hasattr(runtime, "unitary_hvp"):
            raise QJAXUnavailableError(
                "qjax runtime missing required APIs: phase_estimation and/or unitary_hvp"
            )

        state = self._prepare_state(v_vector)

        def hessian_oracle(time: float):
            return runtime.unitary_hvp(loss_fn, theta, time)

        eigenvalue = runtime.phase_estimation(
            hessian_oracle,
            state,
            precision_bits=precision_bits,
        )
        return float(jnp.abs(eigenvalue))

    @staticmethod
    def _prepare_state(vector: jnp.ndarray) -> jnp.ndarray:
        norm = jnp.linalg.norm(vector)
        if float(norm) == 0.0:
            raise ValueError("v_vector must be non-zero for state preparation")
        return vector / norm

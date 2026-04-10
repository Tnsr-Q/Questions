"""RL conjecture-refinement loop for theory-space exploration."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Protocol

import numpy as np


class VerificationEngine(Protocol):
    def verify(self, theorem: np.ndarray): ...


class LedgerLike(Protocol):
    current_tol: float

    def complexity_penalty(self, theta: np.ndarray) -> float: ...

    def is_optimal(self, certificate: Any) -> bool: ...


@dataclass
class TheorySpaceEnv:
    """Simple environment for conjecture-refinement over continuous couplings."""

    engine: VerificationEngine
    ledger: LedgerLike
    dim: int = 3
    init_scale: float = 0.1

    def __post_init__(self) -> None:
        self.theta = np.zeros(self.dim, dtype=float)

    def reset(self, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0.0, self.init_scale, size=self.dim)
        return self.theta.copy()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != self.theta.shape:
            raise ValueError(f"Action shape {action.shape} does not match state shape {self.theta.shape}")

        self.theta = self.theta + action
        cert = self.engine.verify(theorem=self.theta)

        reward = 0.0
        if getattr(cert, "status", None) == "VERIFIED":
            reward += 10.0
            reward += float(max(self.ledger.current_tol, 1e-12) ** -1)

        reward -= float(self.ledger.complexity_penalty(self.theta))
        done = bool(getattr(cert, "status", None) == "VERIFIED" and self.ledger.is_optimal(cert))

        return self.theta.copy(), float(reward), done, {"certificate": cert}


class _FallbackPPO:
    """Minimal fallback used when stable_baselines3 is unavailable."""

    def __init__(self, policy: str, env: TheorySpaceEnv, verbose: int = 0, device: str = "cpu"):
        _ = (policy, verbose, device)
        self.env = env
        self.policy = self

    def learn(self, total_timesteps: int = 1000):
        _ = total_timesteps

    def predict(self, obs, deterministic: bool = True):
        _ = deterministic
        action = np.zeros_like(obs, dtype=float)
        return action, None


class AletheiaAgent:
    """PPO-based agent that searches for higher-quality verified theories."""

    def __init__(
        self,
        env: TheorySpaceEnv,
        topology: Any,
        ppo_cls: Optional[type] = None,
    ):
        self.env = env
        self.topology = topology

        if ppo_cls is None:
            try:
                from stable_baselines3 import PPO as ppo_cls  # type: ignore
            except ModuleNotFoundError:
                ppo_cls = _FallbackPPO

        device = "cuda" if getattr(topology, "n_devices", 0) > 0 else "cpu"
        self.model = ppo_cls("MlpPolicy", env, verbose=1, device=device)

    def refine_theory(self, steps: int = 1000):
        self.model.learn(total_timesteps=int(steps))
        obs = self.env.reset()
        action, _ = self.model.predict(obs, deterministic=True)
        refined_theta, _, _, _ = self.env.step(action)
        return np.asarray(refined_theta, dtype=float)


def make_certificate(status: str = "PENDING") -> Any:
    """Small helper for tests and notebooks."""
    return SimpleNamespace(status=status)

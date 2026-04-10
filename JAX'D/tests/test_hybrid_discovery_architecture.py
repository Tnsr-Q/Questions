from types import SimpleNamespace

import numpy as np

from src.rl_conjecture_loop import AletheiaAgent, TheorySpaceEnv, make_certificate
from src.unified_topology import HybridTopologyManager


class _Ledger:
    current_tol = 1e-2

    @staticmethod
    def complexity_penalty(theta):
        return float(np.linalg.norm(theta))

    @staticmethod
    def is_optimal(certificate):
        return certificate.status == "VERIFIED"


class _Engine:
    def __init__(self, status="VERIFIED"):
        self.status = status

    def verify(self, theorem):
        _ = theorem
        return make_certificate(self.status)


def test_theory_space_env_rewards_verified_points():
    env = TheorySpaceEnv(engine=_Engine("VERIFIED"), ledger=_Ledger(), dim=3)
    env.reset(seed=7)

    _, reward, done, info = env.step(np.array([0.01, -0.02, 0.03]))

    assert reward > 10.0
    assert done is True
    assert info["certificate"].status == "VERIFIED"


def test_hybrid_topology_defaults_to_classical_without_qjax():
    manager = HybridTopologyManager(qpu_count=1)
    assert manager.select_hessian_backend(condition_number=1_000.0) == "classical"


def test_hybrid_topology_routes_to_quantum_with_runtime_stub():
    class FakeQJAX:
        @staticmethod
        def list_devices():
            return ["qpu0", "qpu1"]

        @staticmethod
        def Mesh(devices, axes):
            return {"devices": devices, "axes": axes}

    manager = HybridTopologyManager(qpu_count=1, qjax_runtime=FakeQJAX)
    assert manager.select_hessian_backend(condition_number=1_000.0) == "quantum"


def test_aletheia_agent_uses_injected_ppo_class():
    class DummyPPO:
        def __init__(self, policy, env, verbose=0, device="cpu"):
            _ = (policy, verbose, device)
            self.env = env

        def learn(self, total_timesteps):
            self.timesteps = total_timesteps

        def predict(self, obs, deterministic=True):
            _ = deterministic
            return np.ones_like(obs), None

    env = TheorySpaceEnv(engine=_Engine("PENDING"), ledger=_Ledger(), dim=3)
    topology = SimpleNamespace(n_devices=0)

    agent = AletheiaAgent(env=env, topology=topology, ppo_cls=DummyPPO)
    best = agent.refine_theory(steps=5)

    assert np.allclose(best, np.ones(3))

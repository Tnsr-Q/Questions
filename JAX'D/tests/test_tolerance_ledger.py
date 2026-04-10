from src.tolerance.dynamic_ledger import DynamicToleranceLedger, PhysicsRegime
from src.tolerance.regime_detector import RegimeDetector


def test_adaptive_bounds():
    ledger = DynamicToleranceLedger(base_config_path="configs/tolerance_priors.yaml")
    tol = ledger.get_tolerance("hessian_pl")
    assert 1e-3 <= tol <= 5e-2

    ledger.update_from_residual("hessian_pl", 1e-4, "test")
    assert ledger.get_tolerance("hessian_pl") <= tol

    for _ in range(50):
        ledger.update_from_residual("hessian_pl", 1e-6, "test")
    assert ledger.get_tolerance("hessian_pl") >= 1e-3


def test_freeze_mode():
    ledger = DynamicToleranceLedger(base_config_path="configs/tolerance_priors.yaml", freeze_mode=True)
    base = ledger.get_tolerance("rge_atol")
    ledger.update_from_residual("rge_atol", 1e-15, "frozen_test")
    assert ledger.get_tolerance("rge_atol") == base


def test_regime_detection():
    detector = RegimeDetector()
    state = {"energy_scale": 2.4e22, "f2": 1e-8, "hessian_PL_mu": 0.025, "hessian": True}
    reg = detector.classify(state, {})
    assert reg in {PhysicsRegime.UV_FAKEON, PhysicsRegime.HESSIAN_PL}

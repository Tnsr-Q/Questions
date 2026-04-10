import numpy as np

from src.validators import (
    check_pl_condition,
    rge_2loop_stability,
    thermal_consistency_check,
    verify_lyapunov_decay,
)


def test_hessian_pl_constant():
    mu_lb, loss_gap = 2.4e-2, 1e-4
    grad_norm_sq = 0.5 * (mu_lb * 2.1 * loss_gap)
    assert check_pl_condition(grad_norm_sq, loss_gap, mu_lb)


def test_lyapunov_rg_decay():
    assert verify_lyapunov_decay(V0=1.0, rate=1.8e-2, t=20.0, threshold=0.7)


def test_rge_thermal_bounds():
    lam_min, mu_min = 3.2e-3, 1e11
    _, _, passed = rge_2loop_stability(lam_min, mu_min)
    assert passed
    assert thermal_consistency_check(T_reh=1e15, M2=2.4e23)

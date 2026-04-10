import torch

from src.spectral.robust_estimator import RobustSpectralEstimator


def test_estimator_bounds():
    hessian = torch.diag(torch.tensor([1e-4, 0.5, 1.2], dtype=torch.float64))

    def hv(v_list):
        return [hessian @ v for v in v_list]

    estimator = RobustSpectralEstimator(k_lanczos=8, power_iters=8, safety_factor=1.15, reg_floor=1e-8)
    vecs = [torch.randn(3, dtype=torch.float64)]

    L_safe, mu_safe = estimator.estimate(hv, vecs)

    assert L_safe >= 1.2
    assert mu_safe <= 1e-4 + 5e-5
    assert estimator.compute_adaptive_lr(L_safe, mu_safe) <= 1.9 / L_safe

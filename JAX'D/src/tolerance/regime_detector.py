from typing import Any, Dict

from .dynamic_ledger import PhysicsRegime


class RegimeDetector:
    """Classify active physics regime for tolerance selection."""

    def __init__(self, M2_GeV: float = 2.4e23, f2_target: float = 1e-8):
        self.M2 = M2_GeV
        self.f2_target = f2_target

    def classify(self, solver_state: Dict[str, Any], residuals: Dict[str, float]) -> PhysicsRegime:
        mu = float(solver_state.get("energy_scale", 173.1))
        f2 = float(solver_state.get("f2", 1.0))
        stiffness = float(solver_state.get("jacobian_cond", 1.0))
        hessian_mu = float(solver_state.get("hessian_PL_mu", 1e-4))

        if mu >= 0.1 * self.M2 and abs(f2 - self.f2_target) < 1e-6:
            return PhysicsRegime.UV_FAKEON

        if mu < 1e3 and residuals and all(abs(float(r)) < 1e-4 for r in residuals.values()):
            return PhysicsRegime.IR_SM

        if stiffness > 50.0 or float(solver_state.get("step_size", 1.0)) < 1e-6:
            return PhysicsRegime.STIFF_ODE

        if hessian_mu > 1e-3 and "hessian" in solver_state:
            return PhysicsRegime.HESSIAN_PL

        if "inelasticity_residual" in residuals or "crossing_residual" in residuals:
            return PhysicsRegime.NONPERT_UNITARITY

        return PhysicsRegime.DEFAULT

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Callable, Dict, List, Tuple

import numpy as np
import sympy as sp

from src.proto.constraint_schema import AssumptionTag
from src.proto.registry import PredicateRegistry

log = logging.getLogger("QUFT_TheorySpace")


@dataclass
class TheoryHypothesis:
    """A candidate physical theory with parameters and predictions."""

    name: str
    lagrangian: sp.Expr
    parameters: Dict[str, float]
    symmetries: List[str]
    predictions: Dict[str, Callable]
    assumptions: List[AssumptionTag] = field(default_factory=list)
    confidence: float = 0.0

    def evaluate_prediction(self, observable: str, **kwargs) -> float:
        if observable not in self.predictions:
            raise ValueError(f"Unknown prediction: {observable}")
        return self.predictions[observable](**kwargs, **self.parameters)


class TheorySpaceExplorer:
    """Hybrid symbolic-numeric theory-space exploration engine."""

    def __init__(self, registry: PredicateRegistry, parameter_bounds: Dict[str, Tuple[float, float]]):
        self.registry = registry
        self.bounds = parameter_bounds
        self.discovered_theories: List[TheoryHypothesis] = []

    def generate_candidate(
        self, base_lagrangian: sp.Expr, parameter_samples: Dict[str, np.ndarray]
    ) -> TheoryHypothesis:
        params = {key: float(np.random.choice(values)) for key, values in parameter_samples.items()}
        predictions = self._derive_predictions(base_lagrangian, params)

        return TheoryHypothesis(
            name=f"candidate_{len(self.discovered_theories)}",
            lagrangian=base_lagrangian,
            parameters=params,
            symmetries=self._extract_symmetries(base_lagrangian),
            predictions=predictions,
            assumptions=self._infer_assumptions(base_lagrangian, params),
        )

    def _derive_predictions(self, lagrangian: sp.Expr, params: Dict[str, float]) -> Dict[str, Callable]:
        predictions: Dict[str, Callable] = {}

        couplings = [name for name in params if name in {"f2", "lambda_H", "y_t", "g1", "g2", "g3"}]
        for coupling in couplings:
            predictions[f"beta_{coupling}"] = lambda mu, c=coupling, p=params: float(p[c]) * np.log(max(mu, 1e-30))

        predictions["amplitude_2to2"] = lambda s, t, u: self._compute_amplitude(lagrangian, s, t, u, params)
        return predictions

    def _compute_amplitude(
        self, lagrangian: sp.Expr, s: float, t: float, u: float, params: Dict[str, float]
    ) -> complex:
        _ = (lagrangian, t, u)
        m_pl = 2.435e18
        m2 = params.get("M2", 2.4e23)

        graviton_term = 8 * np.pi * s**2 / (m_pl**2) * (1 / (s + 1e-15j))
        fakeon_term = -8 * np.pi * s**2 / (m_pl**2) * (1 / (s - m2**2 + 1e-15j))
        return graviton_term + fakeon_term

    def _extract_symmetries(self, lagrangian: sp.Expr) -> List[str]:
        lag = str(lagrangian)
        symmetries: List[str] = []

        if any(token in lag for token in ["F_mu_nu", "D_mu", "A_mu"]):
            symmetries.append("GaugeInvariant")
        if "g_mu_nu" in lag or "eta_mu_nu" in lag:
            symmetries.append("LorentzInvariant")
        if "R**2" in lag and "m**2" not in lag:
            symmetries.append("ScaleInvariant")

        return symmetries

    def _infer_assumptions(self, lagrangian: sp.Expr, params: Dict[str, float]) -> List[AssumptionTag]:
        assumptions: List[AssumptionTag] = []
        lag = str(lagrangian).lower()

        if params.get("f2", 1.0) < 1.0:
            assumptions.append(AssumptionTag.A1_PERTURBATIVE)
        if "fakeon" in lag or params.get("M2") is not None:
            assumptions.append(AssumptionTag.A2_FAKEON_VALID)
        if "palatini" in lag or "connection" in lag:
            assumptions.append(AssumptionTag.A3_PALATINI_COMPAT)
        if "lambda_HS" in params and params["lambda_HS"] < 1e-30:
            assumptions.append(AssumptionTag.A5_PORTAL_DOMINANCE)

        return assumptions

    def validate_against_predicates(self, hypothesis: TheoryHypothesis) -> Dict[str, bool]:
        results: Dict[str, bool] = {}

        for pred_id in self.registry.predicate_ids():
            predicate = self.registry.get_latest(pred_id)
            if not predicate:
                continue

            if predicate.predicate_id in hypothesis.predictions:
                try:
                    result = hypothesis.evaluate_prediction(predicate.predicate_id)
                    residual = abs(result - 1.0)
                    results[pred_id] = residual <= predicate.tolerance
                except Exception as exc:
                    log.warning("Failed to evaluate %s: %s", pred_id, exc)
                    results[pred_id] = False
            else:
                results[pred_id] = set(predicate.assumptions).issubset(set(hypothesis.assumptions))

        return results

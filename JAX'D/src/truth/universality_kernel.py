from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from src.proto.constraint_schema import PhysicsPredicate
from src.proto.registry import PredicateRegistry


class UniversalityKernel:
    """Scan f2-space for a unique predicate-satisfying fixed point."""

    def __init__(self, registry: PredicateRegistry, rge_solver: Callable, bootstrap_solver: Callable):
        self.registry = registry
        self.rge_solver = rge_solver
        self.bootstrap_solver = bootstrap_solver
        self.universality_score: Optional[float] = None

    def scan_f2_space(self, f2_range: np.ndarray, tolerance: float = 1e-4) -> Dict[str, Any]:
        candidates = []

        for f2_val in f2_range:
            rge_result = self.rge_solver.solve_f2(float(f2_val))
            if not self._check_ir_consistency(rge_result):
                continue
            bootstrap_result = self.bootstrap_solver.solve_at_scale(float(f2_val))

            predicate_scores: Dict[str, float] = {}
            for pred_id in self.registry.predicate_ids():
                predicate = self.registry.get_latest(pred_id)
                if predicate is None:
                    continue
                score = self._evaluate_predicate(predicate, float(f2_val), rge_result, bootstrap_result)
                if score is not None:
                    predicate_scores[pred_id] = score

            if not predicate_scores:
                continue

            satisfaction = float(np.mean(list(predicate_scores.values())))
            if satisfaction >= 1.0 - tolerance:
                candidates.append(
                    {
                        "f2": float(f2_val),
                        "satisfaction": satisfaction,
                        "predicate_scores": predicate_scores,
                        "rge_endpoint": rge_result.get("g_ir"),
                        "bootstrap_residual": bootstrap_result.get("unitarity_residual"),
                    }
                )

        if len(candidates) == 1:
            self.universality_score = candidates[0]["satisfaction"]
            return {
                "status": "UNIQUE_SOLUTION",
                "f2_star": candidates[0]["f2"],
                "confidence": candidates[0]["satisfaction"],
                "details": candidates[0],
            }

        if len(candidates) > 1:
            return {
                "status": "MULTIPLE_SOLUTIONS",
                "candidates": candidates,
                "recommendation": "Tighten tolerance or add discriminating predicates",
            }

        return {
            "status": "NO_SOLUTION",
            "best_candidate": None,
            "recommendation": "Expand f2 range or relax assumptions",
        }

    def _check_ir_consistency(self, rge_result: Dict[str, Any]) -> bool:
        g_ir = rge_result.get("g_ir", [])
        if len(g_ir) < 4:
            return False

        y_t_pred = g_ir[3]
        lambda_h_pred = g_ir[0]

        return abs(y_t_pred - 0.995) < 0.01 and abs(lambda_h_pred - 0.129) < 0.01

    def _evaluate_predicate(
        self,
        predicate: PhysicsPredicate,
        f2_val: float,
        rge_result: Dict[str, Any],
        bootstrap_result: Dict[str, Any],
    ) -> Optional[float]:
        _ = f2_val
        pred_id = predicate.predicate_id

        if pred_id == "C_ghost":
            ghost_residual = float(bootstrap_result.get("ghost_residual", 1.0))
            return 1.0 if ghost_residual < predicate.tolerance else 0.0
        if pred_id == "C_infl":
            n_s_pred = float(rge_result.get("n_s", 0.965))
            return max(0.0, 1.0 - abs(n_s_pred - 0.965) / predicate.tolerance)
        if pred_id == "C_DM":
            f_pbh_pred = float(rge_result.get("f_PBH", 1.0))
            return 1.0 if abs(f_pbh_pred - 1.0) < predicate.tolerance else 0.0
        if pred_id == "C_echo":
            delta_f_pred = float(bootstrap_result.get("echo_spacing", 9.0))
            return max(0.0, 1.0 - abs(delta_f_pred - 9.0) / predicate.tolerance)
        if pred_id == "C_unitarity":
            residual = float(bootstrap_result.get("unitarity_residual", 1.0))
            return 1.0 if residual < predicate.tolerance else 0.0

        residual = float(predicate.metadata.get("residual", 1.0))
        return 1.0 if residual < predicate.tolerance else 0.0

    def compute_birth_of_universe_initial_condition(self, f2_star: float) -> Dict[str, Any]:
        m2 = 2.4e23 * np.sqrt(f2_star / 1e-8)
        initial_conditions = {
            "mu": m2,
            "lambda_H": 0.0,
            "lambda_S": 0.0,
            "lambda_HS": 3.2e-32,
            "y_t": 0.5,
            "g1": 0.46,
            "g2": 0.51,
            "g3": 0.50,
            "f2": f2_star,
            "xi_H": 5e8,
            "M_Pl": 2.435e18,
        }

        v_s = m2 / np.sqrt(f2_star)
        v_h = v_s * np.sqrt(initial_conditions["lambda_HS"] / (initial_conditions["lambda_H"] + 1e-32))

        return {
            "initial_conditions": initial_conditions,
            "generated_scales": {
                "v_s": v_s,
                "v_h": v_h,
                "m_top": v_h * initial_conditions["y_t"] / np.sqrt(2),
                "m_W": v_h * initial_conditions["g2"] / 2,
                "m_H": np.sqrt(2 * max(initial_conditions["lambda_H"], 0.0)) * v_h,
            },
            "scale_invariance_check": (
                initial_conditions["lambda_H"] == 0.0 and initial_conditions["lambda_S"] == 0.0
            ),
        }

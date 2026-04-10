import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional dependency for parquet audit output
    pa = None
    pq = None

log = logging.getLogger("QUFT_ToleranceLedger")


class PhysicsRegime(str, Enum):
    UV_FAKEON = "UV_fakeon"
    IR_SM = "IR_standard_model"
    STIFF_ODE = "stiff_RGE"
    NONPERT_UNITARITY = "nonperturbative_Smatrix"
    HESSIAN_PL = "Hessian_PL_certification"
    DEFAULT = "default"


@dataclass
class ToleranceConfig:
    base_tol: float
    min_tol: float
    max_tol: float
    adaptation_rate: float = 0.1
    reference_residual: float = 1e-8
    regime: PhysicsRegime = PhysicsRegime.DEFAULT
    last_updated: float = 0.0


class DynamicToleranceLedger:
    """Self-calibrating tolerance manager with bounded feedback and audit trail."""

    def __init__(
        self,
        base_config_path: str = "configs/tolerance_priors.yaml",
        audit_dir: str = "logs/tolerance_audit",
        freeze_mode: bool = False,
    ):
        self.freeze = freeze_mode
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._ledgers: Dict[str, ToleranceConfig] = {}
        self._ewma_residuals: Dict[str, float] = {}
        self._audit_buffer: list[Dict[str, Any]] = []
        self._load_base_config(base_config_path)

    def _load_base_config(self, path: str) -> None:
        if not os.path.exists(path):
            log.warning("Base config %s not found. Using empty ledger.", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        for key, cfg in raw.items():
            tol_cfg = ToleranceConfig(
                base_tol=float(cfg.get("base_tol", 1e-8)),
                min_tol=float(cfg.get("min_tol", 1e-14)),
                max_tol=float(cfg.get("max_tol", 1e-3)),
                adaptation_rate=float(cfg.get("adaptation_rate", 0.1)),
                reference_residual=float(cfg.get("reference_residual", 1e-8)),
                regime=PhysicsRegime(cfg.get("regime", "default")),
            )
            self._ledgers[key] = tol_cfg
            self._ewma_residuals[key] = tol_cfg.reference_residual

    def get_tolerance(self, key: str, regime: Optional[PhysicsRegime] = None) -> float:
        cfg = self._ledgers.get(key)
        if cfg is None:
            raise KeyError(f"Tolerance key '{key}' not registered")
        if regime is not None:
            cfg.regime = regime
        return cfg.base_tol

    def update_from_residual(self, key: str, residual: float, solver_id: str = "unknown") -> float:
        """tol_{k+1} = clip(tol_k * exp(-alpha*(r_ewma/r_ref - 1)), [min_tol, max_tol])."""
        cfg = self._ledgers.get(key)
        if cfg is None:
            raise KeyError(f"Tolerance key '{key}' not registered")

        if self.freeze:
            self._audit_buffer.append(
                {
                    "timestamp": time.time(),
                    "key": key,
                    "solver": solver_id,
                    "residual": float(residual),
                    "ewma": self._ewma_residuals.get(key, cfg.reference_residual),
                    "tol_old": cfg.base_tol,
                    "tol_new": cfg.base_tol,
                    "regime": cfg.regime.value,
                    "freeze": True,
                }
            )
            return cfg.base_tol

        old_ewma = self._ewma_residuals.get(key, cfg.reference_residual)
        new_ewma = (1.0 - cfg.adaptation_rate) * old_ewma + cfg.adaptation_rate * abs(float(residual))
        self._ewma_residuals[key] = new_ewma

        ratio = new_ewma / cfg.reference_residual
        scale = float(np.exp(cfg.adaptation_rate * (ratio - 1.0)))
        new_tol = float(np.clip(cfg.base_tol * scale, cfg.min_tol, cfg.max_tol))

        self._audit_buffer.append(
            {
                "timestamp": time.time(),
                "key": key,
                "solver": solver_id,
                "residual": float(residual),
                "ewma": new_ewma,
                "tol_old": cfg.base_tol,
                "tol_new": new_tol,
                "regime": cfg.regime.value,
                "freeze": False,
            }
        )

        cfg.base_tol = new_tol
        cfg.last_updated = time.time()
        return new_tol

    def flush_audit(self) -> None:
        if not self._audit_buffer:
            return
        ts = int(time.time())
        if pa is not None and pq is not None:
            table = pa.Table.from_pylist(self._audit_buffer)
            pq.write_table(table, self.audit_dir / f"audit_{ts}.parquet")
        else:
            with open(self.audit_dir / f"audit_{ts}.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(self._audit_buffer, f)
        log.info("Flushed %d tolerance updates to %s", len(self._audit_buffer), self.audit_dir)
        self._audit_buffer = []

    def export_snapshot(self) -> Dict[str, Any]:
        return {
            key: {
                "base_tol": cfg.base_tol,
                "min_tol": cfg.min_tol,
                "max_tol": cfg.max_tol,
                "regime": cfg.regime.value,
                "last_updated": cfg.last_updated,
                "ewma_residual": self._ewma_residuals.get(key, cfg.reference_residual),
            }
            for key, cfg in self._ledgers.items()
        }

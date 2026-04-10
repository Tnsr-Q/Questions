
load_and_freeze() + main() into a drop-in implementation that does: schema validate → canonicalize → freeze hash → deterministic RNG → manifest in/out (keeping your overall structure intact).




# File: research-station/echo_search/pipeline.py
"""
RIC Research Station — Frozen Kernel (Go/Nix-friendly)

Key guarantees:
- Strict config schema validation (JSON Schema)
- Canonical config serialization -> freeze hash
- Deterministic RNG derived from (cfg.seed, freeze_hash)
- Optional input hashing / enforcement
- Deterministic provenance output: results.json + run_manifest_out.json
- No interactive I/O; pure CLI inputs -> file outputs

Usage (direct):
  python3 -m echo_search.pipeline --config configs/RIC-LIGO-O4-001.yaml \
    --schema configs/ric_echo_search.schema.json \
    --data-h1 /path/to/H1.hdf5 --data-l1 /path/to/L1.hdf5 \
    --output results.json --manifest-out run_manifest_out.json

Usage (recommended with orchestrator manifest):
  python3 -m echo_search.pipeline --manifest-in run_manifest_in.json \
    --schema configs/ric_echo_search.schema.json \
    --output results.json --manifest-out run_manifest_out.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# optional deps (present in your flake)
import yaml
from jsonschema import Draft202012Validator

# ----------------------------
# Canonicalization & Hashing
# ----------------------------

def _reject_nonfinite_floats(obj: Any) -> Any:
    """
    Ensure JSON-safe & canonical: forbid NaN/Inf and non-JSON scalars.
    """
    if isinstance(obj, float):
        if not np.isfinite(obj):
            raise ValueError(f"Non-finite float is not allowed in config/manifest: {obj}")
        return obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, list):
        return [_reject_nonfinite_floats(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _reject_nonfinite_floats(v) for k, v in obj.items()}
    # Disallow YAML timestamps, sets, tuples, numpy types, etc.
    raise TypeError(f"Non-JSON type in config/manifest: {type(obj).__name__}")


def canonical_json_bytes(obj: Any) -> bytes:
    """
    Canonical JSON:
    - Keys sorted
    - No NaN/Inf
    - UTF-8
    - Minimal separators (stable)
    """
    cleaned = _reject_nonfinite_floats(obj)
    s = json.dumps(
        cleaned,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return s.encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def short_hash12(hex_digest: str) -> str:
    return hex_digest[:12]


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ----------------------------
# Schema validation
# ----------------------------

def load_json_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    # basic sanity: ensure schema is JSON-compatible
    _reject_nonfinite_floats(schema)
    return schema


def validate_instance(schema: Dict[str, Any], instance: Dict[str, Any]) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if errors:
        msg_lines = ["Config validation failed:"]
        for e in errors[:50]:
            path = ".".join([str(p) for p in e.path]) if e.path else "<root>"
            msg_lines.append(f"  - {path}: {e.message}")
        if len(errors) > 50:
            msg_lines.append(f"  ... and {len(errors) - 50} more errors")
        raise ValueError("\n".join(msg_lines))


# ----------------------------
# Config loading & freezing
# ----------------------------

@dataclasses.dataclass(frozen=True)
class FrozenConfig:
    cfg: Dict[str, Any]
    canonical_json_sha256: str
    freeze_hash12: str
    contract_version: str
    experiment_id: str
    seed: int


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")
    # Ensure JSON compatibility early
    return _reject_nonfinite_floats(cfg)


def freeze_config(cfg: Dict[str, Any], schema: Dict[str, Any]) -> FrozenConfig:
    validate_instance(schema, cfg)

    cbytes = canonical_json_bytes(cfg)
    digest = sha256_hex(cbytes)
    fh = short_hash12(digest)

    contract_version = str(cfg.get("contract_version", ""))
    if not contract_version:
        raise ValueError("Missing required field: contract_version")

    experiment_id = str(cfg.get("experiment_id", ""))
    if not experiment_id:
        raise ValueError("Missing required field: experiment_id")

    seed = cfg.get("reproducibility", {}).get("seed", None)
    if seed is None:
        raise ValueError("Missing required field: reproducibility.seed")
    if not isinstance(seed, int):
        raise ValueError("reproducibility.seed must be an integer")

    return FrozenConfig(
        cfg=cfg,
        canonical_json_sha256=digest,
        freeze_hash12=fh,
        contract_version=contract_version,
        experiment_id=experiment_id,
        seed=seed,
    )


def derived_rng(seed: int, freeze_hash12: str) -> np.random.Generator:
    """
    Deterministic RNG derived from (seed, freeze_hash12).
    """
    material = f"{seed}:{freeze_hash12}".encode("utf-8")
    h = hashlib.sha256(material).hexdigest()
    # take 64 bits for numpy seed
    s64 = int(h[:16], 16)
    return np.random.default_rng(s64)


# ----------------------------
# Manifest I/O
# ----------------------------

def capture_runtime_env() -> Dict[str, Any]:
    env = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "executable": sys.executable,
        "argv": sys.argv[:],
        "env": {
            # thread determinism knobs (recorded, not enforced here)
            "PYTHONHASHSEED": os.getenv("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.getenv("NUMEXPR_NUM_THREADS"),
            # provenance knobs
            "RIC_GIT_COMMIT": os.getenv("RIC_GIT_COMMIT"),
            "RIC_NIX_DRV": os.getenv("RIC_NIX_DRV"),
            "RIC_BUILD_ID": os.getenv("RIC_BUILD_ID"),
        },
    }
    try:
        import numpy  # noqa
        env["numpy_version"] = np.__version__
    except Exception:
        env["numpy_version"] = None
    try:
        import scipy  # noqa
        env["scipy_version"] = scipy.__version__  # type: ignore[attr-defined]
    except Exception:
        env["scipy_version"] = None
    return env


def load_manifest_in(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError("manifest-in must be a JSON object.")
    return _reject_nonfinite_floats(m)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def build_run_manifest_out(
    frozen: FrozenConfig,
    input_paths: Dict[str, str],
    input_hashes: Dict[str, str],
    statistic_def: Dict[str, Any],
    null_plan: Dict[str, Any],
    injection_plan: Dict[str, Any],
    started_unix_s: float,
    finished_unix_s: float,
) -> Dict[str, Any]:
    return {
        "contract_version": frozen.contract_version,
        "experiment_id": frozen.experiment_id,
        "freeze_hash12": frozen.freeze_hash12,
        "config_canonical_sha256": frozen.canonical_json_sha256,
        "timing": {
            "started_unix_s": started_unix_s,
            "finished_unix_s": finished_unix_s,
            "elapsed_s": finished_unix_s - started_unix_s,
        },
        "inputs": {
            "paths": input_paths,
            "sha256": input_hashes,
        },
        "definition": {
            "statistic": statistic_def,
            "null_tests": null_plan,
            "injection": injection_plan,
        },
        "runtime": capture_runtime_env(),
    }


# ----------------------------
# Physics placeholders (replace with your real implementation)
# ----------------------------

def compute_parity_statistic_placeholder(
    rng: np.random.Generator,
    cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Placeholder for your real comb/parity statistic.
    Returns (score, best_dt).

    Replace this with:
      - load H1/L1 strain windows
      - PSD estimation from off-source
      - whitening
      - FFTs
      - comb tooth masking
      - dt scan -> maximize statistic
    """
    scan = cfg["hyperparameters"]["scan"]
    dt_min, dt_max = scan["dt_range"]
    dt_step = scan["dt_step"]
    dts = np.arange(dt_min, dt_max + 0.5 * dt_step, dt_step)

    # deterministic “fake” score landscape (for scaffolding)
    best_dt = float(dts[rng.integers(0, len(dts))])
    score = float(abs(rng.normal(loc=10.0, scale=2.0)))
    return score, best_dt


def run_nulls_placeholder(
    rng: np.random.Generator,
    cfg: Dict[str, Any],
    observed_score: float,
) -> Dict[str, Any]:
    """
    Placeholder null tests. Replace with real:
      - shifted comb trials
      - time slides
    """
    null_tests = cfg["validation"]["null_tests"]
    out = {"trials_total": 0, "scores": [], "by_method": {}}

    for t in null_tests:
        method = t["method"]
        if method == "shifted_comb":
            n = int(t["n_trials"])
            # deterministic null scores
            scores = rng.normal(loc=5.0, scale=1.5, size=n).astype(float).tolist()
        elif method == "time_slides":
            n = int(t["n_slides"])
            scores = rng.normal(loc=4.5, scale=1.8, size=n).astype(float).tolist()
        else:
            raise ValueError(f"Unknown null method: {method}")

        out["by_method"][method] = {"n": n}
        out["scores"].extend(scores)
        out["trials_total"] += n

    # conservative p-value with +1 correction (>=)
    null_arr = np.array(out["scores"], dtype=float)
    p = float((np.sum(null_arr >= observed_score) + 1.0) / (len(null_arr) + 1.0))
    out["p_value"] = p
    return out


# ----------------------------
# CLI orchestration
# ----------------------------

def resolve_inputs(
    args: argparse.Namespace,
    frozen: FrozenConfig,
    manifest_in: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Decide which input paths are in use and compute hashes if required.

    Priority:
      - If manifest_in provides inputs, use those.
      - Else use CLI --data-h1/--data-l1.
    """
    require_hash_check = bool(frozen.cfg.get("reproducibility", {}).get("require_hash_check", False))

    input_paths: Dict[str, str] = {}
    expected_hashes: Dict[str, str] = {}

    if manifest_in is not None:
        inp = manifest_in.get("inputs", {})
        paths = inp.get("paths", {})
        hashes = inp.get("sha256", {})
        if not isinstance(paths, dict):
            raise ValueError("manifest-in inputs.paths must be an object")
        if not isinstance(hashes, dict):
            raise ValueError("manifest-in inputs.sha256 must be an object")

        # Expect at least h1/l1
        for k in ("h1", "l1"):
            if k not in paths:
                raise ValueError(f"manifest-in missing inputs.paths.{k}")
            input_paths[k] = str(paths[k])
            if k in hashes:
                expected_hashes[k] = str(hashes[k])

    else:
        if not args.data_h1 or not args.data_l1:
            raise ValueError("Provide either --manifest-in OR both --data-h1 and --data-l1")
        input_paths["h1"] = args.data_h1
        input_paths["l1"] = args.data_l1

        # Allow expected hashes from config.inputs if present
        cfg_inputs = frozen.cfg.get("inputs", {})
        if isinstance(cfg_inputs, dict):
            for k in ("h1", "l1"):
                if isinstance(cfg_inputs.get(k), dict) and "sha256" in cfg_inputs[k]:
                    expected_hashes[k] = str(cfg_inputs[k]["sha256"])

    # Compute actual hashes (if required, or if expected hashes provided)
    actual_hashes: Dict[str, str] = {}
    must_hash = require_hash_check or (len(expected_hashes) > 0)
    if must_hash:
        for k, p in input_paths.items():
            actual_hashes[k] = file_sha256(p)

    # Enforce hash checks if required
    if require_hash_check:
        for k in ("h1", "l1"):
            if k not in expected_hashes:
                raise ValueError(f"Hash check required but missing expected sha256 for input '{k}'")
            if actual_hashes.get(k) != expected_hashes[k]:
                raise ValueError(
                    f"Input hash mismatch for {k}: expected {expected_hashes[k]} got {actual_hashes.get(k)}"
                )

    return input_paths, actual_hashes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to YAML config (required unless --manifest-in provides it)")
    ap.add_argument("--schema", required=True, help="Path to JSON Schema for the config contract")
    ap.add_argument("--manifest-in", help="Optional orchestrator-provided run manifest (JSON)")
    ap.add_argument("--data-h1", help="Path to H1 input (if not using --manifest-in)")
    ap.add_argument("--data-l1", help="Path to L1 input (if not using --manifest-in)")
    ap.add_argument("--output", default="results.json", help="Results JSON output")
    ap.add_argument("--manifest-out", default="run_manifest_out.json", help="Provenance manifest output")
    args = ap.parse_args()

    started = time.time()

    schema = load_json_schema(args.schema)

    manifest_in: Optional[Dict[str, Any]] = None
    if args.manifest_in:
        manifest_in = load_manifest_in(args.manifest_in)

    # Resolve config path: manifest-in may supply it
    config_path = args.config
    if manifest_in is not None:
        m_cfg = manifest_in.get("config", {})
        if isinstance(m_cfg, dict) and "path" in m_cfg:
            config_path = str(m_cfg["path"]) if not config_path else config_path

    if not config_path:
        raise ValueError("Missing --config (or manifest-in config.path)")

    cfg = load_yaml_config(config_path)
    frozen = freeze_config(cfg, schema)

    print(f"CONFIG FROZEN: {frozen.freeze_hash12}  (sha256={frozen.canonical_json_sha256})")

    rng = derived_rng(frozen.seed, frozen.freeze_hash12)

    input_paths, input_hashes = resolve_inputs(args, frozen, manifest_in)

    # ---- Compute statistic (placeholder)
    score, best_dt = compute_parity_statistic_placeholder(rng, frozen.cfg)

    # ---- Null tests (placeholder)
    null_out = run_nulls_placeholder(rng, frozen.cfg, score)
    p_value = float(null_out["p_value"])

    # Definitions to record for auditability
    statistic_def = {
        "name": "T = max_dt |S(dt)|",
        "maximization": {"over": ["dt"], "grid": frozen.cfg["hyperparameters"]["scan"]},
        "comb": frozen.cfg["hyperparameters"]["comb"],
        "band_hz": frozen.cfg["hyperparameters"]["band_hz"],
    }
    null_plan = frozen.cfg["validation"]["null_tests"]
    injection_plan = frozen.cfg["validation"].get("injection", {})

    finished = time.time()

    # ---- Results JSON
    results = {
        "contract_version": frozen.contract_version,
        "experiment_id": frozen.experiment_id,
        "freeze_hash12": frozen.freeze_hash12,
        "config_canonical_sha256": frozen.canonical_json_sha256,
        "score": score,
        "best_dt": best_dt,
        "p_value": p_value,
        "null_trials_total": int(null_out["trials_total"]),
        # keep full null stats for downstream plots (can be large)
        "null_stats": null_out["scores"],
        "null_by_method": null_out["by_method"],
        "timing": {
            "started_unix_s": started,
            "finished_unix_s": finished,
            "elapsed_s": finished - started,
        },
    }
    write_json(args.output, results)
    print(f"Wrote results -> {args.output}")

    # ---- Manifest OUT
    manifest_out = build_run_manifest_out(
        frozen=frozen,
        input_paths=input_paths,
        input_hashes=input_hashes,
        statistic_def=statistic_def,
        null_plan=null_plan,
        injection_plan=injection_plan,
        started_unix_s=started,
        finished_unix_s=finished,
    )
    # If we had a manifest-in, thread it through for lineage
    if manifest_in is not None:
        manifest_out["lineage"] = {"manifest_in_sha256": sha256_hex(canonical_json_bytes(manifest_in))}
        # Record the config path used
        manifest_out["config"] = {"path": config_path}

    write_json(args.manifest_out, manifest_out)
    print(f"Wrote manifest -> {args.manifest_out}")


if __name__ == "__main__":
    main()
"""Tests for code review fixes: NaN-only sanitization, graceful restore,
and best-effort save_state."""

import json
import logging
import os

import numpy as np


# ---------- hessian_vector_product: NaN-only sanitization ----------


def test_hvp_sanitizes_nan_without_clipping():
    """hessian_vector_product should replace NaN with 0 but NOT clip finite values."""
    import jax.numpy as jnp

    from src.hessian_jax import JAXHessianEstimator

    def constraint(theta):
        return jnp.array([theta[0] + 2.0 * theta[1], theta[0] - theta[1]])

    est = JAXHessianEstimator(
        constraint_fn=constraint,
        weights=jnp.array([1.0, 1.0]),
        reg_lambda=1e-4,
    )

    theta = jnp.array([1.0, -0.5])
    v = jnp.array([1.0, 0.0])
    hv = est.hessian_vector_product(theta, v)

    # Result must be finite (NaNs sanitised)
    assert np.isfinite(np.asarray(hv)).all()


def test_hvp_preserves_large_finite_values():
    """Large but finite Hessian-vector products should not be clipped."""
    import jax.numpy as jnp

    from src.hessian_jax import JAXHessianEstimator

    # Use large weights so the HVP magnitude is >> 0.05
    def constraint(theta):
        return jnp.array([10.0 * theta[0] + 20.0 * theta[1]])

    est = JAXHessianEstimator(
        constraint_fn=constraint,
        weights=jnp.array([100.0]),
        reg_lambda=0.0,
    )

    theta = jnp.array([1.0, 1.0])
    v = jnp.array([1.0, 1.0])
    hv = est.hessian_vector_product(theta, v)

    # With large weights, the HVP should be large; it must NOT be clamped
    assert float(jnp.max(jnp.abs(hv))) > 0.05


# ---------- OrbaxAtomicStateIO.restore: graceful handling ----------


def test_restore_missing_file_returns_empty(tmp_path):
    """Restoring from a non-existent file should return {} instead of raising."""
    from src.proto.orbax_atomic import OrbaxAtomicStateIO

    io = OrbaxAtomicStateIO(str(tmp_path / "does_not_exist.json"))
    result = io.restore()
    assert result == {}


def test_restore_corrupt_json_returns_empty(tmp_path):
    """Restoring from a corrupt JSON file should return {} instead of raising."""
    from src.proto.orbax_atomic import OrbaxAtomicStateIO

    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text("NOT VALID JSON {{{{", encoding="utf-8")

    io = OrbaxAtomicStateIO(str(bad_file))
    result = io.restore()
    assert result == {}


def test_restore_valid_json_still_works(tmp_path):
    """A valid JSON file should be restored normally."""
    from src.proto.orbax_atomic import OrbaxAtomicStateIO

    good_file = tmp_path / "state.json"
    state = {"mu_global": 0.1, "L_global": 0.5}
    good_file.write_text(json.dumps(state), encoding="utf-8")

    io = OrbaxAtomicStateIO(str(good_file))
    result = io.restore()
    assert result == state


def test_restore_logs_warning_on_missing_file(tmp_path, caplog):
    """A warning should be logged when the state file is missing."""
    from src.proto.orbax_atomic import OrbaxAtomicStateIO

    io = OrbaxAtomicStateIO(str(tmp_path / "missing.json"))
    with caplog.at_level(logging.WARNING, logger="src.proto.orbax_atomic"):
        io.restore()
    assert "not found" in caplog.text


# ---------- save_state: best-effort persistence ----------


def test_save_state_swallows_io_errors(tmp_path):
    """save_state should not raise even when the I/O layer fails."""
    from src.callbacks.checkpointed_hessian_pl import (
        CheckpointedDistributedHessianPLCallback,
    )
    from src.proto.orbax_atomic import OrbaxAtomicStateIO

    # Use an unwritable path to provoke an error
    bad_path = str(tmp_path / "no_exist_dir" / "sub" / "state.json")
    cb = CheckpointedDistributedHessianPLCallback(state_save_path=bad_path)

    # Monkey-patch save to always raise
    def _boom(state):
        raise OSError("disk full")

    cb.state_io.save = _boom  # type: ignore[union-attr]

    # Must not raise
    cb.save_state()


def test_save_state_logs_warning_on_failure(tmp_path, caplog):
    """save_state should log a warning when persistence fails."""
    from src.callbacks.checkpointed_hessian_pl import (
        CheckpointedDistributedHessianPLCallback,
    )

    bad_path = str(tmp_path / "x" / "state.json")
    cb = CheckpointedDistributedHessianPLCallback(state_save_path=bad_path)

    def _boom(state):
        raise OSError("disk full")

    cb.state_io.save = _boom  # type: ignore[union-attr]

    with caplog.at_level(logging.WARNING, logger="src.callbacks.checkpointed_hessian_pl"):
        cb.save_state()
    assert "Failed to save" in caplog.text

“””
Waveform builder with gravitational wave echo model.

Option A addition: meta now includes ‘omega_full’ (full-spectrum angular frequencies
from np.fft.fftfreq) alongside existing ‘omega’ (half-spectrum from np.fft.rfftfreq).

omega_full is consumed by TensorCell’s CausalEnforcer for KK causality enforcement.
omega (rfft) continues to be used by transfer_function_K for echo construction.
“””

import numpy as np
from typing import Union, Optional, Callable, Tuple

# ────────────────────────────────────────────────────────────────────────

# Transfer Function

# ────────────────────────────────────────────────────────────────────────

def transfer_function_K(
omega: np.ndarray,
T_inf: np.ndarray,
R_inf: np.ndarray,
R_s: Union[complex, np.ndarray],
dt: float,
check_resonances: bool = True,
) -> np.ndarray:
“””
Cavity transfer function for echoes with resonance handling.

```
Parameters
----------
omega : np.ndarray
    Angular frequency array (rad/s).
T_inf : np.ndarray
    Transmission coefficient array.
R_inf : np.ndarray
    Reflection coefficient array (complex).
R_s : complex or np.ndarray
    Surface reflectivity — scalar complex OR frequency-dependent complex array.
dt : float
    Echo time delay (seconds).
check_resonances : bool
    If True, handles near-resonance cases (avoids division by zero).

Returns
-------
K(ω) : np.ndarray
    Complex transfer function array.
"""
phase = np.exp(1j * omega * dt)

# Works for both scalar and array R_s
denominator = 1.0 - R_inf * R_s * phase

# Avoid division by zero near resonances
if check_resonances:
    denominator = np.where(
        np.abs(denominator) < 1e-10,
        1e-10 * np.exp(1j * np.angle(denominator)),
        denominator,
    )

return (T_inf ** 2) * R_s * phase / denominator
```

# ────────────────────────────────────────────────────────────────────────

# Waveform Builder

# ────────────────────────────────────────────────────────────────────────

def build_waveform_with_echoes(
t: np.ndarray,
Mtot_solar: float,
epsilon: float,
h0: np.ndarray,
R_s_model: Optional[Callable] = None,
use_parametric_rs: bool = False,
**R_s_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
“””
Build waveform with echoes using parametric R_s if provided.

```
Parameters
----------
t : np.ndarray
    Time array (seconds).
Mtot_solar : float
    Total mass in solar masses.
epsilon : float
    Quantum gravity parameter.
h0 : np.ndarray
    Base merger waveform (pre-generated).
R_s_model : callable, optional
    Function R_s(omega, **kwargs) returning frequency-dependent R_s.
use_parametric_rs : bool
    If True, use frequency-dependent R_s from R_s_model.
**R_s_kwargs
    Parameters for R_s_model, or R_s_mag/R_s_phase for constant model.

Returns
-------
h_total : np.ndarray
    Complete waveform with echoes.
h0 : np.ndarray
    Base merger waveform (passthrough).
h_echo : np.ndarray
    Echo component only.
meta : dict
    Metadata including both omega (rfft) and omega_full (fft) frequency axes.
"""
dt_samp = t[1] - t[0]
fs = 1.0 / dt_samp
N = t.size

# ── FFT (half-spectrum for echo construction) ────────────────
H0 = np.fft.rfft(h0)
freqs = np.fft.rfftfreq(N, d=dt_samp)
omega = 2 * np.pi * freqs

# ── Full-spectrum frequencies (for KK enforcement in TensorCell) ──
freqs_full = np.fft.fftfreq(N, d=dt_samp)
omega_full = 2 * np.pi * freqs_full

# ── Barrier model ────────────────────────────────────────────
T_inf, R_inf = make_toy_greybody(freqs, f0=240.0, p=4.0)

# ── Echo delay ───────────────────────────────────────────────
dt_echo = echo_delay_seconds(Mtot_solar, epsilon=epsilon)

# ── Surface reflectivity ─────────────────────────────────────
if use_parametric_rs and R_s_model is not None:
    R_s = R_s_model(omega, **R_s_kwargs)
else:
    R_s_mag = R_s_kwargs.get("R_s_mag", 0.70)
    R_s_phase = R_s_kwargs.get("R_s_phase", 0.30 * np.pi)
    R_s = R_s_mag * np.exp(1j * R_s_phase)

# ── Transfer function & echo ─────────────────────────────────
K = transfer_function_K(omega, T_inf=T_inf, R_inf=R_inf, R_s=R_s, dt=dt_echo)

h_echo = np.fft.irfft(H0 * K, n=N)
h_total = h0 + h_echo

# ── Metadata ─────────────────────────────────────────────────
meta = {
    # Existing fields (rfft half-spectrum)
    "fs": fs,
    "freqs": freqs,
    "omega": omega,
    # NEW: full-spectrum for CausalEnforcer (Option A)
    "freqs_full": freqs_full,
    "omega_full": omega_full,
    # Echo model parameters
    "T_inf": T_inf,
    "R_inf": R_inf,
    "R_s": R_s,
    "dt_echo": dt_echo,
    "R_s_model_used": "parametric" if use_parametric_rs else "constant",
    # Sample count (needed for omega reconstruction if meta gets serialized)
    "N": N,
    "dt_samp": dt_samp,
}

return h_total, h0, h_echo, meta
```

# ────────────────────────────────────────────────────────────────────────

# Caller example: wiring meta → TensorCell inputs

# ────────────────────────────────────────────────────────────────────────

def prepare_tensorcell_inputs(strain_dlpack, comb_mask_dlpack, meta: dict) -> dict:
“””
Build the inputs dict for TensorCell.solve_physics from waveform meta.

```
This bridges the waveform builder (numpy/rfft world) to the TensorCell
(vbt/full-FFT world). The key addition is passing omega_full for KK
enforcement.

Parameters
----------
strain_dlpack : DLPack capsule
    Strain tensor from vbt.to_dlpack().
comb_mask_dlpack : DLPack capsule
    Comb mask tensor from vbt.to_dlpack().
meta : dict
    Metadata from build_waveform_with_echoes().

Returns
-------
dict suitable for TensorCell.solve_physics().
"""
return {
    "strain_dlpack": strain_dlpack,
    "comb_mask_dlpack": comb_mask_dlpack,
    "omega": meta["omega_full"],    # Full-spectrum for KK enforcement
    "sample_rate": meta["fs"],
}
```

# ────────────────────────────────────────────────────────────────────────

# Stubs for functions referenced above (replace with your implementations)

# ────────────────────────────────────────────────────────────────────────

def make_toy_greybody(freqs, f0=240.0, p=4.0):
“”“Placeholder — replace with your barrier model.”””
x = (freqs / f0) ** p
T_inf = np.sqrt(x / (1 + x))
R_inf = np.sqrt(1 / (1 + x)) * np.exp(1j * 0.1)  # Small complex phase
return T_inf, R_inf

def echo_delay_seconds(Mtot_solar, epsilon=1e-40):
“”“Placeholder — replace with your echo delay calculation.”””
# dt_echo ≈ -M * ln(epsilon) in geometric units, converted to seconds
M_sec = Mtot_solar * 4.925491e-6  # Solar mass in seconds
return -M_sec * np.log(epsilon)
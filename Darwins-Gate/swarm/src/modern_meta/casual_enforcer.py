“””
CausalEnforcer — Kramers-Kronig causality enforcement for spectral responses.

Enforces minimum-phase constraint on complex transfer functions via the
Hilbert-transform relationship between log-magnitude and phase (KK relations).

Genome-evolvable parameters:
kk_tolerance: float [0, 1] — blend between measured and KK-derived phase
enforce_kk: bool           — on/off switch (safety valve for Darwinian selection)

Physics:
KK relations guarantee that a causal, stable transfer function satisfies:
phase(ω) = H[ln|R(ω)|]
where H is the Hilbert transform. Deviations from this indicate either
acausal artifacts (cable reflections, feedback loops) or genuine physical
phase information (sky position, orbital phase). The tolerance parameter
controls how aggressively we enforce causality vs preserve measured phase.
“””

import numpy as np
import vibetensor as vbt

class CausalEnforcer:
def **init**(self, kk_tolerance: float = 0.8, enforce_kk: bool = True):
self.kk_tolerance = kk_tolerance
self.enforce_kk = enforce_kk

```
def apply_kramers_kronig(self, omega: vbt.tensor, R_s: vbt.tensor) -> vbt.tensor:
    """
    GPU-accelerated KK causality enforcement on a complex spectral response.

    Parameters
    ----------
    omega : vbt.tensor
        Angular frequency array (rad/s), full-spectrum (N bins from fft, NOT rfft).
    R_s : vbt.tensor
        Complex spectral response to enforce causality on.

    Returns
    -------
    vbt.tensor
        Causality-enforced complex spectral response.
    """
    if not self.enforce_kk:
        return R_s

    # Check for evenly spaced omega (KK via FFT requires uniform grid)
    domega = omega[1] - omega[0]
    if not vbt.allclose(vbt.diff(omega), domega):
        print("Warning: Uneven omega spacing – skipping KK enforcement")
        return R_s

    magnitude = vbt.abs(R_s)
    phase = vbt.angle(R_s)

    # Log-magnitude (clipped to avoid log(0))
    log_magnitude = vbt.log(vbt.clip(magnitude, 1e-10, None))

    # Hilbert transform of log-magnitude via FFT
    # H[ln|R|] gives the minimum-phase (causal) phase response
    H_sign = self._hilbert_sign_mask(len(omega), rfft=False)
    H_logR = vbt.real(vbt.ifft(1j * H_sign * vbt.fft(log_magnitude)))

    # KK-derived phase (minimum phase)
    phase_kk = H_logR

    # Blend: tolerance=1.0 → strict causality, 0.0 → preserve measured phase
    phase_corrected = (1 - self.kk_tolerance) * phase + self.kk_tolerance * phase_kk

    # Reconstruct complex response with corrected phase
    # Use cos/sin decomposition for robustness (avoids complex exp edge cases)
    return magnitude * (vbt.cos(phase_corrected) + 1j * vbt.sin(phase_corrected))

@staticmethod
def _hilbert_sign_mask(N: int, rfft: bool = False):
    """
    Frequency-domain signum function for analytic signal construction.

    For full FFT (rfft=False):
      H[0] = 0 (DC), H[1:N/2] = +1 (positive freq),
      H[N/2] = 0 (Nyquist if even), H[N/2+1:] = -1 (negative freq)

    For rfft (rfft=True):
      H[0] = 0 (DC), H[1:-1] = +1 (positive freq), H[-1] = 0 (Nyquist)
    """
    if rfft:
        H = vbt.ones(N)
        H[0] = 0       # DC
        if N % 2 == 0:
            H[-1] = 0  # Nyquist
        return H

    # Full-spectrum sign mask
    H = vbt.zeros(N)
    if N % 2 == 0:
        H[1:N // 2] = 1
        H[N // 2 + 1:] = -1
    else:
        H[1:(N + 1) // 2] = 1
        H[(N + 1) // 2:] = -1
    return H

def get_params(self) -> dict:
    return {"kk_tolerance": self.kk_tolerance, "enforce_kk": self.enforce_kk}

def set_params(self, params: dict):
    if "kk_tolerance" in params:
        self.kk_tolerance = params["kk_tolerance"]
    if "enforce_kk" in params:
        self.enforce_kk = params["enforce_kk"]

def __repr__(self):
    return (
        f"CausalEnforcer(kk_tolerance={self.kk_tolerance}, "
        f"enforce_kk={self.enforce_kk})"
    )
```
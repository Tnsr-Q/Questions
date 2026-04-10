"""
WKB Calculation of Complex R_∞(ω) with Phase
=============================================
Computes both |R_∞| and arg(R_∞) for Schwarzschild barrier scattering.

The phase shift δ(ω) = arg(R_∞) determines the pole ladder structure:
    Resonances when: arg(R_∞) + ωΔt = 2πn

Key physics:
- Below barrier (ω² < V_max): exponential suppression + phase from evanescent matching
- Above barrier (ω² > V_max): oscillatory transmission + phase from path integral
- Near barrier (ω² ≈ V_max): Airy function matching required

Author: Tanner Q / TNSR-Q
"""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.special import airy
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

# =============================================================================
# POTENTIAL AND COORDINATES
# =============================================================================

def regge_wheeler_potential(r: np.ndarray, M: float, ell: int, spin: int) -> np.ndarray:
    """
    V(r) = (1 - 2M/r)[ℓ(ℓ+1)/r² + (1-s²)2M/r³]
    """
    f = 1 - 2*M/r
    V = f * (ell*(ell+1)/r**2 + (1 - spin**2) * 2*M/r**3)
    return V

def tortoise_to_r(r_star: float, M: float, tol: float = 1e-10) -> float:
    """Invert r*(r) = r + 2M ln(r/2M - 1)"""
    def f(r):
        return r + 2*M * np.log(r/(2*M) - 1) - r_star
    
    # Bracket search
    r_min = 2.001 * M
    r_max = max(1000*M, r_star + 100*M)
    
    try:
        return brentq(f, r_min, r_max, xtol=tol)
    except:
        return r_min if r_star < 0 else r_max

def build_potential_interpolator(M: float, ell: int, spin: int, 
                                  r_star_range: Tuple[float, float] = (-50, 50)):
    """Build V(r*) interpolator for WKB calculations."""
    r_star_grid = np.linspace(r_star_range[0], r_star_range[1], 2000)
    r_grid = np.array([tortoise_to_r(rs, M) for rs in r_star_grid])
    V_grid = regge_wheeler_potential(r_grid, M, ell, spin)
    
    return interp1d(r_star_grid, V_grid, kind='cubic', 
                   fill_value=(V_grid[0], V_grid[-1]), bounds_error=False)

# =============================================================================
# TURNING POINT FINDER
# =============================================================================

def find_turning_points(omega: float, V_func, r_star_range: Tuple[float, float] = (-50, 50),
                        M: float = 1.0) -> list:
    """
    Find classical turning points where ω² = V(r*).
    
    For sub-barrier: two turning points bracketing the peak
    For above-barrier: no real turning points (classically allowed everywhere asymptotically)
    """
    r_star_grid = np.linspace(r_star_range[0], r_star_range[1], 1000)
    V_grid = V_func(r_star_grid)
    
    # Find where ω² - V changes sign
    diff = omega**2 - V_grid
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    turning_points = []
    for idx in sign_changes:
        # Refine with bisection
        rs_left = r_star_grid[idx]
        rs_right = r_star_grid[idx + 1]
        
        def f(rs):
            return omega**2 - V_func(rs)
        
        try:
            tp = brentq(f, rs_left, rs_right)
            turning_points.append(tp)
        except:
            pass
    
    return sorted(turning_points)

# =============================================================================
# WKB PHASE CALCULATION
# =============================================================================

@dataclass
class WKBResult:
    """Container for WKB scattering results."""
    omega: float
    R_magnitude: float
    R_phase: float
    R_complex: complex
    T_magnitude: float
    regime: str  # 'sub-barrier', 'above-barrier', 'near-barrier'
    turning_points: list
    barrier_integral: float  # κ integral through barrier

def wkb_sub_barrier(omega: float, V_func, turning_points: list, M: float = 1.0) -> WKBResult:
    """
    WKB for ω² < V_max (tunneling regime).
    
    Between turning points r*₁ and r*₂:
        κ(r*) = √(V(r*) - ω²)
        
    Transmission: |T|² = exp(-2∫κ dr*)
    Reflection: |R|² = 1 - |T|²
    
    Phase shift from connection formulas:
        δ = -π/4 + ∫_{r*₁}^{∞} k dr* - ωr*₁  (incoming phase reference)
    
    where k = √(ω² - V) in classically allowed region.
    """
    if len(turning_points) < 2:
        # No barrier to tunnel through at this frequency
        return WKBResult(omega=omega, R_magnitude=0.0, R_phase=0.0,
                        R_complex=0.0, T_magnitude=1.0, regime='above-barrier',
                        turning_points=turning_points, barrier_integral=0.0)
    
    r1, r2 = turning_points[0], turning_points[-1]
    
    # Barrier integral: ∫_{r1}^{r2} κ dr*
    def kappa(r_star):
        V = V_func(r_star)
        val = V - omega**2
        return np.sqrt(max(0, val))
    
    barrier_integral, _ = quad(kappa, r1, r2, limit=200)
    
    # Transmission coefficient
    T_squared = np.exp(-2 * barrier_integral)
    T_mag = np.sqrt(T_squared)
    R_squared = 1 - T_squared
    R_mag = np.sqrt(max(0, R_squared))
    
    # Phase calculation
    # In the outer region (r* > r2), the wave is:
    #   ψ ~ e^{-iωr*} + R e^{+iωr*}
    # 
    # WKB connection formula gives phase shift:
    #   δ = -π/2 + 2∫_{r2}^{∞} (k - ω) dr*
    # 
    # where k = √(ω² - V) ≈ ω - V/(2ω) for large r*
    
    # Outer phase integral: ∫_{r2}^{r_far} (√(ω²-V) - ω) dr*
    r_far = 100 * M
    
    def phase_integrand(r_star):
        V = V_func(r_star)
        if omega**2 > V:
            return np.sqrt(omega**2 - V) - omega
        return 0
    
    outer_phase, _ = quad(phase_integrand, r2, r_far, limit=200)
    
    # Inner phase integral: ∫_{-r_far}^{r1} (√(ω²-V) - ω) dr*
    r_inner = -50 * M
    inner_phase, _ = quad(phase_integrand, r_inner, r1, limit=200)
    
    # Total WKB phase shift
    # For reflection off a barrier: δ = π/2 + 2×(outer_phase)
    # The π/2 comes from the connection formula at turning point
    delta = np.pi/2 + 2 * outer_phase
    
    # Additional phase from barrier penetration (for transmitted component)
    # The reflected wave picks up phase from the standing wave pattern
    delta += np.arctan2(T_mag, R_mag) if R_mag > 0 else 0
    
    R_complex = R_mag * np.exp(1j * delta)
    
    return WKBResult(
        omega=omega,
        R_magnitude=R_mag,
        R_phase=delta,
        R_complex=R_complex,
        T_magnitude=T_mag,
        regime='sub-barrier',
        turning_points=turning_points,
        barrier_integral=barrier_integral
    )

def wkb_above_barrier(omega: float, V_func, V_max: float, M: float = 1.0) -> WKBResult:
    """
    WKB for ω² > V_max (above barrier).
    
    No classical turning points - wave propagates through.
    Phase shift from path integral.
    """
    # Phase integral over the entire potential
    r_inner = -50 * M
    r_outer = 100 * M
    
    def k_minus_omega(r_star):
        V = V_func(r_star)
        k = np.sqrt(max(0, omega**2 - V))
        return k - omega
    
    phase_integral, _ = quad(k_minus_omega, r_inner, r_outer, limit=200)
    
    # Above barrier: mostly transmission, small reflection from potential variation
    # |R|² ≈ exp(-2π ω / |V''|^{1/2}) for smooth barriers (very small)
    
    # Estimate |R|² from potential curvature at peak
    # Find peak location
    r_star_test = np.linspace(-10, 30, 500)
    V_test = V_func(r_star_test)
    idx_max = np.argmax(V_test)
    
    # Second derivative at peak (numerical)
    dr = r_star_test[1] - r_star_test[0]
    if idx_max > 0 and idx_max < len(V_test) - 1:
        V_pp = (V_test[idx_max+1] - 2*V_test[idx_max] + V_test[idx_max-1]) / dr**2
    else:
        V_pp = -0.01  # Default curvature
    
    # Above-barrier reflection (semiclassical)
    if abs(V_pp) > 1e-10:
        exponent = -np.pi * (omega**2 - V_max) / np.sqrt(abs(V_pp))
        R_squared = np.exp(2 * exponent) if exponent > -50 else 0
    else:
        R_squared = 0
    
    R_mag = np.sqrt(max(0, min(1, R_squared)))
    T_mag = np.sqrt(1 - R_mag**2)
    
    # Phase shift
    delta = 2 * phase_integral + np.pi/4  # π/4 from Maslov index
    
    R_complex = R_mag * np.exp(1j * delta)
    
    return WKBResult(
        omega=omega,
        R_magnitude=R_mag,
        R_phase=delta,
        R_complex=R_complex,
        T_magnitude=T_mag,
        regime='above-barrier',
        turning_points=[],
        barrier_integral=0.0
    )

def wkb_near_barrier(omega: float, V_func, V_max: float, r_star_peak: float, 
                     M: float = 1.0) -> WKBResult:
    """
    Uniform WKB / Airy function matching near V_max.
    
    For ω² ≈ V_max, use parabolic barrier approximation:
        V(r*) ≈ V_max - ½|V''|(r* - r*_peak)²
    
    Exact solution in terms of parabolic cylinder functions.
    """
    # Estimate V'' at peak
    eps = 0.1
    V_p = V_func(r_star_peak + eps)
    V_m = V_func(r_star_peak - eps)
    V_0 = V_func(r_star_peak)
    V_pp = (V_p - 2*V_0 + V_m) / eps**2
    
    # Barrier penetration parameter
    # For parabolic barrier: ν = (V_max - ω²) / √(2|V''|)
    if abs(V_pp) > 1e-10:
        nu = (V_max - omega**2) / np.sqrt(2 * abs(V_pp))
    else:
        nu = 0
    
    # Transmission and reflection from exact parabolic barrier formula
    # |T|² = 1 / (1 + e^{2πν})
    # |R|² = e^{2πν} / (1 + e^{2πν})
    
    if nu > 10:  # Deep below barrier
        T_squared = np.exp(-2*np.pi*nu)
        R_squared = 1 - T_squared
    elif nu < -10:  # Far above barrier
        R_squared = np.exp(2*np.pi*nu)
        T_squared = 1 - R_squared
    else:
        exp_2pi_nu = np.exp(2*np.pi*nu)
        T_squared = 1 / (1 + exp_2pi_nu)
        R_squared = exp_2pi_nu / (1 + exp_2pi_nu)
    
    R_mag = np.sqrt(max(0, min(1, R_squared)))
    T_mag = np.sqrt(max(0, min(1, T_squared)))
    
    # Phase from parabolic barrier
    # arg(R) = arg(Γ(½ + iν)) - ν(ln|ν| - 1) + π/4
    # Using Stirling approximation for Gamma function argument
    
    from scipy.special import loggamma
    
    if abs(nu) > 0.01:
        # arg(Γ(½ + iν)) from imaginary part of log Γ
        log_gamma = loggamma(0.5 + 1j*nu)
        gamma_arg = np.imag(log_gamma)
        delta = gamma_arg - nu * (np.log(abs(nu)) - 1) + np.pi/4
    else:
        # Near top of barrier
        delta = np.pi/4
    
    # Outer region phase contribution
    r_outer = 100 * M
    def phase_integrand(r_star):
        V = V_func(r_star)
        if omega**2 > V:
            return np.sqrt(omega**2 - V) - omega
        return 0
    
    outer_phase, _ = quad(phase_integrand, r_star_peak + 5, r_outer, limit=100)
    delta += 2 * outer_phase
    
    R_complex = R_mag * np.exp(1j * delta)
    
    return WKBResult(
        omega=omega,
        R_magnitude=R_mag,
        R_phase=delta,
        R_complex=R_complex,
        T_magnitude=T_mag,
        regime='near-barrier',
        turning_points=[],
        barrier_integral=nu
    )

# =============================================================================
# MASTER WKB FUNCTION
# =============================================================================

def compute_R_infinity_wkb(omega: float, ell: int = 2, spin: int = 2, 
                           M: float = 1.0) -> WKBResult:
    """
    Compute complex R_∞(ω) using appropriate WKB regime.
    
    Returns both magnitude and phase.
    """
    # Build potential
    V_func = build_potential_interpolator(M, ell, spin)
    
    # Find V_max and its location
    r_star_test = np.linspace(-10, 40, 500)
    V_test = V_func(r_star_test)
    idx_max = np.argmax(V_test)
    V_max = V_test[idx_max]
    r_star_peak = r_star_test[idx_max]
    
    # Determine regime
    omega_max = np.sqrt(V_max)
    
    if omega < 0.8 * omega_max:
        # Deep sub-barrier: standard WKB tunneling
        turning_points = find_turning_points(omega, V_func, M=M)
        return wkb_sub_barrier(omega, V_func, turning_points, M)
    
    elif omega > 1.2 * omega_max:
        # Above barrier: propagating regime
        return wkb_above_barrier(omega, V_func, V_max, M)
    
    else:
        # Near barrier: uniform approximation
        return wkb_near_barrier(omega, V_func, V_max, r_star_peak, M)

# =============================================================================
# NUMERICAL VERIFICATION (FULL WAVE EQUATION)
# =============================================================================

def compute_R_infinity_numerical(omega: float, ell: int = 2, spin: int = 2,
                                  M: float = 1.0) -> Tuple[complex, float, float]:
    """
    Numerical solution of wave equation for comparison.
    Returns (R_complex, |R|, arg(R))
    """
    if omega < 1e-4:
        return 1.0 + 0j, 1.0, 0.0
    
    # Grid
    r_horizon = 2*M * 1.001
    r_far = 300 * M
    r_grid = np.linspace(r_horizon, r_far, 5000)
    r_star = r_grid + 2*M * np.log(r_grid/(2*M) - 1)
    
    # Potential
    f = 1 - 2*M/r_grid
    V = f * (ell*(ell+1)/r_grid**2 + (1 - spin**2) * 2*M/r_grid**3)
    V_interp = interp1d(r_star, V, kind='cubic', fill_value=(V[0], 0), bounds_error=False)
    
    def wave_eq(t, y, omega):
        psi, dpsi = y
        return [dpsi, -(omega**2 - V_interp(t)) * psi]
    
    # BC: ingoing at horizon
    r_star_0 = r_star[0]
    y0 = [np.exp(-1j * omega * r_star_0), 
          -1j * omega * np.exp(-1j * omega * r_star_0)]
    
    sol = solve_ivp(wave_eq, [r_star[0], r_star[-1]], y0, 
                   args=(omega,), method='RK45', max_step=0.5,
                   t_eval=r_star)
    
    # Extract at far field (use multiple points for stability)
    n_avg = 50
    R_values = []
    
    for idx in range(-n_avg, -1):
        psi_L = sol.y[0, idx]
        dpsi_L = sol.y[1, idx]
        r_L = sol.t[idx]
        
        A_out = 0.5 * (psi_L + dpsi_L/(1j*omega)) * np.exp(-1j*omega*r_L)
        A_in = 0.5 * (psi_L - dpsi_L/(1j*omega)) * np.exp(1j*omega*r_L)
        
        if np.abs(A_in) > 1e-15:
            R_values.append(A_out / A_in)
    
    if R_values:
        R_complex = np.mean(R_values)
    else:
        R_complex = 0.0 + 0j
    
    return R_complex, np.abs(R_complex), np.angle(R_complex)

# =============================================================================
# POLE LADDER ANALYSIS
# =============================================================================

def compute_pole_ladder(omega_range: np.ndarray, R_surface: float, delta_t: float,
                        ell: int = 2, spin: int = 2, M: float = 1.0) -> dict:
    """
    Compute the pole ladder structure from resonance condition:
        R_∞(ω) · R_s · e^{iωΔt} = 1
    
    Poles occur when:
        |R_∞| |R_s| = 1  AND  arg(R_∞) + ωΔt = 2πn
    
    Returns frequencies and phases for pole analysis.
    """
    results = {
        'omega': omega_range,
        'R_magnitude': np.zeros(len(omega_range)),
        'R_phase': np.zeros(len(omega_range)),
        'total_phase': np.zeros(len(omega_range)),  # arg(R_∞) + ωΔt
        'resonance_proximity': np.zeros(len(omega_range)),  # distance to nearest 2πn
    }
    
    print(f"Computing pole ladder (R_s={R_surface}, Δt={delta_t:.1f}M)...")
    
    for i, omega in enumerate(omega_range):
        # Use numerical for accuracy
        R_complex, R_mag, R_phase = compute_R_infinity_numerical(omega, ell, spin, M)
        
        results['R_magnitude'][i] = R_mag
        results['R_phase'][i] = R_phase
        
        # Total phase in resonance condition
        total_phase = R_phase + omega * delta_t
        results['total_phase'][i] = total_phase
        
        # Distance to nearest 2πn
        nearest_n = np.round(total_phase / (2*np.pi))
        results['resonance_proximity'][i] = abs(total_phase - 2*np.pi*nearest_n)
        
        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(omega_range)}")
    
    # Find approximate pole frequencies (where resonance_proximity is minimized)
    # and |R_∞||R_s| is close to 1
    
    return results

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_complex_R_analysis(outdir: Path, M_solar: float = 30.0, R_surface: float = 0.3):
    """
    Create comprehensive complex R_∞(ω) analysis with phase structure.
    """
    print("="*70)
    print("COMPLEX R_∞(ω) ANALYSIS WITH PHASE")
    print("="*70)
    
    M = 1.0  # Work in geometric units
    ell, spin = 2, 2
    
    # Echo delay
    M_meters = M_solar * 1.989e30 * 6.674e-11 / (2.998e8)**2
    l_Pl = 1.616e-35
    delta_t = 2 * (3 + 2*np.log(M_meters/l_Pl))
    
    print(f"Echo delay: Δt = {delta_t:.1f} M")
    
    # Frequency range
    omega_range = np.linspace(0.05, 1.2, 150)
    
    # Compute R_∞ with both methods for comparison
    R_numerical = []
    R_wkb = []
    
    print("\nComputing R_∞(ω)...")
    for i, omega in enumerate(omega_range):
        # Numerical
        R_num, _, _ = compute_R_infinity_numerical(omega, ell, spin, M)
        R_numerical.append(R_num)
        
        # WKB
        wkb_result = compute_R_infinity_wkb(omega, ell, spin, M)
        R_wkb.append(wkb_result.R_complex)
        
        if (i+1) % 30 == 0:
            print(f"  Progress: {i+1}/{len(omega_range)}")
    
    R_numerical = np.array(R_numerical)
    R_wkb = np.array(R_wkb)
    
    # Compute pole ladder
    total_phase_num = np.angle(R_numerical) + omega_range * delta_t
    total_phase_wkb = np.angle(R_wkb) + omega_range * delta_t
    
    # Find resonances
    resonance_condition = np.abs(R_numerical) * R_surface
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Panel 1: |R_∞|² comparison
    ax1 = axes[0, 0]
    ax1.plot(omega_range, np.abs(R_numerical)**2, 'b-', lw=2, label='Numerical')
    ax1.plot(omega_range, np.abs(R_wkb)**2, 'r--', lw=2, label='WKB')
    ax1.set_xlabel(r'$\omega M$')
    ax1.set_ylabel(r'$|R_\infty|^2$')
    ax1.set_title('Reflection Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: arg(R_∞) comparison
    ax2 = axes[0, 1]
    ax2.plot(omega_range, np.unwrap(np.angle(R_numerical)), 'b-', lw=2, label='Numerical')
    ax2.plot(omega_range, np.unwrap(np.angle(R_wkb)), 'r--', lw=2, label='WKB')
    ax2.set_xlabel(r'$\omega M$')
    ax2.set_ylabel(r'$\arg(R_\infty)$')
    ax2.set_title('Reflection Phase')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Total phase (determines pole ladder)
    ax3 = axes[0, 2]
    ax3.plot(omega_range, total_phase_num, 'b-', lw=2, label='Numerical')
    # Mark 2πn lines
    for n in range(-5, 50):
        ax3.axhline(2*np.pi*n, color='gray', ls=':', alpha=0.5)
    ax3.set_xlabel(r'$\omega M$')
    ax3.set_ylabel(r'$\arg(R_\infty) + \omega \Delta t$')
    ax3.set_title(f'Pole Ladder Phase ($\Delta t = {delta_t:.0f}M$)')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Complex plane trajectory
    ax4 = axes[1, 0]
    # Color by frequency
    colors = plt.cm.viridis(np.linspace(0, 1, len(omega_range)))
    for i in range(len(omega_range)-1):
        ax4.plot([R_numerical[i].real, R_numerical[i+1].real],
                [R_numerical[i].imag, R_numerical[i+1].imag],
                color=colors[i], lw=2)
    ax4.plot(R_numerical[0].real, R_numerical[0].imag, 'go', markersize=10, label='Low ω')
    ax4.plot(R_numerical[-1].real, R_numerical[-1].imag, 'ro', markersize=10, label='High ω')
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    ax4.set_xlabel(r'Re$(R_\infty)$')
    ax4.set_ylabel(r'Im$(R_\infty)$')
    ax4.set_title('Complex Plane Trajectory')
    ax4.legend()
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Transfer function |K|² with phase
    ax5 = axes[1, 1]
    K = (1 - np.abs(R_numerical)**2) * R_surface * np.exp(1j*omega_range*delta_t) / \
        (1 - R_numerical * R_surface * np.exp(1j*omega_range*delta_t))
    ax5.semilogy(omega_range, np.abs(K)**2, 'b-', lw=2)
    ax5.set_xlabel(r'$\omega M$')
    ax5.set_ylabel(r'$|K(\omega)|^2$')
    ax5.set_title(f'Echo Transfer Function ($R_s = {R_surface}$)')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Resonance condition |R_∞||R_s| and phase mod 2π
    ax6 = axes[1, 2]
    ax6.plot(omega_range, np.abs(R_numerical) * R_surface, 'b-', lw=2, 
            label=r'$|R_\infty| R_s$')
    ax6.axhline(1.0, color='red', ls='--', label='Resonance condition')
    ax6.set_xlabel(r'$\omega M$')
    ax6.set_ylabel('Magnitude condition')
    ax6.set_title('Pole Magnitude Condition')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1.2)
    
    plt.suptitle(f'Complex $R_\\infty(\\omega)$ Analysis (ℓ=2, s=2, $M={M_solar}M_\\odot$)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    png_path = outdir / "complex_R_infinity_analysis.png"
    pdf_path = outdir / "complex_R_infinity_analysis.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {png_path}")
    
    # Print phase table
    print("\n" + "="*70)
    print("PHASE TABLE: R_∞(ω) for pole ladder analysis")
    print("="*70)
    print(f"\n{'ωM':>8} | {'|R_∞|':>10} | {'arg(R_∞)':>12} | {'arg(R)+ωΔt':>12} | {'n (nearest)':>12}")
    print("-"*70)
    
    for omega in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        R, R_mag, R_arg = compute_R_infinity_numerical(omega, ell, spin, M)
        total = R_arg + omega * delta_t
        n_nearest = int(np.round(total / (2*np.pi)))
        print(f"{omega:8.2f} | {R_mag:10.4f} | {R_arg:12.4f} | {total:12.2f} | {n_nearest:12d}")
    
    return png_path, pdf_path

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    outdir = Path("/mnt/user-data/outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    
    plot_complex_R_analysis(outdir, M_solar=30.0, R_surface=0.3)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

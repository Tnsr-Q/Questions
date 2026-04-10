"""
Schwarzschild Black Hole Reflection Coefficient R_∞(ω)
======================================================
Computes the reflection/transmission coefficients for scalar (s=0), 
electromagnetic (s=1), and gravitational (s=2) perturbations.

For the Tanner Framework echo analysis:
    K(ω) = T_∞² · R_s · e^{iωΔt} / (1 - R_∞ · R_s · e^{iωΔt})

Author: Tanner Q / TNSR-Q
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

# Physical constants
G = 6.67430e-11  # m³/kg/s²
c = 299792458    # m/s
M_sun = 1.989e30 # kg

@dataclass
class ScatteringResult:
    """Container for scattering computation results."""
    omega: float           # Dimensionless frequency ωM
    R_infinity: complex    # Reflection coefficient
    T_infinity: complex    # Transmission coefficient
    ell: int              # Angular momentum
    spin: int             # Field spin
    greybody: float       # Greybody factor Γ = |T|²
    phase_shift: float    # Scattering phase shift

# =============================================================================
# EFFECTIVE POTENTIALS
# =============================================================================

def regge_wheeler_potential(r: np.ndarray, M: float, ell: int, spin: int) -> np.ndarray:
    """
    Regge-Wheeler potential for scalar (s=0), EM (s=1), and GW (s=2).
    
    V(r) = (1 - 2M/r) [ℓ(ℓ+1)/r² + (1-s²)·2M/r³]
    
    Note: For s=2 (gravitational), the Zerilli potential differs but gives
    same |R|² and |T|² (isospectrality theorem). We use Regge-Wheeler form.
    """
    f = 1 - 2*M/r
    V = f * (ell*(ell+1)/r**2 + (1 - spin**2) * 2*M/r**3)
    return V

def zerilli_potential(r: np.ndarray, M: float, ell: int) -> np.ndarray:
    """
    Zerilli potential for gravitational perturbations (s=2).
    More accurate for even-parity GW modes.
    
    V_Z = (1-2M/r) · [2λ²(λ+1)r³ + 6λ²Mr² + 18λM²r + 18M³] / [r³(λr + 3M)²]
    where λ = (ℓ-1)(ℓ+2)/2
    """
    if ell < 2:
        return np.zeros_like(r)
    
    lam = (ell - 1) * (ell + 2) / 2
    f = 1 - 2*M/r
    
    numer = 2*lam**2*(lam+1)*r**3 + 6*lam**2*M*r**2 + 18*lam*M**2*r + 18*M**3
    denom = r**3 * (lam*r + 3*M)**2
    
    V = f * numer / denom
    return V

# =============================================================================
# TORTOISE COORDINATE
# =============================================================================

def tortoise_coordinate(r: np.ndarray, M: float) -> np.ndarray:
    """
    Tortoise coordinate: r* = r + 2M ln(r/2M - 1)
    
    Maps r ∈ (2M, ∞) → r* ∈ (-∞, +∞)
    """
    return r + 2*M * np.log(r/(2*M) - 1)

def inverse_tortoise(r_star: float, M: float, r_min: float = 2.001, r_max: float = 1e6) -> float:
    """
    Invert r*(r) numerically.
    """
    def f(r):
        return tortoise_coordinate(np.array([r]), M)[0] - r_star
    
    # Bisection search
    try:
        return brentq(f, r_min * M, r_max * M)
    except ValueError:
        # Outside range - return boundary
        if r_star < tortoise_coordinate(np.array([r_min * M]), M)[0]:
            return r_min * M
        return r_max * M

# =============================================================================
# WAVE EQUATION SOLVER
# =============================================================================

def solve_radial_equation(omega: float, M: float, ell: int, spin: int = 0,
                          r_star_min: float = -100, r_star_max: float = 100,
                          use_zerilli: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the radial wave equation in tortoise coordinates:
    
        d²ψ/dr*² + [ω² - V(r*)] ψ = 0
    
    with ingoing boundary condition at horizon (r* → -∞):
        ψ ~ e^{-iωr*}
    
    Returns:
        r_star: tortoise coordinate array
        psi: wavefunction
        dpsi: derivative dψ/dr*
    """
    # Build coordinate grid in r, then map to r*
    # Dense near horizon (r ~ 2M), coarse at infinity
    r_horizon = 2*M * 1.001  # Just outside horizon
    r_far = 500 * M          # Far zone
    
    # Adaptive grid
    n_points = 5000
    r_near = np.linspace(r_horizon, 5*M, n_points//2)
    r_far_grid = np.linspace(5*M, r_far, n_points//2)
    r_grid = np.unique(np.concatenate([r_near, r_far_grid]))
    
    r_star_grid = tortoise_coordinate(r_grid, M)
    
    # Select potential
    if use_zerilli and spin == 2:
        V_grid = zerilli_potential(r_grid, M, ell)
    else:
        V_grid = regge_wheeler_potential(r_grid, M, ell, spin)
    
    # Interpolate V(r*)
    V_interp = interp1d(r_star_grid, V_grid, kind='cubic', 
                        fill_value=(V_grid[0], 0), bounds_error=False)
    
    def wave_equation(r_star, y, omega):
        """
        dy/dr* = [ψ', ψ'']
        where ψ'' = -(ω² - V)ψ
        """
        psi, dpsi = y
        V = V_interp(r_star)
        d2psi = -(omega**2 - V) * psi
        return [dpsi, d2psi]
    
    # Boundary condition: ingoing wave at horizon
    # ψ ~ e^{-iωr*} → ψ = 1, ψ' = -iω at r*_min
    r_star_start = r_star_grid[0]
    r_star_end = r_star_grid[-1]
    
    y0 = [np.exp(-1j * omega * r_star_start), 
          -1j * omega * np.exp(-1j * omega * r_star_start)]
    
    # Integrate outward
    sol = solve_ivp(wave_equation, [r_star_start, r_star_end], y0, 
                   args=(omega,), method='RK45', max_step=0.5,
                   t_eval=r_star_grid, dense_output=True)
    
    if not sol.success:
        print(f"Warning: Integration failed for ω={omega:.4f}")
    
    return sol.t, sol.y[0], sol.y[1]

# =============================================================================
# EXTRACT REFLECTION COEFFICIENT
# =============================================================================

def extract_reflection_coefficient(r_star: np.ndarray, psi: np.ndarray, 
                                   dpsi: np.ndarray, omega: float) -> Tuple[complex, complex]:
    """
    At large r*, the solution has the form:
        ψ ~ A_in e^{-iωr*} + A_out e^{+iωr*}
    
    where A_in is the incident amplitude (normalized to 1 by BC)
    and A_out is the reflected amplitude.
    
    R_∞ = A_out / A_in
    T_∞ = (transmitted amplitude) - but with pure ingoing BC, |T|² = 1 - |R|²
    """
    # Extract at the far end of integration
    idx_far = -100  # Use last 100 points for stability
    
    psi_L = psi[idx_far]
    dpsi_L = dpsi[idx_far]
    r_star_L = r_star[idx_far]
    
    # Decompose: ψ = A_in e^{-iωr*} + A_out e^{+iωr*}
    #           ψ' = -iω A_in e^{-iωr*} + iω A_out e^{+iωr*}
    # 
    # Solving: A_out = (iωψ + ψ')/(2iω) e^{-iωr*}
    #          A_in = (iωψ - ψ')/(2iω) e^{+iωr*}
    
    exp_minus = np.exp(-1j * omega * r_star_L)
    exp_plus = np.exp(1j * omega * r_star_L)
    
    A_out = 0.5 * (psi_L + dpsi_L / (1j * omega)) * exp_minus
    A_in = 0.5 * (psi_L - dpsi_L / (1j * omega)) * exp_plus
    
    # Reflection coefficient
    if np.abs(A_in) < 1e-15:
        R_inf = 0.0 + 0j
    else:
        R_inf = A_out / A_in
    
    # Transmission from unitarity: |R|² + |T|² = 1
    T_inf_squared = 1 - np.abs(R_inf)**2
    T_inf = np.sqrt(max(0, T_inf_squared))  # Take positive root
    
    return R_inf, T_inf

# =============================================================================
# LOW FREQUENCY APPROXIMATIONS
# =============================================================================

def low_frequency_greybody(omega_M: float, ell: int, spin: int = 0) -> float:
    """
    Low-frequency approximation for greybody factor.
    
    Γ_ℓ(ω) ≈ A_ℓ (ωM)^{2ℓ+2} for ωM ≪ 1
    
    For ℓ=0 scalar: Γ_0 ≈ 4π (ωM)²
    For ℓ=2 s=2:    Γ_2 ≈ (2π/3)² (ωM)^6 (simplified)
    """
    # Coefficients from Unruh/Page greybody calculations
    if spin == 0:
        if ell == 0:
            return 4 * np.pi * omega_M**2
        elif ell == 1:
            return (np.pi / 9) * omega_M**4
        else:
            # General: Γ_ℓ ~ (ωM)^{2ℓ+2}
            return omega_M**(2*ell + 2)
    elif spin == 2:
        if ell == 2:
            return (4/45) * np.pi * omega_M**6
        else:
            return omega_M**(2*ell + 2)
    else:
        return omega_M**(2*ell + 2)

# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_reflection_coefficient(omega_M: float, ell: int, spin: int = 0,
                                   use_zerilli: bool = False) -> ScatteringResult:
    """
    Compute R_∞(ω) for given dimensionless frequency ωM.
    
    Parameters:
        omega_M: Dimensionless frequency (ω in units of 1/M)
        ell: Angular momentum quantum number
        spin: Field spin (0=scalar, 1=EM, 2=gravitational)
        use_zerilli: Use Zerilli potential for s=2 (more accurate)
    
    Returns:
        ScatteringResult with R_∞, T_∞, greybody factor, phase shift
    """
    M = 1.0  # Work in units where M = 1
    
    # Solve wave equation
    r_star, psi, dpsi = solve_radial_equation(omega_M, M, ell, spin, 
                                              use_zerilli=use_zerilli)
    
    # Extract scattering amplitudes
    R_inf, T_inf = extract_reflection_coefficient(r_star, psi, dpsi, omega_M)
    
    # Greybody factor
    greybody = np.abs(T_inf)**2
    
    # Phase shift
    phase = np.angle(R_inf)
    
    return ScatteringResult(
        omega=omega_M,
        R_infinity=R_inf,
        T_infinity=T_inf,
        ell=ell,
        spin=spin,
        greybody=greybody,
        phase_shift=phase
    )

def compute_reflection_spectrum(omega_range: np.ndarray, ell: int, spin: int = 0,
                                use_zerilli: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute R_∞(ω) over a frequency range.
    """
    results = {
        'omega': omega_range,
        'R_magnitude': np.zeros(len(omega_range)),
        'R_phase': np.zeros(len(omega_range)),
        'T_magnitude': np.zeros(len(omega_range)),
        'greybody': np.zeros(len(omega_range)),
        'R_complex': np.zeros(len(omega_range), dtype=complex)
    }
    
    print(f"Computing R_∞(ω) for ℓ={ell}, s={spin}...")
    
    for i, omega in enumerate(omega_range):
        if omega < 1e-4:
            # Use low-freq approximation for very small ω
            greybody = low_frequency_greybody(omega, ell, spin)
            results['greybody'][i] = greybody
            results['R_magnitude'][i] = np.sqrt(1 - greybody)
            results['T_magnitude'][i] = np.sqrt(greybody)
            results['R_phase'][i] = 0  # Unknown phase in approximation
            results['R_complex'][i] = results['R_magnitude'][i]
        else:
            try:
                res = compute_reflection_coefficient(omega, ell, spin, use_zerilli)
                results['R_magnitude'][i] = np.abs(res.R_infinity)
                results['R_phase'][i] = res.phase_shift
                results['T_magnitude'][i] = np.abs(res.T_infinity)
                results['greybody'][i] = res.greybody
                results['R_complex'][i] = res.R_infinity
            except Exception as e:
                print(f"  Error at ω={omega:.4f}: {e}")
                results['R_magnitude'][i] = np.nan
        
        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(omega_range)}")
    
    return results

# =============================================================================
# ECHO TRANSFER FUNCTION
# =============================================================================

def echo_transfer_function(omega: np.ndarray, R_inf: np.ndarray, 
                           R_surface: float, delta_t: float) -> np.ndarray:
    """
    Compute the echo transfer function K(ω) from Appendix D.
    
    K(ω) = T_∞² · R_s · e^{iωΔt} / (1 - R_∞ · R_s · e^{iωΔt})
    
    Parameters:
        omega: Frequency array
        R_inf: Complex reflection coefficient array
        R_surface: Surface reflectivity (real, 0 < R_s < 1)
        delta_t: Echo time delay (in units of M)
    """
    T_inf_squared = 1 - np.abs(R_inf)**2
    
    phase = np.exp(1j * omega * delta_t)
    
    K = T_inf_squared * R_surface * phase / (1 - R_inf * R_surface * phase)
    
    return K

def optimal_detection_frequency(M_solar: float, R_surface: float = 0.3) -> float:
    """
    Estimate optimal frequency for echo detection.
    
    Balance: Higher ω → more transmission through barrier
            Lower ω → stronger thermal occupation
    
    For 30 M_☉, Hawking temperature T_eff ≈ 1/(8πM) corresponds to
    ω_peak ~ T_eff ~ 1/(8πM) in geometric units.
    """
    # Convert to geometric units
    M_geom = M_solar * M_sun * G / c**2  # in meters
    
    # Hawking temperature in geometric units
    T_eff = 1 / (8 * np.pi)  # T_eff M in dimensionless units
    
    # Peak of Planckian spectrum: ω_peak ≈ 2.82 T_eff
    omega_peak = 2.82 * T_eff
    
    return omega_peak  # Returns ωM (dimensionless)

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_reflection_analysis(outdir: Path, M_solar: float = 30.0):
    """
    Create comprehensive R_∞(ω) analysis plots.
    """
    print(f"\n{'='*60}")
    print(f"SCHWARZSCHILD REFLECTION ANALYSIS")
    print(f"M = {M_solar} M_☉")
    print(f"{'='*60}\n")
    
    # Frequency range: from low-ω (thermal peak) to high-ω (QNM region)
    # Hawking peak at ωM ~ 1/(8π) ≈ 0.04
    omega_range = np.linspace(0.01, 2.0, 100)
    
    # Compute for ℓ=0,1,2 scalars and ℓ=2 GW
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Reflection magnitude |R_∞|²
    ax1 = axes[0, 0]
    for ell in [0, 1, 2]:
        results = compute_reflection_spectrum(omega_range, ell, spin=0)
        ax1.plot(omega_range, np.abs(results['R_magnitude'])**2, 
                label=f'ℓ={ell} (scalar)', linewidth=2)
    
    # Also plot s=2, ℓ=2 for GW
    results_gw = compute_reflection_spectrum(omega_range, ell=2, spin=2)
    ax1.plot(omega_range, np.abs(results_gw['R_magnitude'])**2, 
            'k--', label='ℓ=2 (GW)', linewidth=2)
    
    ax1.set_xlabel(r'$\omega M$')
    ax1.set_ylabel(r'$|R_\infty|^2$')
    ax1.set_title('Reflection Coefficient Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 1.05)
    
    # Panel 2: Greybody factor Γ = |T|²
    ax2 = axes[0, 1]
    for ell in [0, 1, 2]:
        results = compute_reflection_spectrum(omega_range, ell, spin=0)
        ax2.semilogy(omega_range, results['greybody'], 
                    label=f'ℓ={ell} (scalar)', linewidth=2)
    
    ax2.semilogy(omega_range, results_gw['greybody'], 
                'k--', label='ℓ=2 (GW)', linewidth=2)
    
    # Mark Hawking peak
    omega_hawking = 1/(8*np.pi)
    ax2.axvline(omega_hawking, color='red', ls=':', label=r'$\omega_{Hawking}$')
    
    ax2.set_xlabel(r'$\omega M$')
    ax2.set_ylabel(r'$\Gamma(\omega) = |T_\infty|^2$')
    ax2.set_title('Greybody Factor (Transmission)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Phase shift
    ax3 = axes[1, 0]
    results_l2 = compute_reflection_spectrum(omega_range, ell=2, spin=0)
    ax3.plot(omega_range, np.unwrap(results_l2['R_phase']), 'b-', linewidth=2)
    ax3.set_xlabel(r'$\omega M$')
    ax3.set_ylabel(r'Phase $\arg(R_\infty)$')
    ax3.set_title('Scattering Phase Shift (ℓ=2 scalar)')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Echo transfer function |K(ω)|
    ax4 = axes[1, 1]
    
    # Echo time delay for 30 M_☉
    # Δt = 2M[3 + 2ln(M/ℓ_Pl)]
    M_meters = M_solar * M_sun * G / c**2
    l_Pl = 1.616e-35  # meters
    delta_t_M = 2 * (3 + 2*np.log(M_meters/l_Pl))  # in units of M
    
    print(f"\nEcho time delay: Δt = {delta_t_M:.1f} M = {delta_t_M * M_meters/c * 1000:.1f} ms")
    
    for R_s in [0.1, 0.3, 0.5]:
        K = echo_transfer_function(omega_range, results_l2['R_complex'], R_s, delta_t_M)
        ax4.semilogy(omega_range, np.abs(K)**2, label=f'$R_s={R_s}$', linewidth=2)
    
    ax4.set_xlabel(r'$\omega M$')
    ax4.set_ylabel(r'$|K(\omega)|^2$')
    ax4.set_title(f'Echo Transfer Function (Δt = {delta_t_M:.0f}M)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Schwarzschild Black Hole Scattering Analysis (M = {M_solar} M$_\\odot$)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    png_path = outdir / "schwarzschild_reflection_analysis.png"
    pdf_path = outdir / "schwarzschild_reflection_analysis.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {png_path}")
    
    return png_path, pdf_path

# =============================================================================
# TABULATED OUTPUT FOR APPENDIX
# =============================================================================

def generate_appendix_table(outdir: Path, M_solar: float = 30.0):
    """
    Generate tabulated R_∞(ω) data for paper appendix.
    """
    omega_range = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 
                           0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
    
    results_s0_l2 = compute_reflection_spectrum(omega_range, ell=2, spin=0)
    results_s2_l2 = compute_reflection_spectrum(omega_range, ell=2, spin=2)
    
    # Write table
    table_path = outdir / "reflection_coefficient_table.txt"
    with open(table_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TABLE: Schwarzschild Reflection Coefficients R_∞(ω)\n")
        f.write(f"For M = {M_solar} M_☉, ℓ = 2\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'ωM':>8} | {'|R_∞|² (s=0)':>12} | {'|R_∞|² (s=2)':>12} | "
               f"{'Γ (s=0)':>12} | {'Γ (s=2)':>12}\n")
        f.write("-"*80 + "\n")
        
        for i, omega in enumerate(omega_range):
            R2_s0 = results_s0_l2['R_magnitude'][i]**2
            R2_s2 = results_s2_l2['R_magnitude'][i]**2
            G_s0 = results_s0_l2['greybody'][i]
            G_s2 = results_s2_l2['greybody'][i]
            
            f.write(f"{omega:8.3f} | {R2_s0:12.6f} | {R2_s2:12.6f} | "
                   f"{G_s0:12.6e} | {G_s2:12.6e}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Notes:\n")
        f.write("- |R_∞|² + Γ = 1 (unitarity)\n")
        f.write("- s=0: scalar field, s=2: gravitational waves\n")
        f.write("- Γ = greybody factor = |T_∞|²\n")
        f.write("- Hawking peak frequency: ωM ≈ 0.04\n")
    
    print(f"\nTable saved: {table_path}")
    return table_path

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    outdir = Path("/mnt/user-data/outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    png_path, pdf_path = plot_reflection_analysis(outdir, M_solar=30.0)
    table_path = generate_appendix_table(outdir, M_solar=30.0)
    
    # Quick verification
    print("\n" + "="*60)
    print("VERIFICATION: Single point check")
    print("="*60)
    
    result = compute_reflection_coefficient(omega_M=0.3, ell=2, spin=2)
    print(f"\nωM = 0.3, ℓ=2, s=2 (gravitational):")
    print(f"  R_∞ = {result.R_infinity:.6f}")
    print(f"  |R_∞|² = {np.abs(result.R_infinity)**2:.6f}")
    print(f"  |T_∞|² = {result.greybody:.6f}")
    print(f"  Unitarity check: |R|² + |T|² = {np.abs(result.R_infinity)**2 + result.greybody:.6f}")
    
    print(f"\n✓ Analysis complete. Files in {outdir}")

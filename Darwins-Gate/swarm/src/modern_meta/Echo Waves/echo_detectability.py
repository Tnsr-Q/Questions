"""
Complete Echo Detectability Analysis
====================================
Integrates R_∞(ω) with the Tanner Framework echo transfer function K(ω)
to compute optimal detection frequencies and SNR predictions.

Connects to Appendix D: Echo Transfer Function and Detectability
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# SCHWARZSCHILD R_∞(ω) COMPUTATION
# =============================================================================

def compute_R_infinity(omega_M: float, ell: int = 2, spin: int = 2, M: float = 1.0) -> complex:
    """
    Compute R_∞(ω) by solving the Regge-Wheeler equation with ingoing BC.
    """
    if omega_M < 1e-4:
        # Low-frequency limit: nearly total reflection
        return 1.0 - 0j
    
    # Grid setup
    r_horizon = 2*M * 1.001
    r_far = 200 * M
    r_grid = np.linspace(r_horizon, r_far, 2000)
    r_star = r_grid + 2*M * np.log(r_grid/(2*M) - 1)
    
    # Regge-Wheeler potential
    f = 1 - 2*M/r_grid
    V = f * (ell*(ell+1)/r_grid**2 + (1 - spin**2) * 2*M/r_grid**3)
    V_interp = interp1d(r_star, V, kind='cubic', fill_value=(V[0], 0), bounds_error=False)
    
    def wave_eq(t, y, omega):
        psi, dpsi = y
        return [dpsi, -(omega**2 - V_interp(t)) * psi]
    
    # Ingoing BC at horizon
    r_star_0 = r_star[0]
    y0 = [np.exp(-1j * omega_M * r_star_0), 
          -1j * omega_M * np.exp(-1j * omega_M * r_star_0)]
    
    sol = solve_ivp(wave_eq, [r_star[0], r_star[-1]], y0, 
                   args=(omega_M,), method='RK45', max_step=1.0)
    
    # Extract R_∞ from asymptotic form
    psi_L, dpsi_L = sol.y[0, -1], sol.y[1, -1]
    r_L = sol.t[-1]
    
    A_out = 0.5 * (psi_L + dpsi_L/(1j*omega_M)) * np.exp(-1j*omega_M*r_L)
    A_in = 0.5 * (psi_L - dpsi_L/(1j*omega_M)) * np.exp(1j*omega_M*r_L)
    
    return A_out / A_in if np.abs(A_in) > 1e-15 else 0.0

# =============================================================================
# ECHO TRANSFER FUNCTION K(ω) - from Appendix D Eq. (A.62)
# =============================================================================

def echo_transfer_function(omega: float, R_inf: complex, R_surface: float, 
                           delta_t: float, n_max: int = 20) -> complex:
    """
    K(ω) = T_∞² · R_s · e^{iωΔt} / (1 - R_∞ · R_s · e^{iωΔt})
    
    Parameters:
        omega: Dimensionless frequency ωM
        R_inf: Complex R_∞(ω)
        R_surface: Surface reflectivity (2-2 hole parameter)
        delta_t: Echo delay in units of M
    """
    T_inf_squared = 1 - np.abs(R_inf)**2
    phase = np.exp(1j * omega * delta_t)
    
    denominator = 1 - R_inf * R_surface * phase
    
    # Avoid division by zero near resonances
    if np.abs(denominator) < 1e-10:
        denominator = 1e-10
    
    K = T_inf_squared * R_surface * phase / denominator
    return K

def individual_echo_amplitude(omega: float, R_inf: complex, R_surface: float, 
                              delta_t: float, n: int) -> complex:
    """
    Amplitude of the n-th echo: h^{(n)} / h_merger
    
    From Appendix D Eq. (A.63):
    h^{(n)}_echo(ω) = h_merger(ω) · T_∞² · R_∞^{n-1} · R_s^n · e^{inωΔt}
    """
    T_inf_squared = 1 - np.abs(R_inf)**2
    T_inf = np.sqrt(max(0, T_inf_squared))
    
    amplitude = T_inf**2 * R_inf**(n-1) * R_surface**n * np.exp(1j * n * omega * delta_t)
    return amplitude

# =============================================================================
# DETECTABILITY LIKELIHOOD ℒ(ω)
# =============================================================================

def detection_likelihood(omega: float, R_inf: complex, R_surface: float, 
                         delta_t: float, noise_model: str = 'LIGO') -> float:
    """
    Compute detection likelihood ℒ(ω) for echo search.
    
    ℒ(ω) ∝ |K(ω)|² / S_n(ω)
    
    where S_n(ω) is the detector noise power spectral density.
    """
    K = echo_transfer_function(omega, R_inf, R_surface, delta_t)
    
    # Simple noise model (normalized)
    # LIGO is most sensitive around 100-300 Hz
    # For 30 M_☉: f = ω/(2πM) where M in seconds
    # M_30 = 30 * 1.5km/c ≈ 1.5e-4 s, so f = ω / (2π × 1.5e-4) ≈ 1000 ω Hz
    
    if noise_model == 'LIGO':
        # Simplified LIGO noise curve (peak sensitivity at ωM ~ 0.1)
        f_opt = 0.1  # optimal dimensionless frequency
        S_n = 1 + 10 * (omega / f_opt - 1)**2 + 100 * (omega / f_opt)**(-4)
    else:
        S_n = 1.0
    
    likelihood = np.abs(K)**2 / S_n
    return likelihood

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_echo_analysis(M_solar: float = 30.0, R_surface: float = 0.3, outdir: Path = None):
    """
    Complete echo detectability analysis.
    """
    print("="*70)
    print(f"ECHO DETECTABILITY ANALYSIS")
    print(f"M = {M_solar} M_☉, R_s = {R_surface}")
    print("="*70)
    
    # Physical parameters
    M_kg = M_solar * 1.989e30
    G = 6.674e-11
    c = 2.998e8
    l_Pl = 1.616e-35
    
    M_meters = G * M_kg / c**2  # M in meters
    M_seconds = M_meters / c    # M in seconds
    
    # Echo time delay: Δt = 2M[3 + 2ln(M/ℓ_Pl)]
    delta_t_M = 2 * (3 + 2*np.log(M_meters/l_Pl))  # in units of M
    delta_t_seconds = delta_t_M * M_seconds
    delta_t_ms = delta_t_seconds * 1000
    
    print(f"\nPhysical parameters:")
    print(f"  M = {M_meters/1000:.2f} km = {M_seconds*1000:.4f} ms")
    print(f"  Echo delay Δt = {delta_t_M:.1f} M = {delta_t_ms:.1f} ms")
    print(f"  Hawking T = {1/(8*np.pi*M_kg*G/c**3):.2e} K")
    
    # Frequency grid
    omega_range = np.linspace(0.01, 1.5, 200)
    
    # Compute R_∞(ω)
    print(f"\nComputing R_∞(ω) for {len(omega_range)} frequencies...")
    R_inf_array = np.array([compute_R_infinity(w, ell=2, spin=2) for w in omega_range])
    print("  Done.")
    
    # Compute transfer function and likelihood
    K_array = np.array([echo_transfer_function(w, R_inf_array[i], R_surface, delta_t_M) 
                       for i, w in enumerate(omega_range)])
    L_array = np.array([detection_likelihood(w, R_inf_array[i], R_surface, delta_t_M) 
                       for i, w in enumerate(omega_range)])
    
    # Find optimal frequency
    idx_optimal = np.argmax(L_array)
    omega_optimal = omega_range[idx_optimal]
    
    # Convert to physical frequency
    f_optimal_Hz = omega_optimal / (2 * np.pi * M_seconds)
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"  Optimal detection frequency: ωM = {omega_optimal:.3f}")
    print(f"                              f = {f_optimal_Hz:.1f} Hz")
    print(f"  |R_∞|² at optimal: {np.abs(R_inf_array[idx_optimal])**2:.4f}")
    print(f"  |K|² at optimal: {np.abs(K_array[idx_optimal])**2:.4e}")
    print(f"  Likelihood at optimal: {L_array[idx_optimal]:.4e}")
    
    # Individual echo amplitudes at optimal frequency
    print(f"\nIndividual echo amplitudes at ω_opt = {omega_optimal:.3f}:")
    R_opt = R_inf_array[idx_optimal]
    for n in range(1, 6):
        amp = individual_echo_amplitude(omega_optimal, R_opt, R_surface, delta_t_M, n)
        print(f"  Echo {n}: |h^({n})/h_merger| = {np.abs(amp):.4e}")
    
    # Plotting
    if outdir is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: R_∞ and T_∞
        ax1 = axes[0, 0]
        ax1.plot(omega_range, np.abs(R_inf_array)**2, 'b-', lw=2, label=r'$|R_\infty|^2$')
        ax1.plot(omega_range, 1 - np.abs(R_inf_array)**2, 'r--', lw=2, label=r'$\Gamma = |T_\infty|^2$')
        ax1.axvline(omega_optimal, color='green', ls=':', lw=2, alpha=0.7)
        ax1.set_xlabel(r'$\omega M$')
        ax1.set_ylabel('Coefficient')
        ax1.set_title('Schwarzschild Scattering Coefficients (ℓ=2, s=2)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Transfer function |K|²
        ax2 = axes[0, 1]
        ax2.semilogy(omega_range, np.abs(K_array)**2, 'b-', lw=2)
        ax2.axvline(omega_optimal, color='green', ls=':', lw=2, 
                   label=f'Optimal: ωM={omega_optimal:.3f}')
        ax2.set_xlabel(r'$\omega M$')
        ax2.set_ylabel(r'$|K(\omega)|^2$')
        ax2.set_title(f'Echo Transfer Function ($R_s={R_surface}$, $\Delta t={delta_t_M:.0f}M$)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Detection likelihood
        ax3 = axes[1, 0]
        ax3.plot(omega_range, L_array / np.max(L_array), 'g-', lw=2)
        ax3.axvline(omega_optimal, color='red', ls='--', lw=2)
        ax3.fill_between(omega_range, L_array / np.max(L_array), alpha=0.3)
        ax3.set_xlabel(r'$\omega M$')
        ax3.set_ylabel(r'$\mathcal{L}(\omega)$ (normalized)')
        ax3.set_title('Detection Likelihood')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Echo train decay
        ax4 = axes[1, 1]
        n_echoes = np.arange(1, 11)
        echo_amps = [np.abs(individual_echo_amplitude(omega_optimal, R_opt, R_surface, delta_t_M, n)) 
                    for n in n_echoes]
        ax4.semilogy(n_echoes, echo_amps, 'ko-', markersize=8, lw=2)
        ax4.set_xlabel('Echo number n')
        ax4.set_ylabel(r'$|h^{(n)}_{echo}| / |h_{merger}|$')
        ax4.set_title(f'Echo Train Decay (at $\omega M = {omega_optimal:.3f}$)')
        ax4.grid(True, alpha=0.3)
        
        # Decay fit
        decay_rate = np.abs(R_opt * R_surface)
        ax4.plot(n_echoes, echo_amps[0] * decay_rate**(n_echoes-1), 'r--', 
                label=f'Decay: $(R_\\infty R_s)^n$, rate={decay_rate:.3f}')
        ax4.legend()
        
        plt.suptitle(f'Echo Analysis: M={M_solar} M$_\\odot$, $R_s$={R_surface}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        png_path = outdir / f"echo_analysis_M{int(M_solar)}_Rs{int(R_surface*100)}.png"
        pdf_path = outdir / f"echo_analysis_M{int(M_solar)}_Rs{int(R_surface*100)}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlots saved to {outdir}")
        return png_path
    
    return None

# =============================================================================
# PARAMETER SPACE SCAN
# =============================================================================

def scan_detectability(outdir: Path):
    """
    Scan detectability across R_s parameter space.
    """
    print("\n" + "="*70)
    print("PARAMETER SPACE SCAN: R_s sensitivity")
    print("="*70)
    
    R_s_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    M_solar = 30.0
    
    # Compute delta_t once
    M_meters = M_solar * 1.989e30 * 6.674e-11 / (2.998e8)**2
    l_Pl = 1.616e-35
    delta_t_M = 2 * (3 + 2*np.log(M_meters/l_Pl))
    
    omega_range = np.linspace(0.01, 1.5, 100)
    
    print(f"\n{'R_s':>6} | {'ω_opt':>8} | {'f_opt (Hz)':>12} | {'|K|²_max':>12} | {'SNR_ratio':>10}")
    print("-"*70)
    
    # Precompute R_∞
    R_inf_array = np.array([compute_R_infinity(w, ell=2, spin=2) for w in omega_range])
    
    M_seconds = M_meters / 2.998e8
    
    for R_s in R_s_values:
        K_array = np.array([echo_transfer_function(w, R_inf_array[i], R_s, delta_t_M) 
                          for i, w in enumerate(omega_range)])
        
        idx_opt = np.argmax(np.abs(K_array)**2)
        omega_opt = omega_range[idx_opt]
        f_opt = omega_opt / (2 * np.pi * M_seconds)
        K2_max = np.abs(K_array[idx_opt])**2
        
        # SNR ratio = sqrt(|K|²_max) × sqrt(T_merger/T_echo)
        # Approximate: SNR_echo/SNR_merger ≈ |K|_max for comparable bandwidths
        SNR_ratio = np.sqrt(K2_max)
        
        print(f"{R_s:6.2f} | {omega_opt:8.3f} | {f_opt:12.1f} | {K2_max:12.4e} | {SNR_ratio:10.4f}")
    
    print("\nNote: Detection requires SNR_echo > 5-8 for confident detection")
    print("      For GW150914-like events (SNR_merger ~ 20):")
    print("      Detectable if SNR_ratio > 0.25 → R_s > ~0.3")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    outdir = Path("/mnt/user-data/outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Main analysis
    png_path = run_echo_analysis(M_solar=30.0, R_surface=0.3, outdir=outdir)
    
    # Parameter scan
    scan_detectability(outdir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

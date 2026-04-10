"""
Parametric Model for Complex Surface Reflectivity R_s(ω)
========================================================
Provides physically-motivated parametric models for both magnitude and phase
of the surface reflectivity R_s(ω) for horizonless compact objects.

Key physics included:
1. Frequency-dependent magnitude due to:
   - Surface absorption (phonon coupling)
   - Tunneling through potential barrier at surface
   - Material-dependent dissipation (viscoelastic losses)
   
2. Frequency-dependent phase due to:
   - Wave propagation time in boundary layer
   - Resonance effects from surface modes
   - Causality constraints (Kramers-Kronig relations)

Models implemented:
- Lorentzian oscillator (viscoelastic solid)
- Drude-like surface (dissipative plasma)
- Power-law dissipation (fractal surface)
- Exponentially suppressed (quantum tunneling)

Author: Tanner Q / TNSR-Q
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import erf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import warnings

# =============================================================================
# PARAMETRIC MODELS FOR MAGNITUDE |R_s(ω)|
# =============================================================================

@dataclass
class MagnitudeModel:
    """Base class for magnitude models."""
    name: str
    params: Dict[str, float]
    
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        """Compute |R_s| at given frequencies."""
        raise NotImplementedError


class LorentzianMagnitude(MagnitudeModel):
    """
    Lorentzian oscillator model for viscoelastic solid surface.
    
    |R_s| = R0 / sqrt[(1 - (ω/ω0)²)² + (Γ ω)²]
    
    Parameters:
        R0: Low-frequency reflectivity (0 < R0 ≤ 1)
        ω0: Resonance frequency (Hz)
        Γ: Damping rate (1/Hz)
        α: High-frequency exponent (optional power-law tail)
    """
    def __init__(self, R0: float = 0.8, omega0: float = 200.0, 
                 Gamma: float = 0.1, alpha: float = 0.0):
        params = {'R0': R0, 'omega0': omega0, 'Gamma': Gamma, 'alpha': alpha}
        super().__init__('Lorentzian', params)
    
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        R0 = self.params['R0']
        omega0 = self.params['omega0']
        Gamma = self.params['Gamma']
        alpha = self.params['alpha']
        
        # Main Lorentzian form
        x = omega / omega0
        denominator = (1 - x**2)**2 + (Gamma * omega)**2
        magnitude = R0 / np.sqrt(denominator)
        
        # Optional high-frequency power-law suppression
        if alpha > 0:
            magnitude *= 1 / (1 + (omega/(2*omega0))**(2*alpha))
        
        return np.clip(magnitude, 0, 1)


class DrudeMagnitude(MagnitudeModel):
    """
    Drude-like model for dissipative surface (plasma-like).
    
    |R_s| = 1 / sqrt[1 + (ω/σ)²] with frequency cutoff.
    
    Parameters:
        σ: Conductivity parameter (Hz)
        ω_c: Cutoff frequency for saturation (Hz)
        R_min: Minimum reflectivity at high frequency
    """
    def __init__(self, sigma: float = 100.0, omega_cutoff: float = 500.0,
                 R_min: float = 0.1):
        params = {'sigma': sigma, 'omega_cutoff': omega_cutoff, 'R_min': R_min}
        super().__init__('Drude', params)
    
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        sigma = self.params['sigma']
        omega_c = self.params['omega_cutoff']
        R_min = self.params['R_min']
        
        # Drude-like form
        magnitude = 1 / np.sqrt(1 + (omega/sigma)**2)
        
        # Apply cutoff to prevent going to zero
        magnitude = R_min + (1 - R_min) * magnitude
        
        # Smooth saturation
        magnitude *= 0.5 * (1 + erf((omega_c - omega)/(0.2*omega_c)))
        
        return np.clip(magnitude, R_min, 1)


class PowerLawMagnitude(MagnitudeModel):
    """
    Power-law dissipation model (fractal/rough surface).
    
    |R_s| = R0 * exp[- (ω/ω_c)^β]
    
    Parameters:
        R0: Low-frequency reflectivity
        ω_c: Characteristic frequency (Hz)
        β: Power-law exponent (typically 0.5-2)
        ω_max: Frequency where reflectivity saturates
    """
    def __init__(self, R0: float = 0.9, omega_c: float = 300.0,
                 beta: float = 1.0, omega_max: float = 1000.0):
        params = {'R0': R0, 'omega_c': omega_c, 'beta': beta, 'omega_max': omega_max}
        super().__init__('PowerLaw', params)
    
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        R0 = self.params['R0']
        omega_c = self.params['omega_c']
        beta = self.params['beta']
        omega_max = self.params['omega_max']
        
        # Power-law suppression
        magnitude = R0 * np.exp(-(omega/omega_c)**beta)
        
        # Prevent going to zero at high frequency
        saturation = 0.5 * (1 + erf((omega_max - omega)/(0.15*omega_max)))
        magnitude = np.maximum(magnitude, 0.05) * saturation
        
        return np.clip(magnitude, 0.05, R0)


class ExponentialTunneling(MagnitudeModel):
    """
    Exponential suppression due to quantum tunneling through surface barrier.
    
    |R_s| = 1 - exp[- (E_tunnel / ħω)^γ]
    
    Parameters:
        E_tunnel: Tunneling energy scale (Hz)
        γ: Tunneling exponent
        R_bg: Background reflectivity (imperfect absorption)
    """
    def __init__(self, E_tunnel: float = 50.0, gamma: float = 1.5,
                 R_bg: float = 0.2):
        params = {'E_tunnel': E_tunnel, 'gamma': gamma, 'R_bg': R_bg}
        super().__init__('ExponentialTunneling', params)
    
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        E_t = self.params['E_tunnel']
        gamma = self.params['gamma']
        R_bg = self.params['R_bg']
        
        # Tunneling probability
        tunneling_prob = np.exp(-(E_t/(omega + 1e-10))**gamma)
        magnitude = R_bg + (1 - R_bg) * (1 - tunneling_prob)
        
        return np.clip(magnitude, R_bg, 1)


# =============================================================================
# PARAMETRIC MODELS FOR PHASE arg[R_s(ω)]
# =============================================================================

@dataclass
class PhaseModel:
    """Base class for phase models."""
    name: str
    params: Dict[str, float]
    
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        """Compute arg(R_s) at given frequencies."""
        raise NotImplementedError


class LinearDispersion(PhaseModel):
    """
    Linear phase dispersion (time delay).
    
    arg(R_s) = τ * ω + φ0
    
    Parameters:
        τ: Time delay (seconds)
        φ0: Constant phase offset (radians)
    """
    def __init__(self, tau: float = 1e-3, phi0: float = 0.3*np.pi):
        params = {'tau': tau, 'phi0': phi0}
        super().__init__('LinearDispersion', params)
    
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        tau = self.params['tau']
        phi0 = self.params['phi0']
        return tau * omega + phi0


class QuadraticDispersion(PhaseModel):
    """
    Quadratic phase dispersion (frequency-dependent time delay).
    
    arg(R_s) = τ0 * ω + ½ * τ1 * ω² + φ0
    
    Parameters:
        τ0: Linear time delay (s)
        τ1: Quadratic dispersion (s²/rad)
        φ0: Constant phase offset
    """
    def __init__(self, tau0: float = 1e-3, tau1: float = -1e-6,
                 phi0: float = 0.2*np.pi):
        params = {'tau0': tau0, 'tau1': tau1, 'phi0': phi0}
        super().__init__('QuadraticDispersion', params)
    
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        tau0 = self.params['tau0']
        tau1 = self.params['tau1']
        phi0 = self.params['phi0']
        return tau0 * omega + 0.5 * tau1 * omega**2 + phi0


class ResonantPhase(PhaseModel):
    """
    Resonant phase model (Lorentzian-like phase shift).
    
    arg(R_s) = arctan[Γω/(ω0² - ω²)] + φ0
    
    Parameters:
        ω0: Resonance frequency (Hz)
        Γ: Damping width (Hz)
        φ0: Background phase
        Δφ_max: Maximum phase shift
    """
    def __init__(self, omega0: float = 150.0, Gamma: float = 30.0,
                 phi0: float = 0.1*np.pi, delta_phi_max: float = 0.5*np.pi):
        params = {'omega0': omega0, 'Gamma': Gamma, 'phi0': phi0, 
                 'delta_phi_max': delta_phi_max}
        super().__init__('ResonantPhase', params)
    
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        omega0 = self.params['omega0']
        Gamma = self.params['Gamma']
        phi0 = self.params['phi0']
        delta_phi_max = self.params['delta_phi_max']
        
        # Lorentzian phase response
        phase = np.arctan2(Gamma * omega, omega0**2 - omega**2)
        
        # Scale to desired maximum phase shift
        phase = phi0 + delta_phi_max * (phase / np.pi)
        
        return phase


class LogarithmicPhase(PhaseModel):
    """
    Logarithmic phase (causality/fractal surface).
    
    arg(R_s) = A * log(1 + ω/ω_ref) + φ0
    
    Parameters:
        A: Amplitude (radians)
        ω_ref: Reference frequency (Hz)
        φ0: Offset phase
    """
    def __init__(self, A: float = 0.2*np.pi, omega_ref: float = 10.0,
                 phi0: float = 0.0):
        params = {'A': A, 'omega_ref': omega_ref, 'phi0': phi0}
        super().__init__('LogarithmicPhase', params)
    
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        A = self.params['A']
        omega_ref = self.params['omega_ref']
        phi0 = self.params['phi0']
        
        return phi0 + A * np.log1p(omega / omega_ref)


# =============================================================================
# COMPLEX SURFACE REFLECTIVITY MODEL
# =============================================================================

@dataclass
class ComplexSurfaceReflectivity:
    """
    Complete model for complex surface reflectivity R_s(ω).
    
    Combines magnitude and phase models with optional Kramers-Kronig
    consistency enforcement.
    """
    magnitude_model: MagnitudeModel
    phase_model: PhaseModel
    enforce_kk: bool = False
    kk_tolerance: float = 1e-3
    
    def __call__(self, omega: np.ndarray) -> np.ndarray:
        """
        Compute complex R_s(ω).
        
        Parameters:
            omega: Angular frequency array (rad/s)
            
        Returns:
            R_s: Complex reflectivity array
        """
        magnitude = self.magnitude_model(omega)
        phase = self.phase_model(omega)
        
        R_s = magnitude * np.exp(1j * phase)
        
        # Optional Kramers-Kronig consistency check/correction
        if self.enforce_kk:
            R_s = self.apply_kramers_kronig(omega, R_s)
        
        return R_s
    
    def apply_kramers_kronig(self, omega: np.ndarray, R_s: np.ndarray) -> np.ndarray:
        """
        Enforce Kramers-Kronig relations on R_s(ω).
        
        Uses Hilbert transform to ensure causality.
        """
        from scipy.fft import fft, ifft, fftfreq
        
        # Ensure frequencies are equally spaced
        if not np.allclose(np.diff(omega), omega[1] - omega[0]):
            warnings.warn("Kramers-Kronig requires equally spaced frequencies")
            return R_s
        
        N = len(omega)
        domega = omega[1] - omega[0]
        
        # Get magnitude and phase
        magnitude = np.abs(R_s)
        phase = np.angle(R_s)
        
        # Kramers-Kronig: log|R| and arg(R) are Hilbert transform pairs
        log_magnitude = np.log(np.clip(magnitude, 1e-10, 1))
        
        # Compute Hilbert transform of log|R| to get phase
        H_logR = self.hilbert_transform(log_magnitude, domega)
        
        # Adjust phase to satisfy KK relations
        phase_kk = H_logR
        
        # Blend original phase with KK phase
        alpha = self.kk_tolerance
        phase_corrected = (1-alpha)*phase + alpha*phase_kk
        
        # Reconstruct R_s with corrected phase
        R_s_kk = magnitude * np.exp(1j * phase_corrected)
        
        return R_s_kk
    
    @staticmethod
    def hilbert_transform(x: np.ndarray, dx: float) -> np.ndarray:
        """Compute Hilbert transform via FFT."""
        N = len(x)
        if N % 2 == 0:
            k = np.arange(N)
            H = np.zeros(N)
            H[1:N//2] = 1
            H[N//2+1:] = -1
        else:
            k = np.arange(N)
            H = np.zeros(N)
            H[1:(N+1)//2] = 1
            H[(N+1)//2:] = -1
        
        X = np.fft.fft(x)
        Hx = np.fft.ifft(1j * H * X)
        
        return np.real(Hx)
    
    def get_params(self) -> Dict:
        """Get all parameters as a dictionary."""
        params = {
            'magnitude_model': self.magnitude_model.name,
            'magnitude_params': self.magnitude_model.params.copy(),
            'phase_model': self.phase_model.name,
            'phase_params': self.phase_model.params.copy(),
            'enforce_kk': self.enforce_kk
        }
        return params
    
    def set_params(self, params: Dict) -> None:
        """Update model parameters."""
        if 'magnitude_params' in params:
            self.magnitude_model.params.update(params['magnitude_params'])
        if 'phase_params' in params:
            self.phase_model.params.update(params['phase_params'])


# =============================================================================
# MODEL FITTING AND OPTIMIZATION
# =============================================================================

class ReflectivityFitter:
    """
    Fit complex R_s(ω) model to data or theoretical constraints.
    """
    
    def __init__(self, omega: np.ndarray, R_s_data: Optional[np.ndarray] = None):
        """
        Initialize fitter.
        
        Parameters:
            omega: Frequency array
            R_s_data: Optional complex data to fit
        """
        self.omega = omega
        self.R_s_data = R_s_data
        
    def fit_to_data(self, model: ComplexSurfaceReflectivity,
                   weight_magnitude: float = 1.0, 
                   weight_phase: float = 0.5) -> Dict:
        """
        Fit model parameters to data.
        
        Returns:
            Dictionary with fitted parameters and loss value.
        """
        if self.R_s_data is None:
            raise ValueError("No data provided for fitting")
        
        # Extract magnitude and phase data
        data_mag = np.abs(self.R_s_data)
        data_phase = np.angle(self.R_s_data)
        
        # Initial parameters
        init_params = self._flatten_params(model)
        
        # Define loss function
        def loss(params):
            model_fit = self._unflatten_params(model, params)
            R_pred = model_fit(self.omega)
            pred_mag = np.abs(R_pred)
            pred_phase = np.angle(R_pred)
            
            # Weighted MSE
            mag_loss = np.mean((pred_mag - data_mag)**2)
            phase_loss = np.mean((np.unwrap(pred_phase) - np.unwrap(data_phase))**2)
            
            return weight_magnitude * mag_loss + weight_phase * phase_loss
        
        # Optimize
        result = minimize(loss, init_params, method='L-BFGS-B',
                         bounds=self._get_param_bounds(model))
        
        # Update model with optimized parameters
        fitted_model = self._unflatten_params(model, result.x)
        
        return {
            'model': fitted_model,
            'params': fitted_model.get_params(),
            'loss': result.fun,
            'success': result.success
        }
    
    def _flatten_params(self, model: ComplexSurfaceReflectivity) -> np.ndarray:
        """Flatten model parameters into array for optimization."""
        params = []
        params.extend(model.magnitude_model.params.values())
        params.extend(model.phase_model.params.values())
        return np.array(params)
    
    def _unflatten_params(self, model: ComplexSurfaceReflectivity, 
                         flat_params: np.ndarray) -> ComplexSurfaceReflectivity:
        """Reconstruct model from flattened parameters."""
        # Create new model instance
        mag_model = type(model.magnitude_model)(**model.magnitude_model.params)
        phase_model = type(model.phase_model)(**model.phase_model.params)
        
        new_model = ComplexSurfaceReflectivity(mag_model, phase_model)
        
        # Update parameters
        n_mag = len(model.magnitude_model.params)
        mag_params = dict(zip(model.magnitude_model.params.keys(), flat_params[:n_mag]))
        phase_params = dict(zip(model.phase_model.params.keys(), flat_params[n_mag:]))
        
        new_model.magnitude_model.params.update(mag_params)
        new_model.phase_model.params.update(phase_params)
        
        return new_model
    
    def _get_param_bounds(self, model: ComplexSurfaceReflectivity) -> list:
        """Get bounds for optimization parameters."""
        bounds = []
        
        # Magnitude model bounds
        for name in model.magnitude_model.params.keys():
            if 'R' in name or 'R' in name.lower():  # Reflectivity parameters
                bounds.append((0.0, 1.0))
            elif 'omega' in name or 'freq' in name:  # Frequency parameters
                bounds.append((0.1, 2000.0))
            elif 'Gamma' in name or 'sigma' in name:  # Width parameters
                bounds.append((0.01, 500.0))
            elif 'alpha' in name or 'beta' in name or 'gamma' in name:  # Exponents
                bounds.append((0.1, 3.0))
            else:
                bounds.append((-10.0, 10.0))
        
        # Phase model bounds
        for name in model.phase_model.params.keys():
            if 'tau' in name:  # Time delay parameters
                bounds.append((-0.1, 0.1))
            elif 'phi' in name or 'phase' in name:  # Phase parameters
                bounds.append((-2*np.pi, 2*np.pi))
            elif 'omega' in name or 'freq' in name:  # Frequency parameters
                bounds.append((0.1, 2000.0))
            elif 'Gamma' in name or 'sigma' in name:  # Width parameters
                bounds.append((0.01, 500.0))
            elif 'A' in name:  # Amplitude parameters
                bounds.append((-np.pi, np.pi))
            else:
                bounds.append((-10.0, 10.0))
        
        return bounds


# =============================================================================
# VISUALIZATION AND ANALYSIS
# =============================================================================

def plot_complex_reflectivity_model(model: ComplexSurfaceReflectivity,
                                   omega_range: np.ndarray,
                                   title: str = "Complex Surface Reflectivity Model",
                                   save_path: Optional[Path] = None):
    """
    Create comprehensive visualization of R_s(ω) model.
    """
    # Compute reflectivity
    R_s = model(omega_range)
    magnitude = np.abs(R_s)
    phase = np.angle(R_s)
    
    # Compute derived quantities
    group_delay = -np.gradient(np.unwrap(phase), omega_range)
    reflectivity_squared = magnitude**2
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: Magnitude
    ax1 = axes[0, 0]
    ax1.plot(omega_range/(2*np.pi), magnitude, 'b-', lw=2)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel(r'$|R_s(\omega)|$')
    ax1.set_title('Reflectivity Magnitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Panel 2: Phase
    ax2 = axes[0, 1]
    ax2.plot(omega_range/(2*np.pi), np.unwrap(phase), 'r-', lw=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel(r'$\arg[R_s(\omega)]$ (rad)')
    ax2.set_title('Reflectivity Phase')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Complex plane
    ax3 = axes[0, 2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(omega_range)))
    for i in range(len(omega_range)-1):
        ax3.plot([R_s[i].real, R_s[i+1].real],
                [R_s[i].imag, R_s[i+1].imag],
                color=colors[i], lw=2, alpha=0.7)
    ax3.plot(R_s[0].real, R_s[0].imag, 'go', markersize=10, label='Low ω')
    ax3.plot(R_s[-1].real, R_s[-1].imag, 'ro', markersize=10, label='High ω')
    
    # Unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    ax3.set_xlabel('Re[$R_s$]')
    ax3.set_ylabel('Im[$R_s$]')
    ax3.set_title('Complex Plane')
    ax3.legend()
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Group delay
    ax4 = axes[1, 0]
    ax4.plot(omega_range/(2*np.pi), group_delay * 1e3, 'g-', lw=2)  # in ms
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Group Delay (ms)')
    ax4.set_title('Frequency-Dependent Time Delay')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Reflectivity squared
    ax5 = axes[1, 1]
    ax5.plot(omega_range/(2*np.pi), reflectivity_squared, 'm-', lw=2)
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel(r'$|R_s|^2$')
    ax5.set_title('Reflectivity Power')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.1)
    
    # Panel 6: Model parameters
    ax6 = axes[1, 2]
    ax6.axis('off')
    params_text = f"Magnitude Model: {model.magnitude_model.name}\n"
    for key, val in model.magnitude_model.params.items():
        params_text += f"  {key}: {val:.4g}\n"
    
    params_text += f"\nPhase Model: {model.phase_model.name}\n"
    for key, val in model.phase_model.params.items():
        params_text += f"  {key}: {val:.4g}\n"
    
    ax6.text(0.1, 0.95, params_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return fig


def create_model_preset(preset_name: str) -> ComplexSurfaceReflectivity:
    """
    Create preset models for common scenarios.
    
    Presets:
        - 'viscoelastic': Lorentzian magnitude + quadratic phase
        - 'quantum': Exponential tunneling + linear phase
        - 'fractal': Power-law magnitude + logarithmic phase
        - 'resonant': Lorentzian magnitude + resonant phase
        - 'simple': Constant magnitude + constant phase
    """
    if preset_name.lower() == 'viscoelastic':
        mag = LorentzianMagnitude(R0=0.85, omega0=150.0, Gamma=0.15)
        phase = QuadraticDispersion(tau0=2e-3, tau1=-1e-6, phi0=0.3*np.pi)
    
    elif preset_name.lower() == 'quantum':
        mag = ExponentialTunneling(E_tunnel=80.0, gamma=1.2, R_bg=0.3)
        phase = LinearDispersion(tau=1e-3, phi0=0.25*np.pi)
    
    elif preset_name.lower() == 'fractal':
        mag = PowerLawMagnitude(R0=0.95, omega_c=200.0, beta=1.5)
        phase = LogarithmicPhase(A=0.15*np.pi, omega_ref=20.0, phi0=0.1*np.pi)
    
    elif preset_name.lower() == 'resonant':
        mag = LorentzianMagnitude(R0=0.9, omega0=120.0, Gamma=0.1)
        phase = ResonantPhase(omega0=120.0, Gamma=30.0, phi0=0.0)
    
    elif preset_name.lower() == 'simple':
        mag = LorentzianMagnitude(R0=0.7, omega0=1000.0, Gamma=0.01)
        phase = LinearDispersion(tau=0.0, phi0=0.3*np.pi)
    
    else:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    return ComplexSurfaceReflectivity(mag, phase)


# =============================================================================
# INTEGRATION WITH ECHO CODE
# =============================================================================

def integrate_with_echo_model(model: ComplexSurfaceReflectivity,
                             echo_frequencies: np.ndarray,
                             transfer_function: Callable) -> np.ndarray:
    """
    Integrate complex R_s(ω) model into echo calculation.
    
    Parameters:
        model: Complex surface reflectivity model
        echo_frequencies: Frequencies for echo calculation
        transfer_function: Function that computes K(ω) given R_s(ω)
    
    Returns:
        echo_spectrum: Complex transfer function including R_s(ω)
    """
    # Compute frequency-dependent R_s
    R_s_complex = model(echo_frequencies)
    
    # Apply to transfer function
    echo_spectrum = transfer_function(R_s_complex)
    
    return echo_spectrum


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def demonstrate_models():
    """Demonstrate different parametric models."""
    # Frequency range
    f = np.linspace(1, 1000, 500)  # Hz
    omega = 2 * np.pi * f
    
    # Create output directory
    outdir = Path("surface_reflectivity_models")
    outdir.mkdir(exist_ok=True)
    
    # Test different presets
    presets = ['viscoelastic', 'quantum', 'fractal', 'resonant', 'simple']
    
    for preset in presets:
        print(f"\nGenerating {preset} model...")
        
        # Create model
        model = create_model_preset(preset)
        
        # Plot
        save_path = outdir / f"model_{preset}.png"
        plot_complex_reflectivity_model(
            model, omega,
            title=f"Surface Reflectivity Model: {preset.capitalize()}",
            save_path=save_path
        )
        
        print(f"  Saved: {save_path}")
        
        # Print parameters
        params = model.get_params()
        print(f"  Magnitude model: {params['magnitude_model']}")
        print(f"  Phase model: {params['phase_model']}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, preset in enumerate(presets[:4]):  # First 4 presets
        model = create_model_preset(preset)
        R_s = model(omega)
        
        row = idx // 2
        col = idx % 2
        
        ax = axes[row, col]
        ax.plot(f, np.abs(R_s), 'b-', lw=2, label='|R_s|')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'{preset.capitalize()} Model')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Add phase on secondary axis
        ax2 = ax.twinx()
        ax2.plot(f, np.unwrap(np.angle(R_s)), 'r--', lw=1.5, label='Phase')
        ax2.set_ylabel('Phase (rad)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.suptitle('Surface Reflectivity Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outdir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved: {outdir}/model_comparison.png")
    
    return outdir


if __name__ == "__main__":
    print("="*70)
    print("PARAMETRIC MODELS FOR COMPLEX SURFACE REFLECTIVITY R_s(ω)")
    print("="*70)
    
    # Demonstrate models
    outdir = demonstrate_models()
    
    print("\n" + "="*70)
    print("MODEL USAGE EXAMPLE:")
    print("="*70)
    
    # Example: Create and use a custom model
    print("\nCreating custom model for echo simulation:")
    
    # Custom magnitude and phase models
    custom_mag = PowerLawMagnitude(R0=0.8, omega_c=150.0, beta=1.2)
    custom_phase = ResonantPhase(omega0=100.0, Gamma=25.0, phi0=0.2*np.pi)
    
    # Combine into complex reflectivity
    custom_model = ComplexSurfaceReflectivity(custom_mag, custom_phase, enforce_kk=True)
    
    # Generate frequencies for GW echo
    f_echo = np.logspace(np.log10(20), np.log10(500), 200)
    omega_echo = 2 * np.pi * f_echo
    
    # Compute R_s at echo frequencies
    R_s_echo = custom_model(omega_echo)
    
    print(f"  Frequency range: {f_echo[0]:.1f} - {f_echo[-1]:.1f} Hz")
    print(f"  |R_s| at 100 Hz: {np.abs(R_s_echo[np.argmin(np.abs(f_echo-100))]):.3f}")
    print(f"  arg(R_s) at 100 Hz: {np.angle(R_s_echo[np.argmin(np.abs(f_echo-100))]):.3f} rad")
    
    # Save custom model parameters
    import json
    params = custom_model.get_params()
    with open(outdir / "custom_model_params.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\nCustom model parameters saved to: {outdir}/custom_model_params.json")
    print("\nDone!")
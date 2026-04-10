"""
Quantum Gravity Echo Detection Framework
=========================================

A unified framework for detecting parity violation signatures in 
gravitational wave data, implementing multiple quantum gravity models.

Based on: Stelle gravity ghost sector analysis
Key predictions tested:
  - Original (tanh): ε → const at high ω
  - Ghost Condensate: ε ∝ ω² 
  - Breit-Wigner: Resonance at ω = m_g
  - Non-Commutative: Moyal-deformed propagator

Author: Tanner / TNSR-Q / Quant Quip Labs
"""

import numpy as np
from scipy.signal import find_peaks, welch, butter, filtfilt
from scipy.optimize import curve_fit, minimize
from scipy.stats import chi2
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import warnings


# ============ DATA STRUCTURES ============

@dataclass
class GWEvent:
    """Gravitational wave event data"""
    event_id: str
    h_plus: np.ndarray
    h_cross: np.ndarray
    fs: float  # Sampling frequency
    M_total: float  # Total mass in solar masses
    distance_Mpc: float
    SNR_optimal: float
    
    @property
    def duration(self) -> float:
        return len(self.h_plus) / self.fs


@dataclass  
class ParityResult:
    """Parity extraction result"""
    freqs: np.ndarray
    epsilon: np.ndarray
    epsilon_mean: float
    epsilon_std: float
    power_L: np.ndarray
    power_R: np.ndarray
    SNR_parity: float
    model_fits: Dict[str, Dict]


@dataclass
class DetectionResult:
    """Detection analysis result"""
    model: str
    SNR: float
    events_used: int
    p_value: float
    epsilon_measured: float
    epsilon_predicted: float
    chi2: float
    is_detected: bool
    parameters: Dict


# ============ QUANTUM GRAVITY MODELS ============

class QuantumGravityModel:
    """Base class for QG models"""
    
    PLANCK_MASS_SOLAR = 2.18e-8  # Planck mass in solar masses
    
    def __init__(self, f_2: float = 2.15, M_bh: float = 30.0):
        self.f_2 = f_2
        self.M_bh = M_bh
        self.M_ratio = M_bh / self.PLANCK_MASS_SOLAR
        self.m_g_eff = f_2 * (self.M_ratio ** (1/3))
        self.theta = np.arctan(self.m_g_eff)
    
    def epsilon(self, omega: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def N_eff(self) -> float:
        return self.theta / (np.pi / 5.6)
    
    def SNR(self, freq: float, h_echo: float = 5.48e-21) -> float:
        omega = 2 * np.pi * freq
        eps = np.abs(self.epsilon(omega))
        return h_echo * eps * np.sqrt(self.N_eff()) / 1e-20
    
    def events_for_detection(self, freq: float = 100, threshold: float = 5.0) -> int:
        snr = self.SNR(freq)
        if snr <= 0:
            return int(1e9)
        return max(1, int(np.ceil((threshold / snr) ** 2)))


class OriginalModel(QuantumGravityModel):
    """Original tanh saturation model"""
    name = "original"
    
    def epsilon(self, omega: np.ndarray) -> np.ndarray:
        x = omega / self.m_g_eff
        return -np.sin(self.theta) * np.tanh(x)


class GhostCondensateModel(QuantumGravityModel):
    """Ghost condensate with ε ∝ ω² growth"""
    name = "ghost_condensate"
    
    def __init__(self, f_2: float = 0.37, M_bh: float = 1.2, M_star: float = 0.001):
        super().__init__(f_2, M_bh)
        self.M_star = M_star
    
    def epsilon(self, omega: np.ndarray) -> np.ndarray:
        x = omega / self.m_g_eff
        return np.sin(self.theta) * (x**2) / (1 + (x * self.M_star)**4)
    
    def N_eff(self) -> float:
        return super().N_eff() * (1 + self.M_star**2)


class BreitWignerModel(QuantumGravityModel):
    """Breit-Wigner resonance at ω = m_g"""
    name = "breit_wigner"
    
    def __init__(self, f_2: float = 3.73, M_bh: float = 1.2, gamma: float = 0.001):
        super().__init__(f_2, M_bh)
        self.gamma = gamma
    
    def epsilon(self, omega: np.ndarray) -> np.ndarray:
        x = omega / self.m_g_eff
        resonance = 1.0 / ((x - 1)**2 + self.gamma**2)
        phase = np.sin(self.theta) * x / (1 + x**2)
        return phase * resonance * self.gamma
    
    def N_eff(self) -> float:
        return super().N_eff() * (1 + 1/self.gamma)
    
    @property
    def resonance_freq(self) -> float:
        """Resonance frequency in Hz (dimensionless internal units)"""
        return self.m_g_eff / (2 * np.pi)


class NonCommutativeModel(QuantumGravityModel):
    """Non-commutative spacetime with Moyal enhancement"""
    name = "non_commutative"
    
    def __init__(self, f_2: float = 0.37, M_bh: float = 1.2, theta_nc: float = 1072):
        super().__init__(f_2, M_bh)
        self.theta_nc = theta_nc
    
    def epsilon(self, omega: np.ndarray) -> np.ndarray:
        x = omega / self.m_g_eff
        moyal = 1 + (x * self.theta_nc)**2 / (1 + (x * self.theta_nc)**4)
        base = -np.sin(self.theta) * x / (1 + x)
        return base * moyal
    
    def N_eff(self) -> float:
        return super().N_eff() * np.sqrt(self.theta_nc)


class HigherSpinModel(QuantumGravityModel):
    """Higher-spin tower (Vasiliev-like)"""
    name = "higher_spin"
    
    def __init__(self, f_2: float = 0.37, M_bh: float = 1.2, s_max: int = 10):
        super().__init__(f_2, M_bh)
        self.s_max = s_max
    
    def epsilon(self, omega: np.ndarray) -> np.ndarray:
        x = omega / self.m_g_eff
        total = np.zeros_like(x, dtype=float)
        for s in range(2, self.s_max + 1):
            weight = (-1)**s * (2*s + 1) / s**2
            total += weight * (x**s) / (1 + x**s)
        return np.sin(self.theta) * total
    
    def N_eff(self) -> float:
        return super().N_eff() * self.s_max


# ============ PARITY EXTRACTION ============

class ParityExtractor:
    """Extract parity violation from GW strain data"""
    
    def __init__(self, fs: float = 4096):
        self.fs = fs
    
    def decompose_circular(self, h_plus: np.ndarray, h_cross: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose into circular polarizations"""
        h_L = (h_plus + 1j * h_cross) / np.sqrt(2)
        h_R = (h_plus - 1j * h_cross) / np.sqrt(2)
        return h_L, h_R
    
    def extract_parity(self, h_plus: np.ndarray, h_cross: np.ndarray,
                       f_band: Tuple[float, float] = (50, 300),
                       nperseg: int = 256) -> ParityResult:
        """
        Extract frequency-dependent parity asymmetry.
        
        ε(f) = (P_L - P_R) / (P_L + P_R)
        
        In GR: ε = 0
        With parity violation: ε ≠ 0
        """
        h_L, h_R = self.decompose_circular(h_plus, h_cross)
        
        # Power spectral density
        freqs, P_L = welch(h_L, fs=self.fs, nperseg=nperseg)
        _, P_R = welch(h_R, fs=self.fs, nperseg=nperseg)
        
        # Take absolute value for power
        P_L = np.abs(P_L)
        P_R = np.abs(P_R)
        
        # Parity asymmetry
        epsilon = (P_L - P_R) / (P_L + P_R + 1e-30)
        
        # Band statistics
        mask = (freqs >= f_band[0]) & (freqs <= f_band[1])
        eps_mean = np.mean(epsilon[mask]) if mask.any() else 0
        eps_std = np.std(epsilon[mask]) if mask.any() else 0
        
        # SNR for parity detection
        noise_floor = np.median(np.abs(epsilon[~mask])) if (~mask).any() else 1
        snr_parity = np.abs(eps_mean) / (noise_floor + 1e-30)
        
        return ParityResult(
            freqs=freqs,
            epsilon=epsilon,
            epsilon_mean=eps_mean,
            epsilon_std=eps_std,
            power_L=P_L,
            power_R=P_R,
            SNR_parity=snr_parity,
            model_fits={}
        )
    
    def fit_models(self, parity_result: ParityResult, M_bh: float = 30.0
                   ) -> Dict[str, Dict]:
        """Fit all QG models to extracted parity"""
        
        models = {
            'original': OriginalModel(M_bh=M_bh),
            'ghost_condensate': GhostCondensateModel(M_bh=M_bh),
            'breit_wigner': BreitWignerModel(M_bh=M_bh),
            'non_commutative': NonCommutativeModel(M_bh=M_bh),
        }
        
        fits = {}
        for name, model in models.items():
            try:
                omega = 2 * np.pi * parity_result.freqs
                eps_theory = model.epsilon(omega)
                
                # Simple amplitude fit
                valid = np.abs(parity_result.epsilon) > 1e-10
                if valid.any():
                    scale = np.mean(parity_result.epsilon[valid] / eps_theory[valid])
                    residuals = parity_result.epsilon - scale * eps_theory
                    chi2_val = np.sum(residuals**2 / (parity_result.epsilon_std**2 + 1e-30))
                    
                    fits[name] = {
                        'scale': scale,
                        'chi2': chi2_val,
                        'theory': eps_theory,
                        'residuals': residuals
                    }
            except Exception as e:
                fits[name] = {'error': str(e)}
        
        parity_result.model_fits = fits
        return fits


# ============ STACKING ANALYSIS ============

class StackingAnalyzer:
    """Coherent stacking of multiple events"""
    
    def __init__(self, extractor: ParityExtractor):
        self.extractor = extractor
        self.events: List[ParityResult] = []
    
    def add_event(self, h_plus: np.ndarray, h_cross: np.ndarray,
                  weight: float = 1.0, f_band: Tuple[float, float] = (50, 300)):
        """Add event to stack"""
        result = self.extractor.extract_parity(h_plus, h_cross, f_band)
        result.weight = weight  # type: ignore
        self.events.append(result)
    
    def compute_stack(self) -> ParityResult:
        """Compute weighted stack of all events"""
        if not self.events:
            raise ValueError("No events to stack")
        
        # Use first event's frequency grid
        freqs = self.events[0].freqs
        
        # Weighted average
        total_weight = sum(getattr(e, 'weight', 1.0) for e in self.events)
        stacked_epsilon = np.zeros_like(freqs)
        stacked_var = np.zeros_like(freqs)
        
        for event in self.events:
            w = getattr(event, 'weight', 1.0)
            stacked_epsilon += w * event.epsilon
            stacked_var += w**2 * event.epsilon_std**2
        
        stacked_epsilon /= total_weight
        stacked_std = np.sqrt(stacked_var) / total_weight
        
        # SNR improves as sqrt(N)
        snr_stacked = np.abs(np.mean(stacked_epsilon)) / (np.mean(stacked_std) + 1e-30)
        
        return ParityResult(
            freqs=freqs,
            epsilon=stacked_epsilon,
            epsilon_mean=np.mean(stacked_epsilon),
            epsilon_std=np.mean(stacked_std),
            power_L=np.mean([e.power_L for e in self.events], axis=0),
            power_R=np.mean([e.power_R for e in self.events], axis=0),
            SNR_parity=snr_stacked,
            model_fits={}
        )


# ============ DETECTION PIPELINE ============

class DetectionPipeline:
    """Complete detection pipeline for QG echoes"""
    
    def __init__(self, fs: float = 4096):
        self.extractor = ParityExtractor(fs)
        self.stacker = StackingAnalyzer(self.extractor)
        self.results: List[DetectionResult] = []
    
    def analyze_event(self, event: GWEvent, 
                      f_band: Tuple[float, float] = (50, 300)) -> ParityResult:
        """Analyze single event"""
        return self.extractor.extract_parity(event.h_plus, event.h_cross, f_band)
    
    def search_resonance(self, parity: ParityResult,
                         f_search: Tuple[float, float] = (50, 500)
                         ) -> Optional[Dict]:
        """Search for Breit-Wigner resonance peak"""
        mask = (parity.freqs >= f_search[0]) & (parity.freqs <= f_search[1])
        eps_search = np.abs(parity.epsilon[mask])
        freqs_search = parity.freqs[mask]
        
        # Find peaks
        peaks, properties = find_peaks(eps_search, height=2*parity.epsilon_std)
        
        if len(peaks) > 0:
            best_peak = peaks[np.argmax(properties['peak_heights'])]
            return {
                'frequency': freqs_search[best_peak],
                'amplitude': eps_search[best_peak],
                'significance': properties['peak_heights'][np.argmax(properties['peak_heights'])] / parity.epsilon_std
            }
        return None
    
    def run_detection(self, events: List[GWEvent], model_name: str = 'all',
                      threshold: float = 5.0) -> List[DetectionResult]:
        """Run full detection analysis"""
        
        models = ['original', 'ghost_condensate', 'breit_wigner', 'non_commutative']
        if model_name != 'all':
            models = [model_name]
        
        results = []
        
        # Stack all events
        for event in events:
            parity = self.analyze_event(event)
            self.stacker.add_event(event.h_plus, event.h_cross)
        
        stacked = self.stacker.compute_stack()
        fits = self.extractor.fit_models(stacked, M_bh=events[0].M_total)
        
        for model in models:
            if model in fits and 'error' not in fits[model]:
                chi2_val = fits[model]['chi2']
                dof = len(stacked.freqs) - 1
                p_value = 1 - chi2.cdf(chi2_val, dof)
                
                results.append(DetectionResult(
                    model=model,
                    SNR=stacked.SNR_parity,
                    events_used=len(events),
                    p_value=p_value,
                    epsilon_measured=stacked.epsilon_mean,
                    epsilon_predicted=np.mean(fits[model]['theory']),
                    chi2=chi2_val,
                    is_detected=stacked.SNR_parity > threshold,
                    parameters={'scale': fits[model]['scale']}
                ))
        
        self.results = results
        return results


# ============ CONVENIENCE FUNCTIONS ============

def create_model(name: str, **kwargs) -> QuantumGravityModel:
    """Factory function to create QG models"""
    models = {
        'original': OriginalModel,
        'ghost_condensate': GhostCondensateModel,
        'breit_wigner': BreitWignerModel,
        'non_commutative': NonCommutativeModel,
        'higher_spin': HigherSpinModel,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name](**kwargs)


def quick_analysis(h_plus: np.ndarray, h_cross: np.ndarray, 
                   fs: float = 4096, M_bh: float = 30.0) -> Dict:
    """Quick analysis of single event"""
    extractor = ParityExtractor(fs)
    parity = extractor.extract_parity(h_plus, h_cross)
    fits = extractor.fit_models(parity, M_bh)
    
    return {
        'epsilon_mean': parity.epsilon_mean,
        'epsilon_std': parity.epsilon_std,
        'SNR': parity.SNR_parity,
        'best_model': min(fits.keys(), key=lambda k: fits[k].get('chi2', 1e10)),
        'fits': fits
    }


if __name__ == "__main__":
    # Example usage
    print("Quantum Gravity Echo Detection Framework")
    print("=" * 50)
    
    # Test models
    for name in ['original', 'ghost_condensate', 'breit_wigner', 'non_commutative']:
        model = create_model(name)
        print(f"{name}: SNR(100Hz) = {model.SNR(100):.4f}, Events = {model.events_for_detection(100)}")

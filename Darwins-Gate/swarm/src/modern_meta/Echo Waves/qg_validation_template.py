"""
QUANTUM GRAVITY ECHO LADDER - VALIDATION TEMPLATE
==================================================

Fill in your actual measured values where indicated with ___
Then run: python qg_validation_template.py

This will generate a submission-grade validation packet with:
  1. Full γ_n table with σ_γ
  2. Fit QA (windows, residuals, DW statistics)
  3. Drift analysis with endpoints
  4. Internal consistency checks
"""

import numpy as np
from scipy.stats import chi2 as chi2_dist
import json

# ============================================================
# FILL IN YOUR DATA HERE
# ============================================================

ECHO_DATA = {
    "echo_number": [1, 2, 3, 4, 5, 6, 7, 8],
    
    # γ_n: damping rate per echo
    # UNITS: Must be consistent with delta_t!
    #   If delta_t in seconds → γ in Hz (1/s)
    #   If delta_t dimensionless → γ dimensionless
    "gamma": [
        21.98,   # n=1: REPLACE with your value
        23.50,   # n=2
        25.10,   # n=3
        26.80,   # n=4
        28.60,   # n=5
        30.50,   # n=6
        32.50,   # n=7
        34.60,   # n=8
    ],
    
    # σ_γ: uncertainty on each γ_n
    # Source: sqrt(fit_covariance² + systematic²)
    "sigma_gamma": [
        2.5,   # n=1
        2.7,   # n=2
        2.9,   # n=3
        3.1,   # n=4
        3.3,   # n=5
        3.5,   # n=6
        3.7,   # n=7
        3.9,   # n=8
    ],
    
    # R_n = |R_∞ R_s|: reflectivity product
    # OPTIONS:
    #   1. Leave as None → will compute from γ via R = exp(-γΔt)
    #   2. Fill with measured values → will use directly
    "R_infinity_Rs": None,  # or [0.312, 0.298, ...] if measured
    "R_source": "computed_from_gamma",  # or "measured"
    
    # Echo spacing
    "delta_t_seconds": 0.053,
    
    # Units specification
    "gamma_units": "Hz",  # "Hz", "Planck", "1/seconds"
    "provenance": "YOUR DATA SOURCE HERE",
}

FIT_QA = {
    "description": "Per-echo exponential envelope fit",
    
    "pipeline": {
        "step_1": "Bandpass filter (Butterworth order 4)",
        "step_2": "Hilbert envelope extraction",
        "step_3": "Exponential fit: A * exp(-γ * t) + C",
        "step_4": "σ_γ from covariance + systematic",
    },
    
    "filter_band_Hz": (80, 300),
    "filter_order": 4,
    "window_type": "tukey",
    "window_alpha": 0.1,
    
    # Per-echo fit windows (seconds)
    "fit_window_sec": {
        1: (0.000, 0.200),
        2: (0.000, 0.200),
        3: (0.000, 0.200),
        4: (0.000, 0.200),
        5: (0.000, 0.200),
        6: (0.000, 0.200),
        7: (0.000, 0.180),
        8: (0.000, 0.160),
    },
    
    # Fit quality per echo
    "fit_metrics": {
        1: {"chi2_red": 1.12, "dof": 45, "rms_resid": 0.023},
        2: {"chi2_red": 1.08, "dof": 45, "rms_resid": 0.025},
        3: {"chi2_red": 1.15, "dof": 45, "rms_resid": 0.027},
        4: {"chi2_red": 1.21, "dof": 45, "rms_resid": 0.029},
        5: {"chi2_red": 1.18, "dof": 45, "rms_resid": 0.031},
        6: {"chi2_red": 1.25, "dof": 45, "rms_resid": 0.034},
        7: {"chi2_red": 1.32, "dof": 40, "rms_resid": 0.038},
        8: {"chi2_red": 1.45, "dof": 35, "rms_resid": 0.042},
    },
    
    # Durbin-Watson per echo (optional)
    "durbin_watson": {
        1: 1.89, 2: 1.92, 3: 1.85, 4: 1.78,
        5: 1.81, 6: 1.74, 7: 1.68, 8: 1.61,
    },
    
    # Residual sample (optional)
    "residual_sample": {
        1: [0.012, -0.008, 0.015, -0.021, 0.003, -0.011, 0.018, -0.005],
    },
}

DRIFT = {
    "description": "Background drift estimate",
    "method": "endpoint_difference",  # or "fit_slope", "bootstrap"
    "drifting_parameter": "gamma",
    
    "endpoints": {
        "start": {
            "label": "first 2 echoes",
            "echo_range": [1, 2],
            "value": None,  # Will compute if None
            "sigma": None,
            "source": "weighted_mean",
        },
        "end": {
            "label": "last 2 echoes",
            "echo_range": [7, 8],
            "value": None,
            "sigma": None,
            "source": "weighted_mean",
        },
    },
    
    "coupling_scale": {
        "description": "Expected QG mode coupling Δγ",
        "value": 1.5,  # Hz (or your units)
        "sigma": 0.3,
    },
}

# ============================================================
# VALIDATION FUNCTIONS (do not edit)
# ============================================================

def compute_R_from_gamma(gamma, delta_t):
    return np.exp(-np.array(gamma) * delta_t)

def compute_sigma_R(R, sigma_gamma, delta_t):
    return np.abs(delta_t * np.array(R) * np.array(sigma_gamma))

def compute_gamma_statistics(data):
    gamma = np.array(data["gamma"])
    sigma_gamma = np.array(data["sigma_gamma"])
    n = np.array(data["echo_number"])
    
    weights = 1.0 / sigma_gamma**2
    gamma_bar = np.sum(weights * gamma) / np.sum(weights)
    sigma_gamma_bar = 1.0 / np.sqrt(np.sum(weights))
    
    chi2_const = np.sum(((gamma - gamma_bar) / sigma_gamma)**2)
    dof = len(gamma) - 1
    p_const = chi2_dist.sf(chi2_const, dof)
    
    # Linear fit
    S, Sx, Sy = np.sum(weights), np.sum(weights * n), np.sum(weights * gamma)
    Sxx, Sxy = np.sum(weights * n**2), np.sum(weights * n * gamma)
    delta = S * Sxx - Sx**2
    
    intercept = (Sxx * Sy - Sx * Sxy) / delta
    slope = (S * Sxy - Sx * Sy) / delta
    sigma_slope = np.sqrt(S / delta)
    
    gamma_linear = intercept + slope * n
    chi2_linear = np.sum(((gamma - gamma_linear) / sigma_gamma)**2)
    
    delta_gamma = np.diff(gamma)
    sigma_delta = np.sqrt(sigma_gamma[:-1]**2 + sigma_gamma[1:]**2)
    
    return {
        "gamma_bar": gamma_bar, "sigma_gamma_bar": sigma_gamma_bar,
        "chi2_const": chi2_const, "dof_const": dof, "p_const": p_const,
        "slope": slope, "sigma_slope": sigma_slope, "intercept": intercept,
        "chi2_linear": chi2_linear, "dof_linear": len(gamma) - 2,
        "delta_gamma": delta_gamma, "sigma_delta_gamma": sigma_delta,
    }

def compute_drift_endpoints(data, drift):
    gamma = np.array(data["gamma"])
    sigma = np.array(data["sigma_gamma"])
    
    for key in ["start", "end"]:
        ep = drift["endpoints"][key]
        idx = [i-1 for i in ep["echo_range"]]
        
        if ep["value"] is None:
            w = 1.0 / sigma[idx]**2
            ep["value"] = np.sum(w * gamma[idx]) / np.sum(w)
            ep["sigma"] = 1.0 / np.sqrt(np.sum(w))
    
    start, end = drift["endpoints"]["start"], drift["endpoints"]["end"]
    drift["delta_gamma"] = end["value"] - start["value"]
    drift["sigma_delta"] = np.sqrt(start["sigma"]**2 + end["sigma"]**2)
    drift["significance"] = abs(drift["delta_gamma"]) / drift["sigma_delta"]
    
    coupling = drift["coupling_scale"]["value"]
    drift["drift_to_coupling_ratio"] = drift["delta_gamma"] / coupling if coupling > 0 else np.inf
    
    return drift

def generate_validation_packet():
    print("=" * 70)
    print("QG ECHO LADDER - FULL VALIDATION PACKET")
    print("=" * 70)
    
    # Prepare R values
    gamma = np.array(ECHO_DATA["gamma"])
    sigma_gamma = np.array(ECHO_DATA["sigma_gamma"])
    delta_t = ECHO_DATA["delta_t_seconds"]
    
    if ECHO_DATA["R_infinity_Rs"] is None:
        R = compute_R_from_gamma(gamma, delta_t)
        ECHO_DATA["R_source"] = "computed_from_gamma"
    else:
        R = np.array(ECHO_DATA["R_infinity_Rs"])
    
    sigma_R = compute_sigma_R(R, sigma_gamma, delta_t)
    
    # === DELIVERABLE 1: γ TABLE ===
    print("\n" + "=" * 70)
    print("DELIVERABLE 1: GAMMA LADDER")
    print("=" * 70)
    print(f"Source: {ECHO_DATA['provenance']}")
    print(f"Units: γ in {ECHO_DATA['gamma_units']}, Δt = {delta_t} s")
    print(f"R source: {ECHO_DATA['R_source']}")
    
    print(f"\n{'n':<4} {'γ_n':<12} {'σ_γ':<12} {'R_n':<12} {'σ_R':<12}")
    print("-" * 52)
    for i, n in enumerate(ECHO_DATA["echo_number"]):
        print(f"{n:<4} {gamma[i]:<12.4f} {sigma_gamma[i]:<12.4f} {R[i]:<12.6f} {sigma_R[i]:<12.6f}")
    
    stats = compute_gamma_statistics(ECHO_DATA)
    
    print(f"\nWeighted mean: γ̄ = {stats['gamma_bar']:.4f} ± {stats['sigma_gamma_bar']:.4f}")
    print(f"\nConstancy test:")
    print(f"  χ² = {stats['chi2_const']:.2f} / {stats['dof_const']} dof, p = {stats['p_const']:.4g}")
    if stats['p_const'] < 0.01:
        print("  → γ NOT CONSTANT (p < 0.01) - LADDER SUPPORTED ✓")
    else:
        print("  → γ consistent with constant (p ≥ 0.01)")
    
    print(f"\nLinear fit: γ = {stats['intercept']:.4f} + {stats['slope']:.4f} × n")
    print(f"  Slope = {stats['slope']:.4f} ± {stats['sigma_slope']:.4f}")
    print(f"  Significance: {abs(stats['slope'])/stats['sigma_slope']:.2f}σ")
    
    print(f"\nLadder spacing Δγ:")
    for i, (dg, sdg) in enumerate(zip(stats['delta_gamma'], stats['sigma_delta_gamma'])):
        print(f"  Δγ_{i+1}→{i+2} = {dg:.4f} ± {sdg:.4f}")
    
    # === DELIVERABLE 2: FIT QA ===
    print("\n" + "=" * 70)
    print("DELIVERABLE 2: FIT QA")
    print("=" * 70)
    print(f"Filter: {FIT_QA['filter_band_Hz']} Hz, order {FIT_QA['filter_order']}")
    print(f"Window: {FIT_QA['window_type']}, α = {FIT_QA['window_alpha']}")
    
    print(f"\n{'Echo':<6} {'Window':<16} {'χ²_red':<10} {'dof':<6} {'RMS':<10} {'DW':<8}")
    print("-" * 66)
    for n in ECHO_DATA["echo_number"]:
        w = FIT_QA["fit_window_sec"].get(n, (0,0))
        m = FIT_QA["fit_metrics"].get(n, {})
        dw = FIT_QA["durbin_watson"].get(n, None)
        print(f"{n:<6} ({w[0]:.3f},{w[1]:.3f}){'':<4} {m.get('chi2_red',''):<10} {m.get('dof',''):<6} {m.get('rms_resid',''):<10} {dw if dw else '':<8}")
    
    dw_vals = [v for v in FIT_QA["durbin_watson"].values() if v]
    if dw_vals:
        mean_dw = np.mean(dw_vals)
        print(f"\nMean Durbin-Watson: {mean_dw:.2f}")
        print("  → No autocorrelation" if 1.5 < mean_dw < 2.5 else "  → POSSIBLE AUTOCORRELATION")
    
    # === DELIVERABLE 3: DRIFT ===
    print("\n" + "=" * 70)
    print("DELIVERABLE 3: DRIFT ANALYSIS")
    print("=" * 70)
    
    drift = compute_drift_endpoints(ECHO_DATA, DRIFT)
    
    print(f"Method: {drift['method']}")
    for key, ep in drift["endpoints"].items():
        print(f"  {key}: {ep['label']} = {ep['value']:.4f} ± {ep['sigma']:.4f} ({ep['source']})")
    
    print(f"\nDrift: Δγ = {drift['delta_gamma']:.4f} ± {drift['sigma_delta']:.4f}")
    print(f"Significance: {drift['significance']:.2f}σ")
    print(f"\nComparison to QG coupling ({drift['coupling_scale']['value']:.4f} ± {drift['coupling_scale']['sigma']:.4f}):")
    print(f"  Drift/Coupling = {drift['drift_to_coupling_ratio']:.2f}")
    
    if drift['drift_to_coupling_ratio'] < 1:
        print("  → Drift < coupling scale ✓")
    elif drift['drift_to_coupling_ratio'] < 3:
        print("  → Drift ~ coupling scale (marginal)")
    else:
        print("  → Drift >> coupling scale (SYSTEMATIC CONCERN)")
    
    # === DELIVERABLE 4: CONSISTENCY ===
    print("\n" + "=" * 70)
    print("DELIVERABLE 4: CONSISTENCY")
    print("=" * 70)
    
    R_check = compute_R_from_gamma(gamma, delta_t)
    print(f"R consistency (measured vs exp(-γΔt)):")
    print(f"  Max |ΔR/R|: {100*np.max(np.abs(R - R_check)/R):.2f}%")
    print(f"  γ monotonic increasing: {all(np.diff(gamma) > 0)}")
    print(f"  R monotonic decreasing: {all(np.diff(R) < 0)}")
    
    print("\n" + "=" * 70)
    print("VALIDATION PACKET COMPLETE")
    print("=" * 70)
    
    return {"stats": stats, "drift": drift}

if __name__ == "__main__":
    results = generate_validation_packet()

#!/usr/bin/env python3
"""
Frequency Band Intersection Analyzer
=====================================

Analyzes how frequency bands evolve across time and identifies intersection points
where nulls in different bands correlate. Inspired by CCC field evolution methodology.

Key Features:
1. Band trajectory tracking: Follow each frequency band's null evolution through time
2. Cross-band intersections: Find where nulls in different bands align
3. Correlation functions: Measure coupling between frequency bands
4. Field flow visualization: Show energy/information flow between bands
5. Resonance nodes: Identify points where multiple bands intersect

This goes beyond time slices to reveal the causal structure of the echo network.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.signal import correlate, correlation_lags
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import json

# Import from existing code - copy necessary functions
import sys
sys.path.append('/home/ubuntu')

# Copy functions from echo_web_3d.py
def build_echo_waveform(M_solar, epsilon, duration, fs):
    """Build synthetic echo waveform."""
    from scipy.signal import chirp
    
    M_kg = M_solar * 1.989e30
    c = 299792458.0
    G = 6.67430e-11
    R_s = 2 * G * M_kg / c**2
    dt_echo = (4 * R_s / c) * np.log(1 / epsilon)
    
    t = np.arange(0, duration, 1/fs)
    
    # Primary chirp
    f0, f1 = 100, 400
    h_primary = chirp(t, f0, duration, f1, method='linear')
    h_primary *= np.exp(-t / (0.3 * duration))
    
    # Echo with frequency-dependent amplitude
    echo_amp_base = 0.8
    h_echo = np.zeros_like(h_primary)
    
    n_echoes = int(duration / dt_echo)
    for i in range(1, n_echoes + 1):
        delay_samples = int(i * dt_echo * fs)
        if delay_samples >= len(h_primary):
            break
        
        amp = echo_amp_base * (0.92 ** i)
        h_echo[delay_samples:] += amp * h_primary[:len(h_primary)-delay_samples]
    
    h = h_primary + h_echo
    
    # Normalize
    h = h / np.max(np.abs(h))
    
    echo_amp = echo_amp_base
    params = {'M_solar': M_solar, 'epsilon': epsilon, 'R_s': R_s}
    
    return h, dt_echo, echo_amp, params

def stft_power_db(signal, fs, nperseg=512, hop=32):
    """Compute STFT power spectrogram in dB."""
    from scipy.signal import stft
    
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=nperseg-hop, window='hann')
    Pxx = np.abs(Zxx) ** 2
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    
    TT, FF = np.meshgrid(t, f)
    return TT, FF, Pxx_db

def detect_nulls_per_band(TT, FF, Pdb, freq_bands, percentile=20.0, prominence_factor=0.18):
    """Detect nulls in each frequency band."""
    from scipy.signal import find_peaks
    
    bands = []
    
    for f_low, f_high in freq_bands:
        f_center = (f_low + f_high) / 2
        
        # Extract band
        freq_mask = (FF[:, 0] >= f_low) & (FF[:, 0] < f_high)
        if not np.any(freq_mask):
            bands.append({
                'freq_range': (f_low, f_high),
                'freq_center': f_center,
                'n_nulls': 0,
                'null_times': [],
                'null_powers': []
            })
            continue
        
        band_power = Pdb[freq_mask, :]
        time_axis = TT[0, :]
        
        # Average over frequency
        avg_power = np.mean(band_power, axis=0)
        
        # Find nulls (local minima)
        threshold = np.percentile(avg_power, percentile)
        prominence = prominence_factor * (np.max(avg_power) - np.min(avg_power))
        
        peaks, properties = find_peaks(-avg_power, height=-threshold, prominence=prominence)
        
        null_times = time_axis[peaks]
        null_powers = avg_power[peaks]
        
        bands.append({
            'freq_range': (f_low, f_high),
            'freq_center': f_center,
            'n_nulls': len(null_times),
            'null_times': null_times.tolist(),
            'null_powers': null_powers.tolist()
        })
    
    return {'bands': bands}


def track_band_trajectories(band_data: Dict, duration: float) -> Dict:
    """
    Track how each frequency band's nulls evolve through time.
    
    Returns:
        Dict with band trajectories, velocity, acceleration, etc.
    """
    bands = band_data['bands']
    trajectories = []
    
    for band in bands:
        if band['n_nulls'] == 0:
            continue
            
        null_times = np.array(band['null_times'])
        null_powers = np.array(band['null_powers'])
        
        # Sort by time
        idx = np.argsort(null_times)
        null_times = null_times[idx]
        null_powers = null_powers[idx]
        
        # Compute trajectory properties
        if len(null_times) > 1:
            # Velocity: rate of null occurrence
            dt = np.diff(null_times)
            velocity = 1.0 / np.mean(dt) if len(dt) > 0 else 0.0
            
            # Acceleration: change in velocity
            if len(dt) > 1:
                ddt = np.diff(dt)
                acceleration = np.mean(ddt) if len(ddt) > 0 else 0.0
            else:
                acceleration = 0.0
                
            # Power evolution: how null depth changes
            power_slope = (null_powers[-1] - null_powers[0]) / (null_times[-1] - null_times[0])
        else:
            velocity = 0.0
            acceleration = 0.0
            power_slope = 0.0
        
        trajectories.append({
            'freq_center': band['freq_center'],
            'freq_range': band['freq_range'],
            'null_times': null_times,
            'null_powers': null_powers,
            'n_nulls': len(null_times),
            'velocity': velocity,  # Nulls per second
            'acceleration': acceleration,
            'power_slope': power_slope,  # dB per second
            'mean_power': np.mean(null_powers),
            'std_power': np.std(null_powers)
        })
    
    return {'trajectories': trajectories, 'duration': duration}


def find_band_intersections(traj_data: Dict, time_tolerance: float = 0.02) -> List[Dict]:
    """
    Find intersection points where nulls in different frequency bands align in time.
    
    Args:
        time_tolerance: Time window (seconds) for considering nulls as intersecting
    
    Returns:
        List of intersection events with participating bands and properties
    """
    trajectories = traj_data['trajectories']
    intersections = []
    
    # For each pair of bands
    for i, traj_i in enumerate(trajectories):
        for j, traj_j in enumerate(trajectories[i+1:], start=i+1):
            # Find time-aligned nulls
            times_i = traj_i['null_times']
            times_j = traj_j['null_times']
            powers_i = traj_i['null_powers']
            powers_j = traj_j['null_powers']
            
            # Check each null in band i against all nulls in band j
            for ti_idx, ti in enumerate(times_i):
                for tj_idx, tj in enumerate(times_j):
                    if abs(ti - tj) < time_tolerance:
                        # Found an intersection!
                        intersections.append({
                            'time': (ti + tj) / 2,
                            'band_i': i,
                            'band_j': j,
                            'freq_i': traj_i['freq_center'],
                            'freq_j': traj_j['freq_center'],
                            'power_i': powers_i[ti_idx],
                            'power_j': powers_j[tj_idx],
                            'power_product': powers_i[ti_idx] * powers_j[tj_idx],
                            'freq_separation': abs(traj_i['freq_center'] - traj_j['freq_center']),
                            'time_diff': abs(ti - tj)
                        })
    
    print(f"   Found {len(intersections)} band intersections")
    return intersections


def compute_cross_band_correlations(traj_data: Dict, dt: float = 0.01) -> Dict:
    """
    Compute cross-correlation functions between frequency bands.
    
    This measures how nulls in one band predict nulls in another band at different time lags.
    """
    trajectories = traj_data['trajectories']
    duration = traj_data['duration']
    
    # Create time grid
    time_grid = np.arange(0, duration, dt)
    n_bands = len(trajectories)
    
    # Convert null times to binary signals (1 = null present, 0 = no null)
    band_signals = []
    for traj in trajectories:
        signal = np.zeros(len(time_grid))
        for null_time in traj['null_times']:
            idx = np.argmin(np.abs(time_grid - null_time))
            signal[idx] = 1.0
        # Smooth slightly to avoid delta functions
        signal = gaussian_filter1d(signal, sigma=2.0)
        band_signals.append(signal)
    
    # Compute cross-correlations
    correlations = []
    for i in range(n_bands):
        for j in range(i+1, n_bands):
            # Cross-correlate band i with band j
            corr = correlate(band_signals[i], band_signals[j], mode='same')
            lags = correlation_lags(len(band_signals[i]), len(band_signals[j]), mode='same')
            lag_times = lags * dt
            
            # Find peak correlation
            peak_idx = np.argmax(np.abs(corr))
            peak_lag = lag_times[peak_idx]
            peak_value = corr[peak_idx]
            
            correlations.append({
                'band_i': i,
                'band_j': j,
                'freq_i': trajectories[i]['freq_center'],
                'freq_j': trajectories[j]['freq_center'],
                'correlation': corr,
                'lags': lag_times,
                'peak_lag': peak_lag,
                'peak_value': peak_value,
                'max_abs_corr': np.max(np.abs(corr))
            })
    
    return {'correlations': correlations, 'time_grid': time_grid, 'dt': dt}


def identify_resonance_nodes(intersections: List[Dict], 
                             time_window: float = 0.05,
                             min_bands: int = 3) -> List[Dict]:
    """
    Identify resonance nodes: points in time where multiple frequency bands intersect.
    
    These are special points where the echo cavity couples many modes simultaneously.
    """
    if len(intersections) == 0:
        return []
    
    # Cluster intersections by time
    times = np.array([inter['time'] for inter in intersections])
    
    nodes = []
    processed = set()
    
    for i, inter in enumerate(intersections):
        if i in processed:
            continue
        
        # Find all intersections within time window
        t_center = inter['time']
        nearby_mask = np.abs(times - t_center) < time_window
        nearby_indices = np.where(nearby_mask)[0]
        
        # Get unique bands involved
        bands_involved = set()
        freqs_involved = []
        powers = []
        
        for idx in nearby_indices:
            processed.add(idx)
            bands_involved.add(intersections[idx]['band_i'])
            bands_involved.add(intersections[idx]['band_j'])
            freqs_involved.append(intersections[idx]['freq_i'])
            freqs_involved.append(intersections[idx]['freq_j'])
            powers.append(intersections[idx]['power_i'])
            powers.append(intersections[idx]['power_j'])
        
        if len(bands_involved) >= min_bands:
            nodes.append({
                'time': t_center,
                'n_bands': len(bands_involved),
                'bands': list(bands_involved),
                'n_intersections': len(nearby_indices),
                'freq_range': (min(freqs_involved), max(freqs_involved)),
                'mean_power': np.mean(powers),
                'coupling_strength': len(nearby_indices) * np.mean(powers)
            })
    
    # Sort by coupling strength
    nodes.sort(key=lambda x: x['coupling_strength'], reverse=True)
    
    print(f"   Identified {len(nodes)} resonance nodes (≥{min_bands} bands)")
    return nodes


def visualize_band_intersections(traj_data: Dict, intersections: List[Dict], 
                                 resonance_nodes: List[Dict], outdir: Path):
    """
    Create comprehensive visualization of band intersections and evolution.
    """
    trajectories = traj_data['trajectories']
    duration = traj_data['duration']
    
    fig = plt.figure(figsize=(24, 18))
    
    # ========== PANEL 1: Band Trajectories with Intersections ==========
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot each band's null trajectory
    for traj in trajectories:
        freq = traj['freq_center']
        times = traj['null_times']
        powers = traj['null_powers']
        
        # Color by power
        scatter = ax1.scatter(times, [freq]*len(times), 
                            c=powers, cmap='plasma', s=50, alpha=0.7,
                            vmin=-130, vmax=-70, edgecolors='black', linewidths=0.5)
    
    # Mark intersections
    if len(intersections) > 0:
        inter_times = [inter['time'] for inter in intersections]
        inter_freqs_i = [inter['freq_i'] for inter in intersections]
        inter_freqs_j = [inter['freq_j'] for inter in intersections]
        
        # Draw lines connecting intersecting bands
        for inter in intersections[:500]:  # Limit to avoid clutter
            ax1.plot([inter['time'], inter['time']], 
                    [inter['freq_i'], inter['freq_j']],
                    'lime', alpha=0.3, lw=1.5, zorder=1)
    
    # Mark resonance nodes
    if len(resonance_nodes) > 0:
        for node in resonance_nodes[:20]:  # Top 20 nodes
            ax1.axvline(node['time'], color='red', alpha=0.5, lw=2, ls='--')
            ax1.text(node['time'], 550, f"{node['n_bands']}B", 
                    ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')
    
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Frequency (Hz)", fontsize=12)
    ax1.set_title(f"Band Trajectories & Intersections ({len(intersections)} total)", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, duration])
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Null Depth (dB)", fontsize=10)
    
    # ========== PANEL 2: Intersection Time Distribution ==========
    ax2 = plt.subplot(2, 3, 2)
    
    if len(intersections) > 0:
        inter_times = [inter['time'] for inter in intersections]
        ax2.hist(inter_times, bins=50, color='lime', alpha=0.7, edgecolor='black')
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("Intersection Count", fontsize=12)
        ax2.set_title("Intersection Time Distribution", fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Mark resonance nodes
        if len(resonance_nodes) > 0:
            for node in resonance_nodes[:10]:
                ax2.axvline(node['time'], color='red', alpha=0.7, lw=2, ls='--')
    
    # ========== PANEL 3: Frequency Separation Distribution ==========
    ax3 = plt.subplot(2, 3, 3)
    
    if len(intersections) > 0:
        freq_seps = [inter['freq_separation'] for inter in intersections]
        ax3.hist(freq_seps, bins=50, color='cyan', alpha=0.7, edgecolor='black')
        ax3.set_xlabel("Frequency Separation (Hz)", fontsize=12)
        ax3.set_ylabel("Count", fontsize=12)
        ax3.set_title("Intersection Frequency Separation", fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # ========== PANEL 4: Band Velocity Profile ==========
    ax4 = plt.subplot(2, 3, 4)
    
    freqs = [traj['freq_center'] for traj in trajectories]
    velocities = [traj['velocity'] for traj in trajectories]
    n_nulls = [traj['n_nulls'] for traj in trajectories]
    
    scatter = ax4.scatter(freqs, velocities, c=n_nulls, cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black', linewidths=1)
    ax4.set_xlabel("Frequency (Hz)", fontsize=12)
    ax4.set_ylabel("Null Velocity (nulls/s)", fontsize=12)
    ax4.set_title("Band Null Production Rate", fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("Number of Nulls", fontsize=10)
    
    # ========== PANEL 5: Power Evolution by Band ==========
    ax5 = plt.subplot(2, 3, 5)
    
    power_slopes = [traj['power_slope'] for traj in trajectories]
    mean_powers = [traj['mean_power'] for traj in trajectories]
    
    scatter = ax5.scatter(freqs, power_slopes, c=mean_powers, cmap='plasma',
                         s=100, alpha=0.7, edgecolors='black', linewidths=1,
                         vmin=-130, vmax=-70)
    ax5.axhline(0, color='red', ls='--', lw=2, alpha=0.5)
    ax5.set_xlabel("Frequency (Hz)", fontsize=12)
    ax5.set_ylabel("Power Slope (dB/s)", fontsize=12)
    ax5.set_title("Null Depth Evolution Rate", fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label("Mean Power (dB)", fontsize=10)
    
    # ========== PANEL 6: Resonance Node Analysis ==========
    ax6 = plt.subplot(2, 3, 6)
    
    if len(resonance_nodes) > 0:
        node_times = [node['time'] for node in resonance_nodes[:20]]
        node_strengths = [node['coupling_strength'] for node in resonance_nodes[:20]]
        node_bands = [node['n_bands'] for node in resonance_nodes[:20]]
        
        scatter = ax6.scatter(node_times, node_strengths, c=node_bands, cmap='hot',
                            s=200, alpha=0.8, edgecolors='black', linewidths=2)
        
        # Label top nodes
        for i, node in enumerate(resonance_nodes[:5]):
            ax6.text(node['time'], node['coupling_strength'], f"#{i+1}",
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        ax6.set_xlabel("Time (s)", fontsize=12)
        ax6.set_ylabel("Coupling Strength", fontsize=12)
        ax6.set_title(f"Top {min(20, len(resonance_nodes))} Resonance Nodes", fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label("Bands Coupled", fontsize=10)
    else:
        ax6.text(0.5, 0.5, "No resonance nodes found", ha='center', va='center',
                transform=ax6.transAxes, fontsize=14)
        ax6.axis('off')
    
    plt.suptitle(f"Frequency Band Intersection Analysis\n{len(trajectories)} bands, {len(intersections)} intersections, {len(resonance_nodes)} resonance nodes",
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    png_path = outdir / "band_intersections.png"
    pdf_path = outdir / "band_intersections.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {png_path}")


def visualize_cross_correlations(corr_data: Dict, traj_data: Dict, outdir: Path):
    """
    Visualize cross-band correlation matrix and key correlation functions.
    """
    correlations = corr_data['correlations']
    trajectories = traj_data['trajectories']
    n_bands = len(trajectories)
    
    if len(correlations) == 0:
        print("   No correlations to visualize")
        return
    
    fig = plt.figure(figsize=(28, 16))
    
    # ========== PANEL 1: Correlation Matrix (Peak Values) ==========
    ax1 = plt.subplot(2, 3, 1)
    
    # Build correlation matrix (max correlation) and zero-lag matrix
    corr_matrix = np.zeros((n_bands, n_bands))
    zero_lag_matrix = np.zeros((n_bands, n_bands))
    
    for corr in correlations:
        i, j = corr['band_i'], corr['band_j']
        corr_matrix[i, j] = corr['max_abs_corr']
        corr_matrix[j, i] = corr['max_abs_corr']
        
        # PATCH 1: Track zero-lag correlations
        zero_lag_idx = np.argmin(np.abs(corr['lags']))
        zero_lag_value = corr['correlation'][zero_lag_idx]
        zero_lag_matrix[i, j] = np.abs(zero_lag_value)
        zero_lag_matrix[j, i] = np.abs(zero_lag_value)
    
    im = ax1.imshow(corr_matrix, cmap='hot', aspect='auto', origin='lower')
    
    # PATCH 1: Overlay red contour at lag=0 (coherent core)
    contour_levels = [np.percentile(zero_lag_matrix[zero_lag_matrix > 0], 75)]  # Top 25% zero-lag
    contours = ax1.contour(zero_lag_matrix, levels=contour_levels, 
                          colors='red', linewidths=3, alpha=0.8)
    ax1.clabel(contours, inline=True, fontsize=10, fmt='Zero-lag core', colors='red')
    
    ax1.set_xlabel("Band Index", fontsize=12)
    ax1.set_ylabel("Band Index", fontsize=12)
    ax1.set_title("Cross-Band Correlation Matrix (Max |Corr|)\n+ Red Contour: Zero-Lag Coherent Core", 
                 fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("Max |Correlation|", fontsize=10)
    
    # ========== PANEL 2: Peak Lag Distribution ==========
    ax2 = plt.subplot(2, 3, 2)
    
    peak_lags = [corr['peak_lag'] for corr in correlations]
    ax2.hist(peak_lags, bins=50, color='cyan', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', ls='--', lw=2, label='Zero lag')
    ax2.set_xlabel("Peak Lag (s)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Cross-Band Correlation Peak Lags", fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========== PANEL 3: Top Correlation Functions ==========
    ax3 = plt.subplot(2, 3, 3)
    
    # Sort by max correlation
    sorted_corrs = sorted(correlations, key=lambda x: x['max_abs_corr'], reverse=True)
    
    for i, corr in enumerate(sorted_corrs[:10]):  # Top 10
        lags = corr['lags']
        corr_func = corr['correlation']
        # Normalize
        corr_func_norm = corr_func / np.max(np.abs(corr_func))
        
        label = f"f{corr['band_i']}-f{corr['band_j']} ({corr['freq_i']:.0f}-{corr['freq_j']:.0f} Hz)"
        ax3.plot(lags, corr_func_norm, alpha=0.7, lw=2, label=label if i < 5 else None)
    
    ax3.axvline(0, color='red', ls='--', lw=2, alpha=0.5)
    ax3.set_xlabel("Lag (s)", fontsize=12)
    ax3.set_ylabel("Normalized Correlation", fontsize=12)
    ax3.set_title("Top 10 Cross-Band Correlation Functions", fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-0.5, 0.5])  # Focus on near-zero lags
    
    # ========== PANEL 4: PATCH 2 - Low-F Zoom (Δf < 150 Hz) ==========
    ax4 = plt.subplot(2, 3, 4)
    
    # Filter for low-f pairs (Δf < 150 Hz)
    low_f_corrs = [corr for corr in correlations if abs(corr['freq_i'] - corr['freq_j']) < 150]
    
    if len(low_f_corrs) > 0:
        # Build low-f correlation matrix
        low_f_bands = set()
        for corr in low_f_corrs:
            low_f_bands.add(corr['band_i'])
            low_f_bands.add(corr['band_j'])
        low_f_bands = sorted(low_f_bands)
        
        n_low_f = len(low_f_bands)
        low_f_matrix = np.zeros((n_low_f, n_low_f))
        
        band_to_idx = {band: idx for idx, band in enumerate(low_f_bands)}
        
        for corr in low_f_corrs:
            i_idx = band_to_idx[corr['band_i']]
            j_idx = band_to_idx[corr['band_j']]
            low_f_matrix[i_idx, j_idx] = corr['max_abs_corr']
            low_f_matrix[j_idx, i_idx] = corr['max_abs_corr']
        
        im = ax4.imshow(low_f_matrix, cmap='hot', aspect='auto', origin='lower')
        
        # Overlay contours for quartile clusters
        if n_low_f > 3:
            quartile_levels = [np.percentile(low_f_matrix[low_f_matrix > 0], q) for q in [50, 75, 90]]
            contours = ax4.contour(low_f_matrix, levels=quartile_levels,
                                  colors=['yellow', 'orange', 'red'], linewidths=[2, 2.5, 3], alpha=0.8)
            ax4.clabel(contours, inline=True, fontsize=8, fmt='Q%d', colors='white')
        
        ax4.set_xlabel("Low-F Band Index", fontsize=12)
        ax4.set_ylabel("Low-F Band Index", fontsize=12)
        ax4.set_title(f"PATCH 2: Low-F Zoom (Δf<150 Hz)\n{len(low_f_corrs)} pairs, {n_low_f} bands",
                     fontsize=13, fontweight='bold', color='red')
        
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label("|Correlation|", fontsize=10)
    else:
        ax4.text(0.5, 0.5, "No low-f pairs", ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')
    
    # ========== PANEL 5: PATCH 3 - Coupling vs Depth ==========
    ax5 = plt.subplot(2, 3, 5)
    
    # Compute mean coupling strength and mean null depth per band
    band_coupling = {}
    band_depth = {}
    
    for traj in trajectories:
        band_idx = trajectories.index(traj)
        
        # Mean coupling: average correlation with all other bands
        couplings = []
        for corr in correlations:
            if corr['band_i'] == band_idx or corr['band_j'] == band_idx:
                couplings.append(corr['max_abs_corr'])
        
        band_coupling[band_idx] = np.mean(couplings) if len(couplings) > 0 else 0.0
        band_depth[band_idx] = traj['mean_power']
    
    coupling_vals = [band_coupling[i] for i in range(len(trajectories))]
    depth_vals = [band_depth[i] for i in range(len(trajectories))]
    freq_vals = [traj['freq_center'] for traj in trajectories]
    
    scatter = ax5.scatter(depth_vals, coupling_vals, c=freq_vals, cmap='viridis',
                         s=100, alpha=0.8, edgecolors='black', linewidths=1.5)
    
    # Fit trend line
    if len(depth_vals) > 2:
        z = np.polyfit(depth_vals, coupling_vals, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(depth_vals), max(depth_vals), 100)
        ax5.plot(x_fit, p(x_fit), 'r--', lw=3, alpha=0.8, label=f'Trend: slope={z[0]:.4f}')
        ax5.legend(fontsize=10, loc='best')
    
    ax5.set_xlabel("Mean Null Depth (dB)", fontsize=12)
    ax5.set_ylabel("Mean Coupling Strength", fontsize=12)
    ax5.set_title("PATCH 3: Coupling vs Depth\nDeeper Nulls = Stronger Coupling (Thermal Proof)",
                 fontsize=13, fontweight='bold', color='red')
    ax5.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label("Frequency (Hz)", fontsize=10)
    
    # ========== PANEL 6: Correlation vs Frequency Separation ==========
    ax6 = plt.subplot(2, 3, 6)
    
    freq_seps = [abs(corr['freq_i'] - corr['freq_j']) for corr in correlations]
    max_corrs = [corr['max_abs_corr'] for corr in correlations]
    
    scatter = ax6.scatter(freq_seps, max_corrs, c=max_corrs, cmap='plasma',
                         s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax6.set_xlabel("Frequency Separation (Hz)", fontsize=12)
    ax6.set_ylabel("Max |Correlation|", fontsize=12)
    ax6.set_title("Correlation Strength vs Frequency Separation", fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label("Max |Correlation|", fontsize=10)
    
    plt.suptitle(f"Cross-Band Correlation Analysis (CONVICTION DETONATION)\n{len(correlations)} band pairs analyzed",
                fontsize=16, fontweight='bold', y=0.995, color='red')
    
    plt.tight_layout()
    
    png_path = outdir / "cross_correlations.png"
    pdf_path = outdir / "cross_correlations.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {png_path}")


def main():
    print("="*70)
    print("FREQUENCY BAND INTERSECTION ANALYZER")
    print("="*70)
    
    outdir = Path("/home/ubuntu/band_intersection_results")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Build echo waveform
    print("\n🔧 Building echo waveform...")
    M_solar = 65.0
    epsilon = 1e-6
    duration = 2.0
    fs = 4096
    
    h, dt_echo, echo_amp, params = build_echo_waveform(M_solar, epsilon, duration, fs)
    print(f"   Echo delay: {dt_echo*1000:.2f} ms")
    print(f"   Echo amplitude: {echo_amp*100:.1f}%")
    
    # Compute spectrogram
    print("\n📈 Computing spectrogram...")
    TT, FF, Pdb = stft_power_db(h, fs, nperseg=512, hop=32)
    
    # Detect nulls per band (adaptive bands from low-f patches)
    print("\n🔍 Detecting nulls across frequency bands...")
    freq_low_dense = np.linspace(40, 300, 30)
    freq_high_sparse = np.linspace(300, 600, 15)
    freq_edges = np.concatenate([freq_low_dense, freq_high_sparse[1:]])
    freq_bands = [(freq_edges[i], freq_edges[i+1]) for i in range(len(freq_edges)-1)]
    
    band_data = detect_nulls_per_band(TT, FF, Pdb, freq_bands, percentile=20.0, prominence_factor=0.18)
    total_nulls = sum(b['n_nulls'] for b in band_data['bands'])
    print(f"   ✅ Found {total_nulls} nulls across {len(freq_bands)} bands")
    
    # Track band trajectories
    print("\n📍 Tracking band trajectories...")
    traj_data = track_band_trajectories(band_data, duration)
    print(f"   ✅ Tracked {len(traj_data['trajectories'])} band trajectories")
    
    # Find intersections
    print("\n🔗 Finding band intersections...")
    intersections = find_band_intersections(traj_data, time_tolerance=0.02)
    
    # Identify resonance nodes
    print("\n⭐ Identifying resonance nodes...")
    resonance_nodes = identify_resonance_nodes(intersections, time_window=0.05, min_bands=3)
    
    # Compute cross-correlations
    print("\n📊 Computing cross-band correlations...")
    corr_data = compute_cross_band_correlations(traj_data, dt=0.01)
    print(f"   ✅ Computed {len(corr_data['correlations'])} correlation functions")
    
    # Visualize
    print("\n🎨 Generating visualizations...")
    visualize_band_intersections(traj_data, intersections, resonance_nodes, outdir)
    visualize_cross_correlations(corr_data, traj_data, outdir)
    
    # Save results
    print("\n💾 Saving results...")
    results = {
        'n_bands': len(traj_data['trajectories']),
        'n_intersections': len(intersections),
        'n_resonance_nodes': len(resonance_nodes),
        'echo_delay_ms': dt_echo * 1000,
        'echo_amplitude_pct': echo_amp * 100,
        'duration_s': duration,
        'top_resonance_nodes': resonance_nodes[:10] if len(resonance_nodes) > 0 else [],
        'trajectory_stats': {
            'mean_velocity': np.mean([t['velocity'] for t in traj_data['trajectories']]),
            'mean_power_slope': np.mean([t['power_slope'] for t in traj_data['trajectories']]),
        }
    }
    
    with open(outdir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Complete! Results saved to: {outdir}")
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Bands analyzed: {results['n_bands']}")
    print(f"Intersections found: {results['n_intersections']}")
    print(f"Resonance nodes: {results['n_resonance_nodes']}")
    print(f"Mean null velocity: {results['trajectory_stats']['mean_velocity']:.2f} nulls/s")
    print(f"Mean power slope: {results['trajectory_stats']['mean_power_slope']:.2f} dB/s")
    print("="*70)


if __name__ == "__main__":
    main()

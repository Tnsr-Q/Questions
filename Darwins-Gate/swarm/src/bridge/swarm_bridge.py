"""
Unified Swarm Bridge: Trans-Planckian Dimensional Reduction ↔ ET Signal Detection

This module connects:
1. HypercubeSwarmTrainer (learns 37D→31D contextual reduction)
2. TensorCell swarm (processes gravitational wave echoes)
3. Waveform builder (generates test signals with Planck-scale structure)

The bridge enables bidirectional information flow:
- ET echo parameters → dimensional reduction strategy
- Reduction quality (R) → TensorCell fitness boost
"""

import torch
import numpy as np
import vibetensor as vbt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from waveform_builder import build_waveform_with_echoes, prepare_tensorcell_inputs
from tensorcell import TensorCell
from casual_enforcer import CausalEnforcer

# ═══════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PlanckScaleSignature:
    """
    Extracted signature from ET signal that encodes dimensional structure.
    
    The 37→31 dimensional gap should appear in the echo delay structure
    if trans-Planckian physics leaves an imprint.
    """
    dt_echo: float              # Echo delay (seconds)
    R_s: complex                # Surface reflectivity (magnitude + phase)
    dimensional_gap: float      # Inferred from ln(|R_s|) / dt_echo
    cavity_resonances: np.ndarray  # Frequencies where echoes resonate
    quality_factor: float       # Q-factor of cavity
    
    def to_hypercube_state(self) -> np.ndarray:
        """
        Encode Planck signature as 8D state for JacobianHypercube.
        
        Maps cavity parameters to dimensional reduction control signals.
        """
        state = np.zeros(8)
        
        # Dimensional gap encodes reduction pressure (37-31=6)
        state[0] = self.dimensional_gap / 10.0  # Normalize to ~0.6
        
        # Reflectivity encodes anchor stability
        state[1] = np.abs(self.R_s)
        state[2] = np.angle(self.R_s) / np.pi  # Phase normalized to [-1,1]
        
        # Echo delay encodes iteration timescale
        state[3] = np.tanh(self.dt_echo * 100)  # Compress to [-1,1]
        
        # Cavity resonances encode clique structure
        state[4] = len(self.cavity_resonances) / 100.0
        state[5] = self.quality_factor / 1000.0
        
        # Reserved for future metrics
        state[6:8] = 0.0
        
        return state

@dataclass
class UnifiedSwarmState:
    """
    Complete state of the unified swarm system.
    Synchronizes dimensional reduction with ET signal processing.
    """
    # Hypercube population
    hypercube_agents: List['HypercubeReductionAgent']
    best_hypercube_idx: int
    best_R: float
    
    # TensorCell population
    tensorcells: List[TensorCell]
    best_cell_idx: int
    best_detection_score: float
    
    # Coupling parameters
    planck_signature: Optional[PlanckScaleSignature] = None
    coupling_strength: float = 0.5  # How much ET signal influences reduction
    
    # Training metrics
    episode: int = 0
    total_detections: int = 0
    confirmed_planck_events: int = 0

# ═══════════════════════════════════════════════════════════════════════
# Planck Signature Extractor
# ═══════════════════════════════════════════════════════════════════════

class PlanckSignatureExtractor:
    """
    Analyzes ET waveform metadata to extract trans-Planckian signatures.
    
    Looks for:
    1. Echo delay structure consistent with Planck-scale cavity
    2. Frequency-dependent reflectivity indicating dimensional compression
    3. Resonance patterns matching 37D→31D reduction geometry
    """
    
    def __init__(self, expected_dim_gap=6):
        self.expected_dim_gap = expected_dim_gap
        
    def extract(self, meta: dict, h_echo: np.ndarray) -> PlanckScaleSignature:
        """
        Extract Planck signature from waveform metadata.
        
        Parameters
        ----------
        meta : dict
            Waveform metadata from build_waveform_with_echoes()
        h_echo : np.ndarray
            Echo component of waveform
            
        Returns
        -------
        PlanckScaleSignature
        """
        dt_echo = meta['dt_echo']
        R_s = meta['R_s']
        
        # Compute dimensional gap from reflectivity
        # Theory: ln(|R_s|) / dt_echo ~ dimensional compression rate
        if np.abs(R_s) > 0:
            dimensional_gap = -np.log(np.abs(R_s)) / (dt_echo + 1e-10)
        else:
            dimensional_gap = 0.0
        
        # Find cavity resonances in echo spectrum
        H_echo = np.fft.rfft(h_echo)
        freqs = meta['freqs']
        
        # Peaks in echo spectrum indicate resonances
        power = np.abs(H_echo) ** 2
        resonance_threshold = np.percentile(power, 95)
        resonance_mask = power > resonance_threshold
        cavity_resonances = freqs[resonance_mask]
        
        # Estimate quality factor from echo decay
        quality_factor = self._estimate_quality_factor(h_echo, dt_echo)
        
        return PlanckScaleSignature(
            dt_echo=dt_echo,
            R_s=R_s,
            dimensional_gap=dimensional_gap,
            cavity_resonances=cavity_resonances,
            quality_factor=quality_factor
        )
    
    @staticmethod
    def _estimate_quality_factor(h_echo: np.ndarray, dt_echo: float) -> float:
        """
        Estimate cavity Q-factor from echo envelope decay.
        
        Q = π * dt_echo / decay_time
        """
        # Envelope via Hilbert transform
        analytic_signal = np.fft.ifft(
            np.fft.fft(h_echo) * np.concatenate([
                [1], 2*np.ones(len(h_echo)//2-1), [1], np.zeros(len(h_echo)//2)
            ])
        )
        envelope = np.abs(analytic_signal)
        
        # Fit exponential decay
        t = np.arange(len(envelope))
        log_env = np.log(envelope + 1e-10)
        
        # Linear fit to log(envelope)
        coeffs = np.polyfit(t, log_env, 1)
        decay_rate = -coeffs[0]
        
        if decay_rate > 0:
            decay_time = 1.0 / decay_rate
            Q = np.pi * dt_echo / (decay_time + 1e-10)
        else:
            Q = 0.0
        
        return Q

# ═══════════════════════════════════════════════════════════════════════
# Unified Swarm Coordinator
# ═══════════════════════════════════════════════════════════════════════

class UnifiedSwarmCoordinator:
    """
    Master coordinator for hypercube-guided dimensional reduction
    and TensorCell-based ET signal detection.
    
    Architecture:
    
    1. Generate ET waveform with Planck-scale echoes
    2. Extract Planck signature (dt_echo, R_s, resonances)
    3. Feed signature to hypercube agents → learn optimal reduction
    4. Use reduced 31D geometry to configure TensorCells
    5. TensorCells process waveform → detection score
    6. Detection quality feeds back to hypercube fitness
    
    This creates a closed loop where ET physics informs quantum geometry.
    """
    
    def __init__(
        self,
        n_hypercube_agents: int = 16,
        n_tensorcells: int = 32,
        rays_37d: Optional[np.ndarray] = None,
        M_QM: float = 0.750
    ):
        self.n_hypercube_agents = n_hypercube_agents
        self.n_tensorcells = n_tensorcells
        self.M_QM = M_QM
        
        # Initialize hypercube swarm
        from hypercube_trainer import HypercubeSwarmTrainer
        self.hypercube_trainer = HypercubeSwarmTrainer(
            n_agents=n_hypercube_agents
        )
        
        # Initialize TensorCell swarm
        self.tensorcells = self._initialize_tensorcell_swarm()
        
        # Planck signature extractor
        self.signature_extractor = PlanckSignatureExtractor()
        
        # State tracking
        self.state = UnifiedSwarmState(
            hypercube_agents=self.hypercube_trainer.agents,
            best_hypercube_idx=0,
            best_R=0.0,
            tensorcells=self.tensorcells,
            best_cell_idx=0,
            best_detection_score=0.0
        )
        
        # Contextuality rays (37D)
        self.rays_37d = rays_37d
        
    def _initialize_tensorcell_swarm(self) -> List[TensorCell]:
        """Create diverse TensorCell population with varying geometries."""
        cells = []
        
        for i in range(self.n_tensorcells):
            # Each cell gets unique geometric configuration
            genome = {
                'fitness_ema_alpha': 0.1,
                'apply_rotation': np.random.rand() > 0.5,
                'use_geometric_projections': np.random.rand() > 0.5,
                'num_projections': np.random.randint(2, 6),
                'projection_rank': np.random.randint(2, 4),
                'tile_m': 128,
                'tile_n': 128,
                'tile_k': 64,
                'precision': 'fp32',
                
                # KK enforcement parameters (evolvable)
                'kk_params': {
                    'kk_tolerance': np.random.uniform(0.5, 1.0),
                    'enforce_kk': True
                }
            }
            
            cells.append(TensorCell(genome))
        
        return cells
    
    def train_unified_episode(
        self,
        t: np.ndarray,
        Mtot_solar: float = 30.0,
        epsilon: float = 1e-40,
        n_episodes: int = 50
    ):
        """
        Main training loop: unified ET detection + dimensional reduction.
        
        Each episode:
        1. Generate waveform with echoes
        2. Extract Planck signature
        3. Train hypercubes on reduction task (guided by signature)
        4. Train TensorCells on detection task (using best hypercube)
        5. Cross-pollinate: detection score boosts hypercube fitness
        """
        
        print(f"\n{'='*80}")
        print(f"UNIFIED SWARM TRAINING: {n_episodes} EPISODES")
        print(f"{'='*80}\n")
        
        for episode in range(n_episodes):
            print(f"\n{'─'*80}")
            print(f"Episode {episode+1}/{n_episodes}")
            print(f"{'─'*80}\n")
            
            # ═══════════════════════════════════════════════════════════
            # Phase 1: Generate ET Waveform
            # ═══════════════════════════════════════════════════════════
            
            # Generate base merger waveform (placeholder - use your actual model)
            h0 = self._generate_merger_waveform(t, Mtot_solar)
            
            # Build waveform with Planck-scale echoes
            h_total, h0, h_echo, meta = build_waveform_with_echoes(
                t=t,
                Mtot_solar=Mtot_solar,
                epsilon=epsilon,
                h0=h0,
                R_s_mag=0.70,
                R_s_phase=0.30 * np.pi
            )
            
            print(f"Generated waveform: dt_echo={meta['dt_echo']:.6f}s, |R_s|={np.abs(meta['R_s']):.3f}")
            
            # ═══════════════════════════════════════════════════════════
            # Phase 2: Extract Planck Signature
            # ═══════════════════════════════════════════════════════════
            
            signature = self.signature_extractor.extract(meta, h_echo)
            self.state.planck_signature = signature
            
            print(f"Planck signature: dim_gap={signature.dimensional_gap:.3f}, Q={signature.quality_factor:.1f}")
            print(f"  Resonances: {len(signature.cavity_resonances)} peaks found")
            
            # ═══════════════════════════════════════════════════════════
            # Phase 3: Train Hypercube Agents (Dimensional Reduction)
            # ═══════════════════════════════════════════════════════════
            
            if self.rays_37d is not None:
                print("\n[Hypercube Training]")
                
                # Inject Planck signature into reduction environment
                reduction_results = self._train_hypercubes_with_signature(signature)
                
                # Update best hypercube
                best_idx = np.argmax([r['R'] for r in reduction_results])
                self.state.best_hypercube_idx = best_idx
                self.state.best_R = reduction_results[best_idx]['R']
                
                print(f"  Best hypercube: Agent {best_idx}, R={self.state.best_R:.4f}")
            
            # ═══════════════════════════════════════════════════════════
            # Phase 4: Train TensorCells (ET Signal Detection)
            # ═══════════════════════════════════════════════════════════
            
            print("\n[TensorCell Training]")
            
            # Prepare inputs for TensorCells
            strain_tensor = vbt.tensor(h_total.reshape(1, -1))  # (1, T)
            comb_mask = self._generate_comb_mask(len(t))
            
            inputs = prepare_tensorcell_inputs(
                strain_dlpack=strain_tensor.to_dlpack(),
                comb_mask_dlpack=comb_mask.to_dlpack(),
                meta=meta
            )
            
            # Each TensorCell processes the waveform
            detection_results = []
            
            for cell_idx, cell in enumerate(self.tensorcells):
                result = cell.solve_physics(inputs)
                detection_results.append({
                    'cell_idx': cell_idx,
                    'score': result['score'],
                    'elapsed_ms': result['elapsed_ms'],
                    'kk_applied': result['kk_info']['applied']
                })
            
            # Update best TensorCell
            best_cell_idx = np.argmax([r['score'] for r in detection_results])
            self.state.best_cell_idx = best_cell_idx
            self.state.best_detection_score = detection_results[best_cell_idx]['score']
            
            print(f"  Best TensorCell: Cell {best_cell_idx}, score={self.state.best_detection_score:.4f}")
            
            # ═══════════════════════════════════════════════════════════
            # Phase 5: Cross-Pollination (Bidirectional Feedback)
            # ═══════════════════════════════════════════════════════════
            
            self._cross_pollinate(reduction_results, detection_results)
            
            # ═══════════════════════════════════════════════════════════
            # Episode Summary
            # ═══════════════════════════════════════════════════════════
            
            self.state.episode = episode + 1
            self._print_episode_summary()
    
    def _train_hypercubes_with_signature(
        self, 
        signature: PlanckScaleSignature
    ) -> List[Dict]:
        """
        Train hypercube agents using Planck signature as prior.
        
        The signature informs:
        - Initial clique weights (from cavity resonances)
        - SVD truncation threshold (from Q-factor)
        - Iteration damping (from dimensional gap)
        """
        from reduction_environment import ReductionEnvironment
        
        results = []
        
        # Encode signature as environment bias
        signature_state = signature.to_hypercube_state()
        
        for agent in self.state.hypercube_agents:
            env = ReductionEnvironment(
                self.rays_37d, 
                self.M_QM,
                signature_bias=signature_state  # Inject Planck prior
            )
            
            state = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                agent.store_transition(
                    state, action, log_prob, reward, value, done
                )
                
                episode_reward += reward
                state = next_state
            
            results.append({
                'agent_id': agent.agent_id,
                'R': info.get('R', 0.0),
                'reward': episode_reward
            })
        
        # PPO update
        self.hypercube_trainer._ppo_update()
        
        return results
    
    def _cross_pollinate(
        self, 
        reduction_results: List[Dict],
        detection_results: List[Dict]
    ):
        """
        Bidirectional fitness transfer between hypercubes and TensorCells.
        
        - High R → boost fitness of TensorCells that used that hypercube
        - High detection score → boost fitness of hypercube that enabled it
        """
        coupling = self.state.coupling_strength
        
        # TensorCell boost from dimensional reduction quality
        best_R = self.state.best_R
        for cell in self.tensorcells:
            # Cells that respect geometry get fitness bonus
            if cell.genome.get('use_geometric_projections', False):
                cell.fitness += coupling * best_R * 10.0
        
        # Hypercube boost from detection success
        best_detection = self.state.best_detection_score
        best_hypercube = self.state.hypercube_agents[self.state.best_hypercube_idx]
        
        # Reward hypercube if its reduction enabled good detection
        hypercube_bonus = coupling * best_detection * 5.0
        
        # Store bonus for next PPO update (add to trajectory buffer rewards)
        if best_hypercube.trajectory_buffer['rewards']:
            best_hypercube.trajectory_buffer['rewards'][-1] += hypercube_bonus
    
    def _generate_merger_waveform(self, t: np.ndarray, Mtot_solar: float) -> np.ndarray:
        """
        Generate base merger waveform (placeholder).
        Replace with your actual inspiral-merger-ringdown model.
        """
        # Simple Gaussian chirp as placeholder
        f0 = 100  # Hz
        tau = 0.01  # seconds
        
        phase = 2 * np.pi * f0 * t * (1 + t / tau)
        envelope = np.exp(-((t - t.mean()) / tau)**2)
        
        h0 = envelope * np.sin(phase)
        return h0
    
    def _generate_comb_mask(self, length: int) -> vbt.tensor:
        """Generate frequency comb mask for TensorCell reduction."""
        mask = np.ones(length)
        # Simple comb: keep every 10th frequency bin
        mask[::10] = 2.0
        return vbt.tensor(mask.reshape(1, -1))
    
    def _print_episode_summary(self):
        """Print detailed episode statistics."""
        print(f"\n{'─'*80}")
        print(f"EPISODE {self.state.episode} SUMMARY")
        print(f"{'─'*80}")
        print(f"Dimensional Reduction:")
        print(f"  Best R: {self.state.best_R:.4f} (Agent {self.state.best_hypercube_idx})")
        print(f"ET Signal Detection:")
        print(f"  Best Score: {self.state.best_detection_score:.4f} (Cell {self.state.best_cell_idx})")
        print(f"Planck Signature:")
        if self.state.planck_signature:
            sig = self.state.planck_signature
            print(f"  Dimensional Gap: {sig.dimensional_gap:.3f}")
            print(f"  Quality Factor: {sig.quality_factor:.1f}")
            print(f"  Resonances: {len(sig.cavity_resonances)}")
        print(f"{'─'*80}\n")
    
    def get_best_system(self) -> Tuple[TensorCell, 'JacobianHypercube']:
        """
        Return the optimal hypercube-TensorCell pair.
        
        This pair represents the best configuration for:
        1. Dimensional reduction (37D→31D)
        2. ET signal detection with Planck-scale echoes
        """
        best_cell = self.tensorcells[self.state.best_cell_idx]
        best_hypercube = self.state.hypercube_agents[self.state.best_hypercube_idx].hypercube
        
        return best_cell, best_hypercube
    
    def deploy_operational(self, t: np.ndarray, live_strain: np.ndarray):
        """
        Deploy trained system for operational ET detection.
        
        Uses best hypercube-TensorCell pair to process live data.
        """
        best_cell, best_hypercube = self.get_best_system()
        
        # Generate metadata (in real deployment, extract from detector)
        meta = {
            'fs': 1.0 / (t[1] - t[0]),
            'omega_full': 2 * np.pi * np.fft.fftfreq(len(t), d=t[1] - t[0])
        }
        
        # Prepare inputs
        strain_tensor = vbt.tensor(live_strain.reshape(1, -1))
        comb_mask = self._generate_comb_mask(len(t))
        
        inputs = prepare_tensorcell_inputs(
            strain_dlpack=strain_tensor.to_dlpack(),
            comb_mask_dlpack=comb_mask.to_dlpack(),
            meta=meta
        )
        
        # Process with best cell
        result = best_cell.solve_physics(inputs)
        
        # If contextual reduction is needed, use best hypercube
        if result['score'] > 0.8:  # High confidence detection
            print("High-confidence detection! Analyzing with hypercube...")
            
            # Use hypercube to guide further analysis
            # (This would connect to your 37D→31D reduction pipeline)
            
        return result

# ═══════════════════════════════════════════════════════════════════════
# Usage Example
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Time array for waveform
    fs = 4096  # Hz
    duration = 4  # seconds
    t = np.arange(0, duration, 1/fs)
    
    # Load 37D contextuality rays (your Kochen-Specker configuration)
    rays_37d = load_contextuality_rays()  # Implement this
    M_QM = 0.750  # Known ideal violation
    
    # Initialize unified swarm
    coordinator = UnifiedSwarmCoordinator(
        n_hypercube_agents=16,
        n_tensorcells=32,
        rays_37d=rays_37d,
        M_QM=M_QM
    )
    
    # Train unified system
    coordinator.train_unified_episode(
        t=t,
        Mtot_solar=30.0,
        epsilon=1e-40,
        n_episodes=50
    )
    
    # Extract best system
    best_cell, best_hypercube = coordinator.get_best_system()
    
    print(f"\nBest System Configuration:")
    print(f"  TensorCell: {best_cell}")
    print(f"  Hypercube R: {coordinator.state.best_R:.4f}")
    print(f"  Detection Score: {coordinator.state.best_detection_score:.4f}")
    
    # Deploy for operational detection
    # live_strain = get_live_et_data()  # Your real-time data feed
    # result = coordinator.deploy_operational(t, live_strain)

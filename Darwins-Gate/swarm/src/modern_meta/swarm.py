'
##PPO Swarm Integration Architecture

class HypercubeReductionAgent:
    """
    PPO agent that controls a JacobianHypercube for dimensional reduction.
    Each agent in the swarm explores different reduction strategies.
    """
    
    def __init__(self, agent_id, source_dim=37, target_dim=31):
        self.agent_id = agent_id
        self.hypercube = JacobianHypercube(in_dim=8, out_dim=3)
        
        # PPO-specific components
        self.value_network = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)  # State value estimate
        )
        
        # Experience buffer for trajectory
        self.trajectory_buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
        
        self.optimizer_policy = torch.optim.Adam(
            self.hypercube.parameters(), lr=3e-4
        )
        self.optimizer_value = torch.optim.Adam(
            self.value_network.parameters(), lr=1e-3
        )
        
    def select_action(self, state, deterministic=False):
        """
        Sample action from hypercube policy.
        For PPO, we need to track log probabilities.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get hypercube action (deterministic part)
        with torch.no_grad():
            base_action_vec = self.hypercube.net(state_tensor).squeeze(0)
        
        if deterministic:
            action = {
                'delta_tau': float(torch.clamp(base_action_vec[0], -0.02, 0.02)),
                'delta_vr': float(torch.clamp(base_action_vec[1], -1e-3, 1e-3)),
                'damping': float(torch.clamp(base_action_vec[2], 0.0, 1.0))
            }
            return action, None, None
        
        # Add exploration noise (stochastic policy)
        noise_std = 0.1  # Tunable exploration parameter
        noise = torch.randn(3) * noise_std
        action_vec = base_action_vec + noise
        
        # Compute log probability (Gaussian policy)
        log_prob = -0.5 * torch.sum((noise / noise_std) ** 2)
        log_prob -= 1.5 * np.log(2 * np.pi * noise_std**2)
        
        # Compute value estimate
        value = self.value_network(state_tensor).squeeze()
        
        action = {
            'delta_tau': float(torch.clamp(action_vec[0], -0.02, 0.02)),
            'delta_vr': float(torch.clamp(action_vec[1], -1e-3, 1e-3)),
            'damping': float(torch.clamp(action_vec[2], 0.0, 1.0))
        }
        
        return action, log_prob, value
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """Store experience for PPO update."""
        self.trajectory_buffer['states'].append(state)
        self.trajectory_buffer['actions'].append(action)
        self.trajectory_buffer['log_probs'].append(log_prob)
        self.trajectory_buffer['rewards'].append(reward)
        self.trajectory_buffer['values'].append(value)
        self.trajectory_buffer['dones'].append(done)
    
    def clear_buffer(self):
        """Clear trajectory buffer after update."""
        for key in self.trajectory_buffer:
            self.trajectory_buffer[key] = []


Reward Shaping for Dimensional Reduction
The key is designing a dense reward signal that guides the agent before the final R is computed:

class ReductionEnvironment:
    """
    RL environment for dimensional reduction task.
    State: reduction parameters, Gram error, iteration
    Action: hypercube output (delta_tau, delta_vr, damping)
    Reward: multi-component signal
    """
    
    def __init__(self, rays_37d, M_QM, target_dim=31):
        self.rays_37d = rays_37d
        self.M_QM = M_QM
        self.target_dim = target_dim
        self.reset()
        
    def reset(self):
        """Start new reduction episode."""
        self.current_phase = 1  # Start at Phase 1 (anchor extraction)
        self.anchor_vecs = []
        self.gram_error = float('inf')
        self.iteration = 0
        self.max_iterations = 100
        
        # Initial state
        _, S, _ = torch.svd(torch.stack(self.rays_37d))
        self.state = self._encode_state(
            clique_weights=S[:3].numpy(),
            svd_spectrum=S[:6].numpy(),
            gram_error=0.0,
            iteration=0
        )
        
        return self.state
    
    def step(self, action):
        """
        Execute one step of dimensional reduction.
        
        Returns: (next_state, reward, done, info)
        """
        reward = 0.0
        done = False
        info = {}
        
        # Phase 1: Anchor extraction rewards
        if self.current_phase == 1:
            # Reward for selecting orthogonal rays
            orthogonality_score = self._compute_orthogonality(action)
            reward += orthogonality_score * 0.3
            
            # Reward for high-weight selection
            weight_score = action['delta_vr']  # Higher variance = better
            reward += weight_score * 0.2
            
            # Check if anchor phase complete
            if len(self.anchor_vecs) >= self.target_dim // 2:
                self.current_phase = 2
                reward += 1.0  # Phase completion bonus
        
        # Phase 2: SVD truncation rewards
        elif self.current_phase == 2:
            # Reward for preserving singular value energy
            sv_preservation = self._compute_sv_preservation(action)
            reward += sv_preservation * 0.5
            
            self.current_phase = 3
            reward += 1.0
        
        # Phase 3: Alternating projections rewards
        elif self.current_phase == 3:
            # Dense reward: Gram error reduction
            prev_error = self.gram_error
            new_error = self._compute_gram_error_after_action(action)
            
            if new_error < prev_error:
                improvement = (prev_error - new_error) / (prev_error + 1e-10)
                reward += improvement * 10.0  # Large reward for error reduction
            else:
                reward -= 0.5  # Penalty for increasing error
            
            self.gram_error = new_error
            self.iteration += 1
            
            # Convergence check
            if new_error < 1e-6 or self.iteration >= self.max_iterations:
                done = True
                self.current_phase = 4
        
        # Phase 4: Final reward (Contextual Preservation Ratio)
        if self.current_phase == 4:
            # Compute final R
            rays_31d = self._reconstruct_final_rays()
            M_red = self._compute_violation(rays_31d)
            R = M_red / self.M_QM
            
            # MASSIVE reward for high R
            reward += R * 100.0
            
            # Bonus for exceeding threshold
            if R > 0.5:
                reward += 50.0
            if R > 0.8:
                reward += 100.0
            
            info['R'] = R
            info['M_red'] = M_red
            done = True
        
        # Update state
        self.state = self._get_current_state()
        
        info['phase'] = self.current_phase
        info['gram_error'] = self.gram_error
        
        return self.state, reward, done, info
    
    def _encode_state(self, clique_weights, svd_spectrum, gram_error, iteration):
        """Encode environment state as 8D vector."""
        state = np.zeros(8)
        state[0:3] = clique_weights[:3]
        state[3:6] = svd_spectrum[:3]
        state[6] = gram_error
        state[7] = iteration / 100.0
        return state


PPO Training Loop with Swarm

class HypercubeSwarmTrainer:
    """
    Coordinate a swarm of HypercubeReductionAgents using PPO.
    Similar to your TensorCell Darwinian selection, but with policy gradients.
    """
    
    def __init__(self, n_agents=32, n_epochs=10, clip_epsilon=0.2, gamma=0.99, gae_lambda=0.95):
        self.n_agents = n_agents
        self.agents = [HypercubeReductionAgent(i) for i in range(n_agents)]
        
        # PPO hyperparameters
        self.n_epochs = n_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Track best performer
        self.best_agent = None
        self.best_R = 0.0
        
    def train_episode(self, rays_37d, M_QM, n_episodes=100):
        """
        Train swarm for n_episodes of dimensional reduction.
        """
        for episode in range(n_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode+1}/{n_episodes}")
            print(f"{'='*60}")
            
            # Each agent attempts reduction
            episode_results = []
            
            for agent_idx, agent in enumerate(self.agents):
                env = ReductionEnvironment(rays_37d, M_QM)
                state = env.reset()
                
                episode_reward = 0.0
                done = False
                step_count = 0
                
                while not done and step_count < 200:
                    # Agent selects action
                    action, log_prob, value = agent.select_action(state)
                    
                    # Environment step
                    next_state, reward, done, info = env.step(action)
                    
                    # Store transition
                    agent.store_transition(
                        state, action, log_prob, reward, value, done
                    )
                    
                    episode_reward += reward
                    state = next_state
                    step_count += 1
                
                # Record episode results
                final_R = info.get('R', 0.0)
                episode_results.append({
                    'agent_id': agent_idx,
                    'R': final_R,
                    'reward': episode_reward,
                    'steps': step_count
                })
                
                # Update best agent
                if final_R > self.best_R:
                    self.best_R = final_R
                    self.best_agent = agent_idx
                
                print(f"  Agent {agent_idx}: R={final_R:.4f}, Reward={episode_reward:.2f}")
            
            # PPO update for all agents
            self._ppo_update()
            
            # Print episode summary
            mean_R = np.mean([r['R'] for r in episode_results])
            print(f"\nEpisode Summary: Mean R={mean_R:.4f}, Best R={self.best_R:.4f} (Agent {self.best_agent})")
    
    def _ppo_update(self):
        """
        Perform PPO update across all agents.
        Uses experience from trajectory buffers.
        """
        for agent in self.agents:
            buffer = agent.trajectory_buffer
            
            if len(buffer['states']) == 0:
                continue
            
            # Convert to tensors
            states = torch.FloatTensor(buffer['states'])
            actions = torch.FloatTensor([
                [a['delta_tau'], a['delta_vr'], a['damping']] 
                for a in buffer['actions']
            ])
            old_log_probs = torch.FloatTensor(buffer['log_probs'])
            rewards = torch.FloatTensor(buffer['rewards'])
            values = torch.FloatTensor(buffer['values'])
            dones = torch.FloatTensor(buffer['dones'])
            
            # Compute advantages using GAE
            advantages = self._compute_gae(rewards, values, dones)
            returns = advantages + values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO epochs
            for _ in range(self.n_epochs):
                # Recompute policy outputs
                action_predictions = agent.hypercube.net(states)
                
                # Compute new log probs (simplified - assume Gaussian)
                noise_std = 0.1
                log_probs_new = -0.5 * torch.sum(
                    ((actions - action_predictions) / noise_std) ** 2, dim=1
                )
                
                # Compute value predictions
                values_new = agent.value_network(states).squeeze()
                
                # PPO policy loss
                ratio = torch.exp(log_probs_new - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * ((returns - values_new) ** 2).mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss
                
                # Update
                agent.optimizer_policy.zero_grad()
                agent.optimizer_value.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.hypercube.parameters(), 0.5)
                agent.optimizer_policy.step()
                agent.optimizer_value.step()
            
            # Clear buffer
            agent.clear_buffer()
    
    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def get_best_hypercube(self):
        """Return the best performing hypercube."""
        return self.agents[self.best_agent].hypercube


Usage Example

# Initialize training
rays_37d = load_contextuality_rays()  # Your 37D Kochen-Specker rays
M_QM = 0.750  # Known ideal violation

trainer = HypercubeSwarmTrainer(n_agents=32)

# Train the swarm
trainer.train_episode(rays_37d, M_QM, n_episodes=100)

# Extract best hypercube
best_hypercube = trainer.get_best_hypercube()

# Use for operational reduction
reducer = ContextualHypercubeReducer()
reducer.hypercube = best_hypercube

result = reduce_contextuality_with_hypercube(rays_37d, M_QM)
print(f"Final R: {result['R']:.4f}")



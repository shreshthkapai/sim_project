"""
dynamics.py

Dynamics evolution (core simulation loop):
- 5-term dynamics equation with RK45 integration
- Temporary Hebbian weight updates
- Stochastic noise via Euler-Maruyama
- Windowed convergence checking
- Operator splitting for numerical stability
- ✅ NEW: Biological mechanisms (adaptation, synaptic depression, divisive normalization)
"""

import numpy as np
from scipy.integrate import solve_ivp
from collections import deque
from typing import Dict, Tuple, Optional
import time

from sim_config import SimulationParameters
from sim_utils import validate_activations


class DynamicsEngine:
    """Handles neural dynamics integration with RK45 + Hebbian updates + biological mechanisms."""
    
    def __init__(self, graph_loader, state, params: Optional[SimulationParameters] = None):
        self.graph = graph_loader
        self.state = state
        self.params = params if params is not None else SimulationParameters()
        
        self.entity_inhibition = self._build_entity_inhibition_matrix()
        
        # ✅ NEW: Biological state variables
        self.adaptation = np.zeros(graph_loader.num_nodes, dtype=np.float32)  # Spike-rate adaptation
        self.synaptic_resources = np.ones(graph_loader.num_nodes, dtype=np.float32)  # Synaptic depression (1=full, 0=depleted)
        
        # Biological time constants
        self.tau_adapt = 0.2  # 200ms - adaptation builds up slowly
        self.tau_recovery = 0.5  # 500ms - synaptic recovery
        self.depletion_rate = 0.1  # How fast synapses deplete
        self.gamma_adapt = 0.5  # Adaptation strength
        self.divisive_strength = 0.05  # Divisive normalization strength
        
        self.trajectory_times = []
        self.trajectory_states = []
        
        # Convergence tracking
        self.W = 5  # Window size
        self.P = 2  # Patience
        self.eps_rel = 1e-3  # Relative tolerance
        self.eps_abs = 1e-4  # Absolute tolerance
        self.t_min = 1.0  # Transient time
        
        self.derivative_history = deque(maxlen=self.W)
        self.state_change_history = deque(maxlen=self.W)
        self.consecutive_passes = 0
        self.x_prev_macro = None
        
        print("Dynamics engine initialized (with biological mechanisms)")
    
    def _build_entity_inhibition_matrix(self) -> np.ndarray:
        """Build entity inhibition using existing co-occurrence edges from graph."""
        n = self.graph.num_nodes
        I_entity = np.zeros((n, n), dtype=np.float32)
        
        # ✅ STRONGER inhibition parameters (brain has ~4x stronger inhibition)
        alpha = 0.8  # Increased from 0.7
        baseline_same = 0.5  # Increased from 0.3
        beta = 0.3  # Increased from 0.1 (3x stronger!)
        
        print(f"Building entity inhibition from graph co-occurrence edges...")
        
        edge_count = 0
        for source, target, edge_data in self.graph.graph.edges(data=True):
            # Only process entity-entity co-occurrence edges
            if not (source.startswith('entity_') and target.startswith('entity_')):
                continue
            
            if edge_data.get('edge_type') != 'co_occurs':
                continue
            
            idx_i = self.graph.node_to_idx[source]
            idx_j = self.graph.node_to_idx[target]
            
            # Edge weight represents co-occurrence strength (0-1)
            co_occur_rate = edge_data.get('weight', 0.0)
            
            # Check if same category
            same_category = self._entities_share_category(source, target)
            
            # Compute inhibition (inverse of co-occurrence)
            if same_category:
                inhibition = alpha * (1.0 - co_occur_rate) + (1.0 - alpha) * baseline_same
            else:
                inhibition = beta * (1.0 - co_occur_rate)
            
            # Set symmetric inhibition
            I_entity[idx_i, idx_j] = inhibition
            I_entity[idx_j, idx_i] = inhibition
            
            edge_count += 1
            
            if edge_count % 50000 == 0:
                print(f"  Processed {edge_count} edges...")
        
        print(f"Built entity inhibition from {edge_count} co-occurrence edges")
        return I_entity

    def _entities_share_category(self, ent_i: str, ent_j: str) -> bool:
        cats_i = set()
        for succ in self.graph.graph.successors(ent_i):
            if succ in self.graph.category_nodes:
                cats_i.add(succ)
        
        cats_j = set()
        for succ in self.graph.graph.successors(ent_j):
            if succ in self.graph.category_nodes:
                cats_j.add(succ)
        
        return len(cats_i & cats_j) > 0
    
    def _compute_external_input(self, t: float) -> np.ndarray:
        tau_rise = 0.1
        tau_decay = 2.0
        rise_factor = 1.0 - np.exp(-t / tau_rise)
        decay_factor = np.exp(-t / tau_decay)
        modulation = rise_factor * decay_factor
        return self.state.s * modulation
    
    def _compute_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Compute dx/dt using biologically-realistic dynamics.
        
        Includes:
        - Excitation (weighted inputs with synaptic depression)
        - Inhibition (competitive + divisive normalization)
        - Decay (leak currents)
        - Adaptation (spike-rate adaptation)
        - External input
        """
        n = self.graph.num_nodes
        dx = np.zeros(n, dtype=np.float32)
        
        s_t = self._compute_external_input(t)
        I_combined = self.graph.inhibition_matrix + self.entity_inhibition
        
        # ✅ Compute total activity for divisive normalization
        total_activity = np.sum(x)
        
        for i in range(n):
            # 1. EXCITATION (with synaptic depression)
            excitation = 0.0
            for j in self.graph.get_neighbors(i):
                w_baseline = self.graph.get_baseline_weight(j, i)
                w_eff = self.state.get_effective_weight(j, i, w_baseline)
                
                # ✅ Apply synaptic depression (depleted synapses are weaker)
                w_depressed = w_eff * self.synaptic_resources[j]
                excitation += w_depressed * x[j]
            
            # 2. INHIBITION (competitive + divisive)
            # Competitive inhibition (local, from specific inhibitory connections)
            competitive_inhibition = np.dot(I_combined[i, :], x)
            
            # ✅ Divisive normalization (global activity suppression)
            # When total network activity is high, suppress all neurons
            divisive_inhibition = self.divisive_strength * x[i] * total_activity / (1.0 + total_activity)
            
            inhibition = competitive_inhibition + divisive_inhibition
            
            # 3. DECAY (leak current)
            decay = self.params.lambda_decay * x[i]
            
            # 4. ADAPTATION (spike-rate adaptation)
            # ✅ This builds up with sustained activity and suppresses further firing
            adaptation_current = self.gamma_adapt * self.adaptation[i]
            
            # 5. EXTERNAL INPUT
            external = s_t[i]
            
            # 6. COMBINED DYNAMICS
            dx_i = excitation - inhibition - decay - adaptation_current + external
            
            # Clip to prevent numerical overflow
            dx_i = np.clip(dx_i, -100.0, 100.0)
            
            dx[i] = dx_i
        
        return dx
    
    def _update_biological_variables(self, dt: float, x: np.ndarray):
        """
        Update slow biological variables (adaptation and synaptic resources).
        These evolve on slower timescales than the main dynamics.
        """
        # ✅ Update adaptation (builds up with activity, decays slowly)
        # d(adaptation)/dt = (x - adaptation) / tau_adapt
        d_adaptation = (x - self.adaptation) / self.tau_adapt
        self.adaptation += d_adaptation * dt
        self.adaptation = np.maximum(self.adaptation, 0.0)  # Non-negative
        
        # ✅ Update synaptic resources (deplete when active, recover when inactive)
        # d(resources)/dt = (1 - resources) / tau_recovery - depletion_rate * x * resources
        recovery = (1.0 - self.synaptic_resources) / self.tau_recovery
        depletion = self.depletion_rate * x * self.synaptic_resources
        
        d_resources = recovery - depletion
        self.synaptic_resources += d_resources * dt
        self.synaptic_resources = np.clip(self.synaptic_resources, 0.1, 1.0)  # Never fully depleted or over-saturated
    
    def _update_hebbian_from_trajectory(self, t_samples: np.ndarray, x_samples: np.ndarray):
        eta = self.params.eta
        threshold = self.params.hebbian_threshold
        
        for i in range(len(t_samples) - 1):
            dt_i = t_samples[i+1] - t_samples[i]
            x_avg = 0.5 * (x_samples[:, i] + x_samples[:, i+1])
            
            for idx_i in range(self.graph.num_nodes):
                if x_avg[idx_i] <= threshold:
                    continue
                
                neighbors = self.graph.get_neighbors(idx_i)
                for idx_j in neighbors:
                    if x_avg[idx_j] <= threshold:
                        continue
                    
                    delta_increment = eta * x_avg[idx_i] * x_avg[idx_j] * dt_i
                    edge = (idx_j, idx_i)
                    current_delta = self.state.delta_w.get(edge, 0.0)
                    self.state.delta_w[edge] = current_delta + delta_increment
    
    def _check_windowed_convergence(self, t: float, x_current: np.ndarray, dx_current: np.ndarray) -> bool:
        if t < self.t_min:
            return False
        
        # Compute metrics
        max_derivative = np.max(np.abs(dx_current))
        
        if self.x_prev_macro is not None:
            state_change = np.linalg.norm(x_current - self.x_prev_macro)
            x_norm = np.linalg.norm(x_current)
            relative_change = state_change / (x_norm + 1e-10)
        else:
            relative_change = np.inf
        
        # Store in history
        self.derivative_history.append(max_derivative)
        self.state_change_history.append(relative_change)
        
        # Update previous state
        self.x_prev_macro = x_current.copy()
        
        # Check if window full
        if len(self.derivative_history) < self.W:
            return False
        
        # Check convergence criteria
        deriv_converged = all(d < self.eps_abs for d in self.derivative_history)
        state_converged = all(s < self.eps_rel for s in self.state_change_history)
        window_converged = deriv_converged and state_converged
        
        if window_converged:
            self.consecutive_passes += 1
            if self.consecutive_passes >= self.P:
                return True
        else:
            self.consecutive_passes = 0
        
        return False
    
    def run_simulation(self, macrostep_size: float = 0.1) -> Dict:
        print(f"\n{'='*60}")
        print(f"RUNNING SIMULATION (WITH BIOLOGICAL MECHANISMS)")
        print(f"{'='*60}")
        print(f"Parameters: {self.params.num_steps()} total microsteps")
        print(f"Macrostep size: {macrostep_size}s")
        print(f"Biological features: adaptation, synaptic depression, divisive normalization")
        
        t = 0.0
        converged = False
        macrostep_count = 0
        start_time = time.time()
        
        while t < self.params.T and not converged:
            macrostep_count += 1
            t_next = min(t + macrostep_size, self.params.T)
            
            print(f"\nMacrostep {macrostep_count}: t={t:.2f}s → {t_next:.2f}s")
            
            # Deterministic evolution (RK45)
            result = solve_ivp(
                fun=self._compute_derivative,
                t_span=(t, t_next),
                y0=self.state.x,
                method='RK45',
                dense_output=True,
                rtol=1e-3,
                atol=1e-6
            )
            
            x_det = result.y[:, -1]
            
            # Stochastic increment
            dt_macro = t_next - t
            noise = self.params.sigma * np.sqrt(dt_macro) * np.random.randn(self.graph.num_nodes)
            x_new = x_det + noise
            x_new = np.maximum(x_new, 0.0)
            x_new = np.minimum(x_new, self.params.x_max)
            
            # ✅ Update biological variables
            self._update_biological_variables(dt_macro, x_new)
            
            # Hebbian update
            t_samples = np.linspace(t, t_next, 20)
            x_samples = result.sol(t_samples)
            self._update_hebbian_from_trajectory(t_samples, x_samples)
            
            # Temporary weight decay
            if self.params.use_weight_decay:
                decay_factor = 1.0 - self.params.lambda_w * dt_macro
                for edge in list(self.state.delta_w.keys()):
                    self.state.delta_w[edge] *= decay_factor
                    if abs(self.state.delta_w[edge]) < 1e-6:
                        del self.state.delta_w[edge]
            
            # Update state
            self.state.x = x_new
            self.state.t = t_next
            self.state.step_count += 1
            
            self.trajectory_times.append(t_next)
            self.trajectory_states.append(x_new.copy())
            
            # Windowed convergence check
            dx_current = self._compute_derivative(t_next, x_new)
            converged = self._check_windowed_convergence(t_next, x_new, dx_current)
            
            if converged:
                print(f"✓ Converged at t={t_next:.2f}s (W={self.W}, P={self.P})")
            
            validate_activations(self.state.x, self.graph.all_nodes)
            
            stats = self.state.get_stats()
            
            # ✅ Add biological stats
            avg_adaptation = np.mean(self.adaptation[self.adaptation > 0.01])
            avg_resources = np.mean(self.synaptic_resources)
            
            print(f"  Active nodes: {stats['active_nodes']}/{self.graph.num_nodes}")
            print(f"  Max activation: {stats['max_activation']:.3f}")
            print(f"  Hebbian edges: {stats['hebbian_edges']}")
            print(f"  Avg adaptation: {avg_adaptation:.3f}")
            print(f"  Avg synaptic resources: {avg_resources:.3f}")
            
            t = t_next
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Final time: {t:.2f}s")
        print(f"Converged: {converged}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"{'='*60}\n")
        
        return {
            'final_time': t,
            'converged': converged,
            'macrosteps': macrostep_count,
            'elapsed_time': elapsed,
            'final_state': self.state.x.copy(),
            'final_adaptation': self.adaptation.copy(),
            'final_resources': self.synaptic_resources.copy(),
            'trajectory_times': np.array(self.trajectory_times),
            'trajectory_states': np.array(self.trajectory_states).T,
            'final_stats': self.state.get_stats()
        }
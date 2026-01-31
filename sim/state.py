"""
state.py

Simulation state initialization (per simulation):
- Activation vector x(t)
- Temporary Hebbian weights Δw
- External input s(t)
"""

import numpy as np
from typing import Dict, Tuple, Optional

# Import from local sim/config.py, not kg/config.py
from sim_config import SimulationParameters
from sim_utils import validate_activations


class SimulationState:
    """Manages temporary state for a single simulation run."""
    
    def __init__(self, num_nodes: int, params: Optional[SimulationParameters] = None):
        """
        Initialize simulation state.
        
        Args:
            num_nodes: Total number of nodes in graph
            params: Simulation parameters (uses defaults if None)
        """
        self.num_nodes = num_nodes
        self.params = params if params is not None else SimulationParameters()
        
        # Initialize activation vector (all zeros)
        self.x = np.zeros(num_nodes, dtype=np.float32)
        
        # Initialize temporary Hebbian weights (sparse dict)
        # Key: (source_idx, target_idx), Value: Δw_ij
        self.delta_w: Dict[Tuple[int, int], float] = {}
        
        # External input vector (set during injection)
        self.s = np.zeros(num_nodes, dtype=np.float32)
        
        # Track previous activation for convergence checking
        self.x_prev = np.zeros(num_nodes, dtype=np.float32)
        
        # Simulation time
        self.t = 0.0
        self.step_count = 0
        
        # Convergence flag
        self.has_converged = False
        
        print(f"Initialized simulation state: {num_nodes} nodes, {self.params.num_steps()} steps")
    
    def reset(self):
        """Reset state for a new simulation (reuse object)."""
        self.x.fill(0.0)
        self.delta_w.clear()
        self.s.fill(0.0)
        self.x_prev.fill(0.0)
        self.t = 0.0
        self.step_count = 0
        self.has_converged = False
    
    def set_external_input(self, node_indices: list, activation_strength: float = 1.0):
        """
        Set external input for specific nodes.
        
        Args:
            node_indices: List of node indices to activate
            activation_strength: Strength of input (0.5-1.0 typical)
        """
        for idx in node_indices:
            if 0 <= idx < self.num_nodes:
                self.s[idx] = activation_strength
            else:
                raise ValueError(f"Node index {idx} out of bounds [0, {self.num_nodes})")
        
        print(f"Set external input for {len(node_indices)} nodes (strength={activation_strength})")
    
    def get_effective_weight(self, source_idx: int, target_idx: int, 
                            baseline_weight: float) -> float:
        """
        Compute effective weight: w_ij^eff = w_baseline + Δw.
        
        Args:
            source_idx: Source node index
            target_idx: Target node index
            baseline_weight: Baseline weight from frozen graph
            
        Returns:
            Effective weight
        """
        delta = self.delta_w.get((source_idx, target_idx), 0.0)
        return baseline_weight + delta
    
    def update_hebbian_weight(self, source_idx: int, target_idx: int):
        """
        Update temporary Hebbian weight for an edge.
        
        Δw_ij(t+dt) = Δw_ij(t) + η * x_i(t) * x_j(t) * dt
        
        Only updates if both nodes are above threshold.
        """
        x_i = self.x[source_idx]
        x_j = self.x[target_idx]
        
        # Check threshold
        if x_i > self.params.hebbian_threshold and x_j > self.params.hebbian_threshold:
            # Compute Hebbian update
            delta_increment = self.params.eta * x_i * x_j * self.params.dt
            
            # Get current Δw (default 0.0)
            current_delta = self.delta_w.get((source_idx, target_idx), 0.0)
            
            # Update with cap to prevent explosion
            new_delta = current_delta + delta_increment
            new_delta = min(new_delta, 1.0)  # Cap at 1.0
            
            self.delta_w[(source_idx, target_idx)] = new_delta
    
    def apply_weight_decay(self):
        """
        Apply decay to temporary Hebbian weights (optional).
        
        Δw_ij(t+dt) = Δw_ij(t) - λ_w * Δw_ij(t) * dt
        """
        if not self.params.use_weight_decay:
            return
        
        decay_factor = 1.0 - self.params.lambda_w * self.params.dt
        
        # Decay all Hebbian weights
        for edge in list(self.delta_w.keys()):
            self.delta_w[edge] *= decay_factor
            
            # Remove very small weights (cleanup)
            if abs(self.delta_w[edge]) < 1e-6:
                del self.delta_w[edge]
    
    def update_activation(self, node_idx: int, dx_dt: float):
        """
        Update activation for a single node.
        
        x_i(t+dt) = x_i(t) + dx_dt * dt
        
        Enforces non-negativity and maximum constraints.
        """
        new_x = self.x[node_idx] + dx_dt * self.params.dt
        
        # Enforce constraints
        new_x = max(0.0, new_x)  # Non-negativity
        new_x = min(new_x, self.params.x_max)  # Maximum cap
        
        self.x[node_idx] = new_x
    
    def check_convergence(self) -> bool:
        """
        Check if activations have converged.
        
        Returns:
            True if ||x(t) - x(t-1)|| < epsilon
        """
        if self.step_count % self.params.check_convergence_every != 0:
            return False
        
        diff = np.linalg.norm(self.x - self.x_prev)
        
        if diff < self.params.convergence_epsilon:
            self.has_converged = True
            print(f"Converged at t={self.t:.2f}s (step {self.step_count})")
            return True
        
        # Update previous activation
        np.copyto(self.x_prev, self.x)
        return False
    
    def validate(self, node_names: list = None):
        """Validate state for numerical issues."""
        validate_activations(self.x, node_names)
    
    def step(self):
        """Advance simulation by one timestep."""
        self.t += self.params.dt
        self.step_count += 1
    
    def get_stats(self) -> dict:
        """Get statistics about current state."""
        active_nodes = np.sum(self.x > 0.01)
        max_activation = np.max(self.x)
        mean_activation = np.mean(self.x[self.x > 0]) if active_nodes > 0 else 0.0
        num_hebbian_edges = len(self.delta_w)
        
        return {
            'time': self.t,
            'step': self.step_count,
            'active_nodes': active_nodes,
            'max_activation': max_activation,
            'mean_activation': mean_activation,
            'hebbian_edges': num_hebbian_edges,
            'converged': self.has_converged
        }
    
    def cleanup(self):
        """Discard all temporary state."""
        self.x.fill(0.0)
        self.delta_w.clear()
        self.s.fill(0.0)
        self.x_prev.fill(0.0)
        self.t = 0.0
        self.step_count = 0
        self.has_converged = False
        
        print("State cleaned up - ready for next simulation")   
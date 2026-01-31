"""
config.py

Simulation parameters and configuration constants.
"""

from dataclasses import dataclass


@dataclass
class SimulationParameters:
    """Parameters for simulation dynamics."""
    
    # Hebbian learning rate
    eta: float = 0.03  # η - strength of temporary connection strengthening
    
    # Activation decay rate
    lambda_decay: float = 0.5  # λ - how fast activation fades
    
    # Temporary weight decay rate (optional)
    lambda_w: float = 0.05  # λ_w - how fast Hebbian boosts fade
    use_weight_decay: bool = True  # Toggle weight decay on/off
    
    # Stochastic noise level
    sigma: float = 0.01  # σ - Gaussian noise standard deviation
    
    # Time parameters
    T: float = 5.0  # Total simulation duration (seconds)
    dt: float = 0.01  # Timestep size (seconds)
    
    # Hebbian update threshold
    hebbian_threshold: float = 0.1  # Only update if both nodes > threshold
    
    # Activation constraints
    x_max: float = 10.0  # Maximum activation (prevent runaway)
    
    # Convergence checking
    convergence_epsilon: float = 1e-4  # ||x(t) - x(t-1)|| threshold
    check_convergence_every: int = 10  # Check every N steps
    
    def num_steps(self) -> int:
        """Calculate total number of timesteps."""
        return int(self.T / self.dt)


# Default parameter presets
DEFAULT_PARAMS = SimulationParameters()

FAST_PARAMS = SimulationParameters(
    T=2.0,
    dt=0.02,
    eta=0.05
)

PRECISE_PARAMS = SimulationParameters(
    T=10.0,
    dt=0.005,
    convergence_epsilon=1e-5
)


# Other constants
ENTITY_PREFIX = "entity_"
USER_NODE_TYPE = "user"
CATEGORY_NODE_TYPE = "category"
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
from .base import BaseController

class MPPIController(BaseController):
    """Model Predictive Path Integral (MPPI) controller implementation."""
    
    def __init__(
        self,
        dt: float = 1.0/240.0,
        horizon: int = 20,
        num_samples: int = 1000,
        lambda_: float = 1.0,
        sigma: float = 0.5,
        target_position: Optional[torch.Tensor] = None,
        motor_limits: Optional[Tuple[float, float]] = None,
        model_path: Optional[str] = None
    ):
        """Initialize the MPPI controller.
        
        Args:
            dt: Time step
            horizon: Prediction horizon
            num_samples: Number of samples for MPPI
            lambda_: Temperature parameter for MPPI
            sigma: Standard deviation for action sampling
            target_position: Target position [x, y, z]
            motor_limits: Motor thrust limits (min, max)
            model_path: Path to the dynamics model (optional)
        """
        self.dt = dt
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_ = lambda_
        self.sigma = sigma
        
        # Set default values if not provided
        if target_position is None:
            target_position = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            
        if motor_limits is None:
            motor_limits = (0.0, 1.0)
            
        self.target_position = target_position
        self.motor_limits = motor_limits
        
        # Load or create dynamics model
        self.dynamics_model = self._load_or_create_model(model_path)
        
        # Initialize state
        self.reset()
        
    def _load_or_create_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create a dynamics model.
        
        Args:
            model_path: Path to the model file (optional)
            
        Returns:
            Dynamics model
        """
        # This is a placeholder - in a real implementation, you would load or create the actual dynamics model
        # For now, we'll create a simple neural network as a placeholder
        model = nn.Sequential(
            nn.Linear(16, 64),  # Input: state(13) + action(4) - 1 (time is implicit)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 13)  # Output: next state (13)
        )
        
        # Load weights if available
        if model_path is not None:
            try:
                model.load_state_dict(torch.load(model_path))
                print(f"Loaded dynamics model from {model_path}")
            except:
                print(f"Could not load model from {model_path}, using initialized weights")
                
        return model
    
    def reset(self) -> None:
        """Reset controller state."""
        # Initialize mean action sequence
        self.mean_actions = torch.zeros((self.horizon, 4))
        
    def _sample_actions(self) -> torch.Tensor:
        """Sample action sequences using Gaussian noise.
        
        Returns:
            Sampled action sequences
        """
        # Generate noise
        noise = torch.randn(self.num_samples, self.horizon, 4) * self.sigma
        
        # Add noise to mean actions
        actions = self.mean_actions + noise
        
        # Clip actions to valid range
        actions = torch.clamp(actions, self.motor_limits[0], self.motor_limits[1])
        
        return actions
    
    def _predict_trajectory(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Predict trajectory using the dynamics model.
        
        Args:
            initial_state: Initial state
            actions: Sequence of actions
            
        Returns:
            Predicted trajectory
        """
        # Initialize trajectory with initial state
        trajectory = [initial_state]
        current_state = initial_state
        
        # Predict trajectory
        for action in actions:
            # Prepare input for the model
            model_input = torch.cat([current_state, action])
            
            # Predict next state
            next_state = self.dynamics_model(model_input)
            
            # Add to trajectory
            trajectory.append(next_state)
            current_state = next_state
            
        return torch.stack(trajectory)
    
    def _compute_cost(
        self,
        trajectory: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute cost for a trajectory.
        
        Args:
            trajectory: Predicted trajectory
            actions: Sequence of actions
            
        Returns:
            Total cost
        """
        # Position error cost
        position_errors = torch.norm(trajectory[:, :3] - self.target_position, dim=1)
        position_cost = torch.sum(position_errors)
        
        # Orientation cost (penalize non-zero roll and pitch)
        orientation_cost = torch.sum(torch.abs(trajectory[:, 3:5]))
        
        # Velocity cost (penalize high velocities)
        velocity_cost = torch.sum(torch.norm(trajectory[:, 7:10], dim=1))
        
        # Angular velocity cost
        angular_velocity_cost = torch.sum(torch.norm(trajectory[:, 10:13], dim=1))
        
        # Action cost (penalize large actions)
        action_cost = torch.sum(torch.norm(actions, dim=1))
        
        # Total cost
        total_cost = (
            position_cost +
            0.1 * orientation_cost +
            0.01 * velocity_cost +
            0.01 * angular_velocity_cost +
            0.001 * action_cost
        )
        
        return total_cost
    
    def _update_mean_actions(
        self,
        sampled_actions: torch.Tensor,
        costs: torch.Tensor
    ) -> None:
        """Update mean actions using MPPI update rule.
        
        Args:
            sampled_actions: Sampled action sequences
            costs: Costs for each sampled sequence
        """
        # Compute weights
        weights = torch.exp(-1.0 / self.lambda_ * (costs - torch.min(costs)))
        weights = weights / torch.sum(weights)
        
        # Update mean actions
        self.mean_actions = torch.sum(
            weights.view(-1, 1, 1) * sampled_actions,
            dim=0
        )
        
    def compute_control(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute control action based on current observation.
        
        Args:
            observation: Dictionary containing position, orientation, linear velocity, and angular velocity
            
        Returns:
            Control action as motor commands
        """
        # Extract state
        pos = observation["position"]
        quat = observation["orientation"]
        lin_vel = observation["linear_velocity"]
        ang_vel = observation["angular_velocity"]
        
        # Prepare initial state
        initial_state = torch.cat([pos, quat, lin_vel, ang_vel])
        
        # Sample action sequences
        sampled_actions = self._sample_actions()
        
        # Compute costs for each sample
        costs = torch.zeros(self.num_samples)
        for i in range(self.num_samples):
            trajectory = self._predict_trajectory(initial_state, sampled_actions[i])
            costs[i] = self._compute_cost(trajectory, sampled_actions[i])
        
        # Update mean actions
        self._update_mean_actions(sampled_actions, costs)
        
        # Return first action
        return self.mean_actions[0] 
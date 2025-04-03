import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
from .base import BaseController

class MPCController(BaseController):
    """Model Predictive Control (MPC) controller implementation."""
    
    def __init__(
        self,
        dt: float = 1.0/240.0,
        horizon: int = 10,
        target_position: Optional[torch.Tensor] = None,
        motor_limits: Optional[Tuple[float, float]] = None,
        model_path: Optional[str] = None
    ):
        """Initialize the MPC controller.
        
        Args:
            dt: Time step
            horizon: Prediction horizon
            target_position: Target position [x, y, z]
            motor_limits: Motor thrust limits (min, max)
            model_path: Path to the dynamics model (optional)
        """
        self.dt = dt
        self.horizon = horizon
        
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
        # Initialize trajectory
        self.trajectory = []
        
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
    
    def _optimize_actions(
        self,
        initial_state: torch.Tensor
    ) -> torch.Tensor:
        """Optimize actions using gradient descent.
        
        Args:
            initial_state: Initial state
            
        Returns:
            Optimized actions
        """
        # Initialize actions
        actions = torch.zeros((self.horizon, 4), requires_grad=True)
        
        # Optimize actions
        optimizer = torch.optim.Adam([actions], lr=0.01)
        
        for _ in range(100):  # Number of optimization steps
            optimizer.zero_grad()
            
            # Predict trajectory
            trajectory = self._predict_trajectory(initial_state, actions)
            
            # Compute cost
            cost = self._compute_cost(trajectory, actions)
            
            # Backpropagate
            cost.backward()
            
            # Update actions
            optimizer.step()
            
            # Clip actions to valid range
            with torch.no_grad():
                actions.clamp_(self.motor_limits[0], self.motor_limits[1])
        
        return actions
    
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
        
        # Optimize actions
        actions = self._optimize_actions(initial_state)
        
        # Return first action
        return actions[0] 
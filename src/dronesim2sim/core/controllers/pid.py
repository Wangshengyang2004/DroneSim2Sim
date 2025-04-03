import torch
from typing import Dict, Optional, Tuple
from .base import BasePIDController

class PIDController(BasePIDController):
    """PID controller implementation."""
    
    def __init__(
        self,
        dt: float = 1.0/240.0,
        target_position: Optional[torch.Tensor] = None,
        motor_limits: Optional[Tuple[float, float]] = None,
        position_gains: Optional[Dict[str, float]] = None,
        attitude_gains: Optional[Dict[str, float]] = None
    ):
        """Initialize the PID controller.
        
        Args:
            dt: Time step
            target_position: Target position [x, y, z]
            motor_limits: Motor thrust limits (min, max)
            position_gains: Position control gains
            attitude_gains: Attitude control gains
        """
        # Convert position_gains to the format expected by BasePIDController
        position_gains_dict = {
            "x": position_gains or {"p": 0.5, "i": 0.05, "d": 0.2},
            "y": position_gains or {"p": 0.5, "i": 0.05, "d": 0.2},
            "z": position_gains or {"p": 0.5, "i": 0.05, "d": 0.2}
        }
        
        # Convert attitude_gains to the format expected by BasePIDController
        attitude_gains_dict = {
            "roll": attitude_gains or {"p": 5.0, "i": 0.5, "d": 2.0},
            "pitch": attitude_gains or {"p": 5.0, "i": 0.5, "d": 2.0},
            "yaw": attitude_gains or {"p": 5.0, "i": 0.5, "d": 2.0}
        }
        
        # Set default values if not provided
        if target_position is None:
            target_position = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            
        if motor_limits is None:
            motor_limits = (0.0, 1.0)
            
        # Call the parent constructor
        super().__init__(
            position_gains=position_gains_dict,
            attitude_gains=attitude_gains_dict,
            dt=dt,
            target_position=target_position,
            motor_limits=motor_limits
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
        
        # Compute position control
        desired_acceleration = self.compute_position_control(pos, lin_vel)
        
        # Compute attitude control
        motor_commands = self.compute_attitude_control(quat, ang_vel, desired_acceleration)
        
        return motor_commands
        
    def reset(self):
        """Reset controller state."""
        super().reset() 
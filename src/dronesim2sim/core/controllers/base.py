from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from ..utils import (
    quaternion_to_euler,
    compute_rotation_error,
    compute_position_error,
    compute_velocity_error,
    compute_attitude_error,
    euler_to_quaternion
)

class BaseController(ABC):
    """Base class for all controllers."""
    
    @abstractmethod
    def compute_control(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute control action from observation.
        
        Args:
            observation: Dictionary containing state information
            
        Returns:
            Control action as torch tensor
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset controller state."""
        pass

class BasePIDController(BaseController):
    """Base class for PID controllers with common functionality."""
    
    def __init__(
        self,
        position_gains: Dict[str, Dict[str, float]],
        attitude_gains: Dict[str, Dict[str, float]],
        dt: float,
        target_position: torch.Tensor,
        motor_limits: tuple[float, float],
    ):
        """Initialize PID controller.
        
        Args:
            position_gains: Dictionary of PID gains for position control
            attitude_gains: Dictionary of PID gains for attitude control
            dt: Time step
            target_position: Target position to maintain
            motor_limits: Tuple of (min, max) motor commands
        """
        self.position_gains = position_gains
        self.attitude_gains = attitude_gains
        self.dt = dt
        self.target_position = target_position
        self.motor_limits = motor_limits
        
        # Initialize error integrals and previous errors
        self.reset()
    
    def reset(self) -> None:
        """Reset controller state."""
        # Position control
        self.position_error_integral = torch.zeros(3, dtype=torch.float32)
        self.position_error_previous = torch.zeros(3, dtype=torch.float32)
        
        # Attitude control
        self.attitude_error_integral = torch.zeros(3, dtype=torch.float32)
        self.attitude_error_previous = torch.zeros(3, dtype=torch.float32)
        
        # Store control errors for analysis
        self.control_errors = []
    
    def compute_position_control(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute position control signal using PID."""
        # Compute position error
        position_error = compute_position_error(position, self.target_position)
        
        # Update integral
        self.position_error_integral = self.position_error_integral + position_error * self.dt
        
        # Compute derivative
        position_error_derivative = compute_velocity_error(
            position_error,
            self.position_error_previous
        ) / self.dt
        
        # Compute control signal
        control = torch.zeros(3, dtype=torch.float32)
        for i, axis in enumerate(['x', 'y', 'z']):
            gains = self.position_gains[axis]
            control[i] = (
                gains['p'] * position_error[i] +
                gains['i'] * self.position_error_integral[i] +
                gains['d'] * position_error_derivative[i]
            )
        
        # Update previous error
        self.position_error_previous = position_error.clone()
        
        # Store error for analysis
        self.control_errors.append(torch.norm(position_error).item())
        
        return control
    
    def compute_attitude_control(
        self,
        orientation: torch.Tensor,
        angular_velocity: torch.Tensor,
        desired_acceleration: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attitude control signal using PID."""
        # Convert quaternion to Euler angles
        euler = quaternion_to_euler(orientation)
        
        # Compute desired attitude from desired acceleration
        # This is a simplified model - in practice, you'd want a more sophisticated mapping
        desired_roll = torch.atan2(desired_acceleration[1], desired_acceleration[2])
        desired_pitch = torch.atan2(-desired_acceleration[0], torch.sqrt(
            desired_acceleration[1]**2 + desired_acceleration[2]**2
        ))
        desired_yaw = torch.tensor(0.0, dtype=torch.float32)  # Maintain current yaw
        
        desired_attitude = torch.stack([desired_roll, desired_pitch, desired_yaw])
        
        # Compute attitude error using Euler angles directly
        attitude_error = compute_attitude_error(euler, desired_attitude)
        
        # Update integral
        self.attitude_error_integral = self.attitude_error_integral + attitude_error * self.dt
        
        # Compute derivative
        attitude_error_derivative = compute_velocity_error(
            attitude_error,
            self.attitude_error_previous
        ) / self.dt
        
        # Compute control signal
        control = torch.zeros(3, dtype=torch.float32)
        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
            gains = self.attitude_gains[axis]
            control[i] = (
                gains['p'] * attitude_error[i] +
                gains['i'] * self.attitude_error_integral[i] +
                gains['d'] * attitude_error_derivative[i]
            )
        
        # Update previous error
        self.attitude_error_previous = attitude_error.clone()
        
        return control
    
    def compute_control(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute control action from observation."""
        # Extract state information
        position = observation["position"]
        orientation = observation["orientation"]
        velocity = observation["linear_velocity"]
        angular_velocity = observation["angular_velocity"]
        
        # Compute position control
        desired_acceleration = self.compute_position_control(position, velocity)
        
        # Compute attitude control
        attitude_control = self.compute_attitude_control(
            orientation,
            angular_velocity,
            desired_acceleration
        )
        
        # Convert to motor commands
        # This is a simplified model - in practice, you'd want a more sophisticated mapping
        motor_commands = torch.zeros(4, dtype=torch.float32)
        
        # Base thrust to counteract gravity
        base_thrust = 9.81 / 4.0  # Simplified gravity compensation
        
        # Map attitude control to motor commands
        motor_commands[0] = base_thrust + attitude_control[0] + attitude_control[1] + attitude_control[2]
        motor_commands[1] = base_thrust - attitude_control[0] + attitude_control[1] - attitude_control[2]
        motor_commands[2] = base_thrust + attitude_control[0] - attitude_control[1] - attitude_control[2]
        motor_commands[3] = base_thrust - attitude_control[0] - attitude_control[1] + attitude_control[2]
        
        # Clip to valid range
        motor_commands = torch.clamp(motor_commands, self.motor_limits[0], self.motor_limits[1])
        
        return motor_commands 
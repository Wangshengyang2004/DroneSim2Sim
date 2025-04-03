from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
from .utils import quaternion_to_rotation_matrix

class BaseDroneModel(ABC):
    """Base class for drone models."""
    
    def __init__(
        self,
        mass: float = 1.0,
        motor_thrust_max: float = 5.0,
        motor_torque_max: float = 0.2,
        arm_length: float = 0.15,
        inertia: Optional[np.ndarray] = None,
    ):
        """Initialize drone model.
        
        Args:
            mass: Mass of the drone in kg
            motor_thrust_max: Maximum thrust per motor in N
            motor_torque_max: Maximum torque per motor in Nm
            arm_length: Length of each arm in m
            inertia: Inertia matrix (3x3) in kg⋅m²
        """
        self.mass = mass
        self.motor_thrust_max = motor_thrust_max
        self.motor_torque_max = motor_torque_max
        self.arm_length = arm_length
        
        # Default inertia matrix if none provided
        if inertia is None:
            inertia = np.array([
                [0.1, 0, 0],
                [0, 0.1, 0],
                [0, 0, 0.2]
            ])
        self.inertia = inertia
        
        # Define propeller positions relative to center of mass
        self.propeller_positions = [
            [arm_length, 0, 0],    # Front
            [0, arm_length, 0],    # Right
            [-arm_length, 0, 0],   # Back
            [0, -arm_length, 0],   # Left
        ]
    
    @abstractmethod
    def compute_dynamics(
        self,
        state: Dict[str, np.ndarray],
        action: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute the dynamics of the drone.
        
        Args:
            state: Current state of the drone
            action: Control action (motor commands)
            
        Returns:
            Dictionary containing next state
        """
        pass
    
    def compute_thrust_mapping(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized actions to motor thrusts.
        
        Args:
            action: Normalized actions in [-1, 1]
            
        Returns:
            Motor thrusts in N
        """
        # Scale from [-1, 1] to [0, max_thrust]
        base_thrust = (self.mass * 9.81 / 4.0) * 1.1  # Add 10% margin for stability
        
        # Convert to physical units
        motor_thrusts = (action * 0.3 + 0.5) * self.motor_thrust_max
        
        # Add base thrust to counteract gravity
        motor_thrusts = motor_thrusts + base_thrust
        
        # Clip to valid range
        motor_thrusts = np.clip(motor_thrusts, 0.0, self.motor_thrust_max)
        
        return motor_thrusts
    
    def compute_force_torque(
        self,
        motor_thrusts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute total force and torque from motor thrusts.
        
        Args:
            motor_thrusts: Thrust of each motor in N
            
        Returns:
            Tuple of (total_force, total_torque)
        """
        # Initialize force and torque
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        
        # Add forces and torques from each motor
        for i in range(4):
            # Force in z-direction
            force = np.array([0, 0, motor_thrusts[i]])
            
            # Torque from thrust
            torque = np.cross(self.propeller_positions[i], force)
            
            # Add reaction torque (motor spin)
            # Motors 0 and 2 spin clockwise, 1 and 3 spin counter-clockwise
            torque_factor = -1.0 if i % 2 == 0 else 1.0
            reaction_torque = np.array([0, 0, torque_factor * motor_thrusts[i] * 0.02])
            
            total_force += force
            total_torque += torque + reaction_torque
        
        return total_force, total_torque
    
    def compute_linear_acceleration(
        self,
        force: np.ndarray,
        orientation: np.ndarray,
    ) -> np.ndarray:
        """Compute linear acceleration from force and orientation.
        
        Args:
            force: Total force in world frame
            orientation: Quaternion orientation
            
        Returns:
            Linear acceleration in world frame
        """
        # Transform force to world frame and compute acceleration
        rotation_matrix = quaternion_to_rotation_matrix(orientation)
        world_force = rotation_matrix @ force
        acceleration = world_force / self.mass
        
        return acceleration
    
    def compute_angular_acceleration(
        self,
        torque: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        """Compute angular acceleration from torque and angular velocity.
        
        Args:
            torque: Total torque in world frame
            angular_velocity: Current angular velocity
            
        Returns:
            Angular acceleration in world frame
        """
        # Compute angular acceleration using Euler's equations
        # τ = Iα + ω × (Iω)
        # α = I⁻¹(τ - ω × (Iω))
        inertia_term = np.cross(angular_velocity, self.inertia @ angular_velocity)
        angular_acceleration = np.linalg.inv(self.inertia) @ (torque - inertia_term)
        
        return angular_acceleration 
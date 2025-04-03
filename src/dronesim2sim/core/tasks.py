from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from .utils import quaternion_to_euler, compute_position_error

class BaseTask(ABC):
    """Base class for all tasks."""
    
    def __init__(self, name: str):
        """Initialize the task.
        
        Args:
            name: Name of the task
        """
        self.name = name
    
    @abstractmethod
    def compute_reward(self, observation: Dict[str, np.ndarray]) -> float:
        """Compute the reward based on the current observation."""
        pass
    
    @abstractmethod
    def compute_done(self, observation: Dict[str, np.ndarray]) -> bool:
        """Determine if the episode is done."""
        pass
    
    @abstractmethod
    def compute_info(self, observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute additional information."""
        pass
    
    @abstractmethod
    def get_target_state(self) -> Dict[str, np.ndarray]:
        """Get the target state for the task."""
        pass

class HoverTask(BaseTask):
    """Task for hovering at a specific position."""
    
    def __init__(
        self,
        target_position: np.ndarray = np.array([0.0, 0.0, 1.0]),
        position_tolerance: float = 0.1,
        orientation_tolerance: float = 0.1,
        velocity_tolerance: float = 0.5,
    ):
        """Initialize hover task.
        
        Args:
            target_position: Target position to hover at
            position_tolerance: Maximum allowed position error
            orientation_tolerance: Maximum allowed orientation error
            velocity_tolerance: Maximum allowed velocity
        """
        super().__init__("hover")
        self.target_position = target_position
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.velocity_tolerance = velocity_tolerance
    
    def compute_reward(self, observation: Dict[str, np.ndarray]) -> float:
        """Compute reward based on distance to target position."""
        position = observation["position"]
        position_error = np.linalg.norm(compute_position_error(position, self.target_position))
        
        # Penalize position error (higher reward when close to target)
        position_reward = np.exp(-2.0 * position_error)
        
        # Penalize tilting from upright
        quat = observation["orientation"]
        euler = quaternion_to_euler(quat)
        orientation_penalty = 0.05 * (abs(euler[0]) + abs(euler[1]))
        
        # Penalize velocity
        velocity_penalty = 0.02 * np.linalg.norm(observation["linear_velocity"])
        
        # Combine rewards
        reward = position_reward - orientation_penalty - velocity_penalty
        return float(reward)
    
    def compute_done(self, observation: Dict[str, np.ndarray]) -> bool:
        """Determine if episode is done."""
        # Episode terminates if drone crashes (below ground plane)
        if observation["position"][2] < 0.0:
            return True
        
        # Episode terminates if drone goes far out of bounds
        if np.any(np.abs(observation["position"]) > 10.0):
            return True
        
        return False
    
    def compute_info(self, observation: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute additional information."""
        position = observation["position"]
        position_error = np.linalg.norm(compute_position_error(position, self.target_position))
        return {"position_error": float(position_error)}
    
    def get_target_state(self) -> Dict[str, np.ndarray]:
        """Get target state for hovering."""
        return {
            "position": self.target_position,
            "orientation": np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            "linear_velocity": np.zeros(3),
            "angular_velocity": np.zeros(3),
        } 
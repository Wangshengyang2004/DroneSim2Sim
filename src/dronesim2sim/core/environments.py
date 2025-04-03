from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

class BaseDroneEnv(gym.Env):
    """Base class for drone environments."""
    
    def __init__(
        self,
        task_name: str = "hover",
        render_mode: str = None,
        seed: int = 0,
        freq: int = 240,
        gui: bool = False,
        record_video: bool = False,
    ):
        """Initialize the drone environment.
        
        Args:
            task_name: The name of the task (hover, trajectory, etc.)
            render_mode: The render mode (human, rgb_array, None)
            seed: Random seed
            freq: Control frequency (Hz)
            gui: Whether to use GUI (visualize simulation)
            record_video: Whether to record video
        """
        # Store parameters
        self.task_name = task_name
        self.render_mode = render_mode
        self.seed = seed
        self.freq = freq
        self.gui = gui
        self.record_video = record_video
        
        # Set time step
        self.dt = 1.0 / self.freq
        
        # Common drone properties
        self.drone_mass = 1.0  # kg
        self.motor_thrust_max = 5.0  # Newtons
        self.motor_torque_max = 0.2  # Nm
        self.g = 9.81  # gravity acceleration
        
        # Define action space (4 motor commands normalized between -1 and 1)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32),
            "orientation": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),  # Quaternion
            "linear_velocity": spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32),
            "angular_velocity": spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32),
        })
        
        # Video recording setup
        self.frame_count = 0
        self.video_writer = None
        
        # Initialize random number generator
        self.np_random = np.random.RandomState(seed)
    
    @abstractmethod
    def _connect_to_physics_server(self):
        """Connect to the physics server."""
        pass
    
    @abstractmethod
    def _load_drone(self):
        """Load the drone model into the simulation."""
        pass
    
    @abstractmethod
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        pass
    
    @abstractmethod
    def _compute_reward(self, obs: Dict[str, np.ndarray]) -> float:
        """Compute the reward based on the current observation."""
        pass
    
    @abstractmethod
    def _compute_done(self, obs: Dict[str, np.ndarray]) -> bool:
        """Determine if the episode is done."""
        pass
    
    @abstractmethod
    def _compute_info(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute additional information."""
        pass
    
    @abstractmethod
    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Process the action from normalized space to motor commands."""
        pass
    
    @abstractmethod
    def _apply_action(self, action: np.ndarray) -> None:
        """Apply the action to the environment."""
        pass
    
    @abstractmethod
    def _record_frame(self):
        """Record a video frame if enabled."""
        pass
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        # Set seed if provided
        if seed is not None:
            self.seed = seed
            self.np_random = np.random.RandomState(seed)
        
        # Reset physics simulation
        self._connect_to_physics_server()
        
        # Load drone model
        self._load_drone()
        
        # Add slight randomization for robustness
        pos_noise = self.np_random.uniform(-0.1, 0.1, size=3)
        pos_noise[2] = abs(pos_noise[2])  # Don't start below ground
        
        # Set initial position with small noise
        init_pos = np.array([0, 0, 0.2]) + pos_noise
        self._set_initial_state(init_pos)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._compute_info(obs)
        
        # Reset video recording
        self.frame_count = 0
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        return obs, info
    
    @abstractmethod
    def _set_initial_state(self, position: np.ndarray):
        """Set the initial state of the drone."""
        pass
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        self._step_simulation()
        
        # Optional sleep for real-time visualization
        if self.gui:
            time.sleep(self.dt)
        
        # Record frame if enabled
        self._record_frame()
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward, done flag, and info
        reward = self._compute_reward(obs)
        terminated = self._compute_done(obs)
        truncated = False  # We're not truncating episodes in this example
        info = self._compute_info(obs)
        
        return obs, reward, terminated, truncated, info
    
    @abstractmethod
    def _step_simulation(self):
        """Step the physics simulation."""
        pass
    
    @abstractmethod
    def render(self):
        """Render the environment."""
        pass
    
    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass 
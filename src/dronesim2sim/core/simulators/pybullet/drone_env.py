import os
import torch
import pybullet as p
import pybullet_data
import time
from typing import Dict, Optional, Tuple
import gymnasium as gym
import pkg_resources

from ...environments import BaseDroneEnv
from ...utils import quaternion_to_euler, compute_position_error, compute_velocity_error
from ...utils.simulator_urdf import PyBulletURDFLoader

class DroneEnv(BaseDroneEnv):
    """PyBullet implementation of the drone environment."""
    
    def __init__(self, task_name: str = "hover", render_mode: Optional[str] = None, 
                 seed: Optional[int] = None, freq: int = 240, gui: bool = False,
                 record_video: bool = False):
        """Initialize the PyBullet drone environment.
        
        Args:
            task_name: Name of the task to perform
            render_mode: Rendering mode
            seed: Random seed
            freq: Simulation frequency
            gui: Whether to show the PyBullet GUI
            record_video: Whether to record video
        """
        super().__init__(task_name, render_mode, seed, freq, gui, record_video)
        
        # Initialize connection state
        self.physics_client = None
        self.connected = False
        
        # Initialize PyBullet
        self._connect_to_physics_server()
        
        # Load drone model
        self._load_drone_model()
        
        # Setup video recording if enabled
        if self.record_video:
            self._setup_video_recording()
            
    def _connect_to_physics_server(self):
        """Connect to PyBullet physics server."""
        # Only connect if not already connected
        if not self.connected:
            if self.gui:
                try:
                    self.physics_client = p.connect(p.GUI)
                except p.error:
                    print("Warning: GUI connection failed, falling back to DIRECT mode")
                    self.physics_client = p.connect(p.DIRECT)
                    self.gui = False
            else:
                self.physics_client = p.connect(p.DIRECT)
                
            self.connected = True
            
            # Configure PyBullet
            if self.gui:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
                
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -self.g)
            p.setRealTimeSimulation(0)
            p.setTimeStep(self.dt)
            
    def _disconnect_from_physics_server(self):
        """Disconnect from PyBullet physics server."""
        if self.connected:
            p.disconnect(self.physics_client)
            self.connected = False
            self.physics_client = None
            
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment."""
        # Reset physics server
        if self.connected:
            p.resetSimulation()
        else:
            self._connect_to_physics_server()
            
        # Load drone model
        self._load_drone_model()
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
            
    def _load_drone_model(self):
        """Load the drone URDF model."""
        urdf_path = pkg_resources.resource_filename('dronesim2sim', 'models/quadrotor.urdf')
        self.urdf_loader = PyBulletURDFLoader(urdf_path)
        self.drone_id = self.urdf_loader.load(
            position=[0, 0, 1],
            orientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Get drone properties from URDF
        self.link_names = self.urdf_loader.get_link_names()
        self.joint_names = self.urdf_loader.get_joint_names()
        self.link_masses = self.urdf_loader.get_link_masses()
        self.link_inertias = self.urdf_loader.get_link_inertias()
        self.joint_limits = self.urdf_loader.get_joint_limits()
        self.joint_types = self.urdf_loader.get_joint_types()
        
    def _setup_video_recording(self):
        """Setup video recording."""
        if self.record_video:
            self.video_logger = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4,
                fileName=os.path.join("results", "video"),
                cameraDistance=3,
                cameraYaw=30,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0],
                width=1920,
                height=1080,
                fileNamePrefix="drone",
                fps=self.freq,
                frameInterval=1,
                flags=p.ER_USE_MATERIAL_COLORS_FROM_URDF
            )
            
    def _get_observation(self) -> Dict[str, torch.Tensor]:
        """Get the current observation.
        
        Returns:
            Dictionary containing position, orientation, linear velocity, and angular velocity
        """
        # Get base link state
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)
        
        # Convert quaternion to PyTorch tensor
        quat_tensor = torch.tensor(quat, dtype=torch.float32)
        euler = quaternion_to_euler(quat_tensor)
        
        # Convert to tensors
        pos_tensor = torch.tensor(pos, dtype=torch.float32)
        lin_vel_tensor = torch.tensor(lin_vel, dtype=torch.float32)
        ang_vel_tensor = torch.tensor(ang_vel, dtype=torch.float32)
        
        return {
            "position": pos_tensor,
            "orientation": quat_tensor,
            "linear_velocity": lin_vel_tensor,
            "angular_velocity": ang_vel_tensor
        }
        
    def _process_action(self, action: torch.Tensor) -> torch.Tensor:
        """Process the raw action into motor commands.
        
        Args:
            action: Raw action array
            
        Returns:
            Processed motor commands
        """
        # Convert action to motor commands (assuming action is already normalized)
        return action
        
    def _apply_action(self, action: torch.Tensor):
        """Apply the action to the drone.
        
        Args:
            action: Motor commands
        """
        # Apply forces to propellers
        for i, joint_name in enumerate(self.joint_names):
            if joint_name.startswith("prop"):
                # Convert normalized action to actual force
                force = action[i] * self.motor_thrust_max
                p.applyExternalForce(
                    self.drone_id,
                    i,  # Link index
                    forceObj=[0, 0, force],
                    posObj=[0, 0, 0],
                    flags=p.WORLD_FRAME
                )
                
    def _step_simulation(self):
        """Step the PyBullet simulation."""
        p.stepSimulation()
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            
    def close(self):
        """Close the environment."""
        if self.record_video:
            p.stopStateLogging(self.video_logger)
        self._disconnect_from_physics_server() 
import os
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple
import numpy as np
from .urdf import URDFLoader

class PyBulletURDFLoader(URDFLoader):
    """PyBullet-specific URDF loader implementation."""
    
    def __init__(self, urdf_path: str):
        """Initialize the URDF loader.
        
        Args:
            urdf_path: Path to the URDF file
        """
        super().__init__(urdf_path)
        self._robot_id = None
        
    def load(self, position: list = [0, 0, 0], orientation: list = [0, 0, 0, 1]) -> int:
        """Load the URDF file into PyBullet.
        
        Args:
            position: Initial position [x, y, z]
            orientation: Initial orientation as quaternion [x, y, z, w]
            
        Returns:
            Body ID of the loaded model
        """
        # Add data path to PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load URDF
        self._robot_id = p.loadURDF(
            self.model_path,  # Use model_path from base class
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        
        return self._robot_id
        
    def get_link_names(self) -> List[str]:
        """Get the names of all links in the URDF."""
        if self._robot_id is None:
            raise RuntimeError("URDF not loaded. Call load() first.")
        
        # Get base link name
        base_link_name = p.getBodyInfo(self._robot_id)[0].decode('utf-8')
        
        # Get joint link names
        joint_link_names = [p.getJointInfo(self._robot_id, i)[12].decode('utf-8') 
                           for i in range(p.getNumJoints(self._robot_id))]
        
        # Combine all link names
        return [base_link_name] + joint_link_names
        
    def get_joint_names(self) -> List[str]:
        """Get the names of all joints in the URDF."""
        if self._robot_id is None:
            raise RuntimeError("URDF not loaded. Call load() first.")
        return [p.getJointInfo(self._robot_id, i)[1].decode('utf-8') 
                for i in range(p.getNumJoints(self._robot_id))]
        
    def get_link_masses(self) -> Dict[str, float]:
        """Get the masses of all links."""
        if self._robot_id is None:
            raise RuntimeError("URDF not loaded. Call load() first.")
        
        masses = {}
        
        # Get base link mass
        base_link_name = p.getBodyInfo(self._robot_id)[0].decode('utf-8')
        base_mass = p.getDynamicsInfo(self._robot_id, -1)[0]
        masses[base_link_name] = base_mass
        
        # Get joint link masses
        for i in range(p.getNumJoints(self._robot_id)):
            joint_info = p.getJointInfo(self._robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            mass = p.getDynamicsInfo(self._robot_id, i)[0]
            masses[link_name] = mass
            
        return masses
        
    def get_link_inertias(self) -> Dict[str, np.ndarray]:
        """Get the inertia matrices of all links."""
        if self._robot_id is None:
            raise RuntimeError("URDF not loaded. Call load() first.")
        
        inertias = {}
        
        # Get base link inertia
        base_link_name = p.getBodyInfo(self._robot_id)[0].decode('utf-8')
        base_inertia = np.array(p.getDynamicsInfo(self._robot_id, -1)[2])
        inertias[base_link_name] = base_inertia
        
        # Get joint link inertias
        for i in range(p.getNumJoints(self._robot_id)):
            joint_info = p.getJointInfo(self._robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            inertia = np.array(p.getDynamicsInfo(self._robot_id, i)[2])
            inertias[link_name] = inertia
            
        return inertias
        
    def get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """Get the joint limits for all joints."""
        if self._robot_id is None:
            raise RuntimeError("URDF not loaded. Call load() first.")
        limits = {}
        for i in range(p.getNumJoints(self._robot_id)):
            info = p.getJointInfo(self._robot_id, i)
            if info[8] != -1:  # If joint has limits
                limits[info[1].decode('utf-8')] = (info[8], info[9])
        return limits
        
    def get_joint_types(self) -> Dict[str, str]:
        """Get the types of all joints."""
        if self._robot_id is None:
            raise RuntimeError("URDF not loaded. Call load() first.")
        types = {}
        for i in range(p.getNumJoints(self._robot_id)):
            info = p.getJointInfo(self._robot_id, i)
            types[info[1].decode('utf-8')] = info[2]
        return types 
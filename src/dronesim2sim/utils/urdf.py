import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class URDFLoader:
    """Base class for loading and managing URDF models."""
    
    def __init__(self, model_path: str):
        """Initialize the URDF loader.
        
        Args:
            model_path: Path to the URDF file
        """
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"URDF file not found: {model_path}")
            
    def get_model_path(self) -> str:
        """Get the path to the URDF file."""
        return self.model_path
        
    def get_model_name(self) -> str:
        """Get the name of the model from the URDF file."""
        # This is a placeholder - specific implementations will parse the URDF
        return os.path.splitext(os.path.basename(self.model_path))[0]
        
    def get_link_names(self) -> List[str]:
        """Get the names of all links in the URDF."""
        raise NotImplementedError
        
    def get_joint_names(self) -> List[str]:
        """Get the names of all joints in the URDF."""
        raise NotImplementedError
        
    def get_link_masses(self) -> Dict[str, float]:
        """Get the masses of all links."""
        raise NotImplementedError
        
    def get_link_inertias(self) -> Dict[str, np.ndarray]:
        """Get the inertia matrices of all links."""
        raise NotImplementedError
        
    def get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """Get the joint limits for all joints."""
        raise NotImplementedError
        
    def get_joint_types(self) -> Dict[str, str]:
        """Get the types of all joints."""
        raise NotImplementedError 
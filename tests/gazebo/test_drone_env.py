import pytest
import torch
from typing import Dict, Tuple

from dronesim2sim.core.simulators.gazebo.drone_env import DroneEnv

def test_gazebo_env_initialization():
    """Test initialization of the Gazebo drone environment."""
    # Initialize environment with default parameters
    env = DroneEnv(
        task_name="hover",
        render_mode=None,
        seed=0,
        freq=240,
        gui=False,
        record_video=False
    )
    
    # Check that the environment was initialized correctly
    assert env.task_name == "hover"
    assert env.render_mode is None
    assert env.seed == 0
    assert env.freq == 240
    assert env.gui is False
    assert env.record_video is False
    
    # Clean up
    env.close() 
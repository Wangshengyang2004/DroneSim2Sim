import pytest
import torch
from typing import Dict, Tuple

from dronesim2sim.core.simulators.pybullet.drone_env import DroneEnv

def test_drone_env_initialization():
    """Test initialization of the PyBullet drone environment."""
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

def test_reset_and_step():
    """Test reset and step functions of the PyBullet drone environment."""
    # Initialize environment
    env = DroneEnv(
        task_name="hover",
        render_mode=None,
        seed=0,
        freq=240,
        gui=False,
        record_video=False
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Check observation format
    assert isinstance(obs, dict)
    assert "position" in obs
    assert "orientation" in obs
    assert "linear_velocity" in obs
    assert "angular_velocity" in obs
    
    # Check observation types and shapes
    assert isinstance(obs["position"], torch.Tensor)
    assert obs["position"].shape == (3,)
    assert isinstance(obs["orientation"], torch.Tensor)
    assert obs["orientation"].shape == (4,)
    assert isinstance(obs["linear_velocity"], torch.Tensor)
    assert obs["linear_velocity"].shape == (3,)
    assert isinstance(obs["angular_velocity"], torch.Tensor)
    assert obs["angular_velocity"].shape == (3,)
    
    # Step environment with random action
    action = torch.rand(4, dtype=torch.float32)  # Random motor commands
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # Check step output
    assert isinstance(next_obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Clean up
    env.close()

def test_observation_and_action_spaces():
    """Test observation and action spaces of the PyBullet drone environment."""
    # Initialize environment
    env = DroneEnv(
        task_name="hover",
        render_mode=None,
        seed=0,
        freq=240,
        gui=False,
        record_video=False
    )
    
    # Check action space
    assert hasattr(env, 'action_space')
    assert env.action_space.shape == (4,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0
    
    # Check observation space
    assert hasattr(env, 'observation_space')
    assert isinstance(env.observation_space, dict) or hasattr(env.observation_space, 'spaces')
    
    # Clean up
    env.close()

def test_multiple_episodes():
    """Test running multiple episodes in the PyBullet drone environment."""
    # Initialize environment
    env = DroneEnv(
        task_name="hover",
        render_mode=None,
        seed=0,
        freq=240,
        gui=False,
        record_video=False
    )
    
    # Run 2 short episodes
    for episode in range(2):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        
        for step in range(10):  # Run 10 steps per episode
            action = torch.rand(4, dtype=torch.float32)  # Random motor commands
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
    
    # Clean up
    env.close()

def test_reward_and_done():
    """Test reward and done computation in the PyBullet drone environment."""
    # Initialize environment
    env = DroneEnv(
        task_name="hover",
        render_mode=None,
        seed=0,
        freq=240,
        gui=False,
        record_video=False
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Run a few steps to check reward and done
    for _ in range(5):
        action = torch.rand(4, dtype=torch.float32)  # Random motor commands
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Reward and done should be valid types
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    # Clean up
    env.close() 
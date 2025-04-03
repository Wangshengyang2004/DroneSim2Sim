import pytest
import os
import sys
import numpy as np
import torch
import jax
import jax.numpy as jnp

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dronesim2sim.core.tasks import BaseTask
from dronesim2sim.core.models import BaseDroneModel
from dronesim2sim.core.utils import quaternion_to_euler, euler_to_quaternion

def test_base_task_initialization():
    """Test if BaseTask can be initialized"""
    class DummyTask(BaseTask):
        def __init__(self):
            super().__init__()
            self.state_dim = 12
            self.action_dim = 4
            
        def reset(self):
            return np.zeros(self.state_dim), {}
            
        def step(self, action):
            return np.zeros(self.state_dim), 0.0, False, False, {}
    
    task = DummyTask()
    assert task.state_dim == 12
    assert task.action_dim == 4

def test_base_drone_model_initialization():
    """Test if BaseDroneModel can be initialized"""
    class DummyDrone(BaseDroneModel):
        def __init__(self):
            super().__init__()
            self.state_dim = 12
            self.action_dim = 4
            
        def reset(self):
            return np.zeros(self.state_dim)
            
        def step(self, action):
            return np.zeros(self.state_dim)
    
    drone = DummyDrone()
    assert drone.state_dim == 12
    assert drone.action_dim == 4

def test_math_utils():
    """Test mathematical utility functions"""
    # Test quaternion to euler conversion
    q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    euler = quaternion_to_euler(q)
    assert np.allclose(euler, np.zeros(3))
    
    # Test euler to quaternion conversion
    euler = np.zeros(3)
    q = euler_to_quaternion(euler)
    assert np.allclose(q, np.array([1.0, 0.0, 0.0, 0.0]))

def test_numpy_pid_controller():
    """Test if NumPy PID controller can be imported and initialized"""
    from dronesim2sim.pybullet.pid_controller import PIDController
    
    controller = PIDController(
        kp=np.array([1.0, 1.0, 1.0]),
        ki=np.array([0.1, 0.1, 0.1]),
        kd=np.array([0.01, 0.01, 0.01])
    )
    assert controller.kp.shape == (3,)
    assert controller.ki.shape == (3,)
    assert controller.kd.shape == (3,)

def test_torch_pid_controller():
    """Test if PyTorch PID controller can be imported and initialized"""
    from dronesim2sim.pybullet.torch_pid_controller import TorchPIDController
    
    controller = TorchPIDController(
        kp=torch.tensor([1.0, 1.0, 1.0]),
        ki=torch.tensor([0.1, 0.1, 0.1]),
        kd=torch.tensor([0.01, 0.01, 0.01])
    )
    assert controller.kp.shape == (3,)
    assert controller.ki.shape == (3,)
    assert controller.kd.shape == (3,)

def test_jax_pid_controller():
    """Test if JAX PID controller can be imported and initialized"""
    from dronesim2sim.pybullet.jax_pid_controller import JaxPIDController
    
    controller = JaxPIDController(
        kp=jnp.array([1.0, 1.0, 1.0]),
        ki=jnp.array([0.1, 0.1, 0.1]),
        kd=jnp.array([0.01, 0.01, 0.01])
    )
    assert controller.kp.shape == (3,)
    assert controller.ki.shape == (3,)
    assert controller.kd.shape == (3,) 
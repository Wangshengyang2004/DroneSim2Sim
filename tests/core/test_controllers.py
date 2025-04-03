import pytest
import torch
from typing import Dict

from dronesim2sim.core.controllers import (
    BaseController,
    BasePIDController,
    PIDController,
    SKRLController,
    MPCController,
    MPPIController
)

# Mock observation data for testing
@pytest.fixture
def mock_observation() -> Dict[str, torch.Tensor]:
    """Create a mock observation for testing controllers."""
    return {
        "position": torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32),
        "orientation": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),  # Identity quaternion
        "linear_velocity": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        "angular_velocity": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    }

def test_pid_controller(mock_observation):
    """Test the PID controller."""
    # Initialize PID controller
    controller = PIDController(
        dt=0.01,
        target_position=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        motor_limits=(0.0, 1.0)
    )
    
    # Compute control
    action = controller.compute_control(mock_observation)
    
    # The controller should output a 4-element tensor for a quadcopter
    assert isinstance(action, torch.Tensor)
    assert action.shape == (4,)
    assert torch.all(action >= 0.0) and torch.all(action <= 1.0)
    
    # Reset controller
    controller.reset()
    
    # Compute control again after reset
    action_after_reset = controller.compute_control(mock_observation)
    assert isinstance(action_after_reset, torch.Tensor)
    assert action_after_reset.shape == (4,)

def test_skrl_controller(mock_observation):
    """Test the SKRL controller."""
    # Initialize SKRL controller with a placeholder model
    controller = SKRLController(
        dt=0.01,
        target_position=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        motor_limits=(0.0, 1.0)
    )
    
    # Compute control
    action = controller.compute_control(mock_observation)
    
    # The controller should output a 4-element tensor for a quadcopter
    assert isinstance(action, torch.Tensor)
    assert action.shape == (4,)
    assert torch.all(action >= 0.0) and torch.all(action <= 1.0)
    
    # Reset controller
    controller.reset()

def test_mpc_controller(mock_observation):
    """Test the MPC controller."""
    # Initialize MPC controller
    controller = MPCController(
        dt=0.01,
        horizon=5,  # Short horizon for testing
        target_position=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        motor_limits=(0.0, 1.0)
    )
    
    # Compute control
    action = controller.compute_control(mock_observation)
    
    # The controller should output a 4-element tensor for a quadcopter
    assert isinstance(action, torch.Tensor)
    assert action.shape == (4,)
    assert torch.all(action >= 0.0) and torch.all(action <= 1.0)
    
    # Reset controller
    controller.reset()

def test_mppi_controller(mock_observation):
    """Test the MPPI controller."""
    # Initialize MPPI controller
    controller = MPPIController(
        dt=0.01,
        horizon=5,  # Short horizon for testing
        num_samples=10,  # Few samples for testing
        target_position=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        motor_limits=(0.0, 1.0)
    )
    
    # Compute control
    action = controller.compute_control(mock_observation)
    
    # The controller should output a 4-element tensor for a quadcopter
    assert isinstance(action, torch.Tensor)
    assert action.shape == (4,)
    assert torch.all(action >= 0.0) and torch.all(action <= 1.0)
    
    # Reset controller
    controller.reset() 
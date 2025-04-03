import pytest
import torch
import jax
import jax.numpy as jnp

from dronesim2sim.utils import (
    quaternion_to_euler,
    euler_to_quaternion,
    compute_rotation_error,
    compute_position_error,
    compute_velocity_error,
    compute_attitude_error
)

def test_quaternion_conversion():
    """Test quaternion to euler conversion and back."""
    # Test with identity quaternion
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    euler = quaternion_to_euler(identity_quat)
    assert torch.allclose(euler, torch.zeros(3, dtype=torch.float32), atol=1e-6)
    
    # Test conversion back to quaternion
    quat = euler_to_quaternion(euler)
    assert torch.allclose(quat, identity_quat, atol=1e-6)
    
    # Test with 90 degree rotation around X
    x90_quat = torch.tensor([0.7071, 0.7071, 0.0, 0.0], dtype=torch.float32)
    euler = quaternion_to_euler(x90_quat)
    expected = torch.tensor([1.5708, 0.0, 0.0], dtype=torch.float32)  # π/2 around X
    assert torch.allclose(euler, expected, atol=1e-4)

def test_error_computation():
    """Test error computation functions."""
    # Position error
    pos1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    pos2 = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    pos_error = compute_position_error(pos1, pos2)
    expected_pos_error = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    assert torch.allclose(pos_error, expected_pos_error, atol=1e-6)
    
    # Velocity error
    vel1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    vel2 = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    vel_error = compute_velocity_error(vel1, vel2)
    expected_vel_error = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    assert torch.allclose(vel_error, expected_vel_error, atol=1e-6)
    
    # Attitude error
    att1 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)  # Roll, pitch, yaw
    att2 = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32)
    att_error = compute_attitude_error(att1, att2)
    expected_att_error = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
    assert torch.allclose(att_error, expected_att_error, atol=1e-6)
    
    # Rotation error
    quat1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    quat2 = euler_to_quaternion(torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32))
    rot_error = compute_rotation_error(quat1, quat2)
    # The rotation error should be non-zero for different quaternions
    assert torch.norm(rot_error) > 0.0 
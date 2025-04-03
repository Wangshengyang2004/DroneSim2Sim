"""PyTest configuration file for DroneSim2Sim."""

import os
import sys
import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp

# Add project root to Python path to allow for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Define fixtures that can be used by all tests

@pytest.fixture
def temp_directory(tmpdir):
    """Create a temporary directory for test outputs."""
    return tmpdir

@pytest.fixture
def mock_state():
    """Create a mock drone state dictionary."""
    import torch
    return {
        "position": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        "orientation": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),  # Identity quaternion
        "linear_velocity": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        "angular_velocity": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    }

@pytest.fixture
def mock_action():
    """Create a mock drone action tensor."""
    import torch
    return torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)  # Mid-range motor values

@pytest.fixture
def numpy_array():
    """Fixture for NumPy array operations"""
    return np.array([1.0, 2.0, 3.0])

@pytest.fixture
def torch_tensor():
    """Fixture for PyTorch tensor operations"""
    return torch.tensor([1.0, 2.0, 3.0])

@pytest.fixture
def jax_array():
    """Fixture for JAX array operations"""
    return jnp.array([1.0, 2.0, 3.0])

@pytest.fixture
def dummy_state():
    """Fixture for a dummy drone state"""
    return np.zeros(12)  # 12-dimensional state vector

@pytest.fixture
def dummy_action():
    """Fixture for a dummy drone action"""
    return np.zeros(4)   # 4-dimensional action vector 
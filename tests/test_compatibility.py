import pytest
import torch
import jax
import jax.numpy as jnp
import gymnasium as gym
import pybullet as p
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm
import seaborn as sns

def test_numpy_compatibility():
    """Test NumPy version and basic functionality."""
    import numpy as np
    assert np.__version__ >= "1.24.0"
    # Test basic NumPy operations
    arr = np.array([1, 2, 3])
    assert arr.mean() == 2.0
    assert arr.sum() == 6

def test_torch_compatibility():
    """Test PyTorch version and basic functionality."""
    assert torch.__version__ >= "2.0.0"
    # Test basic PyTorch operations
    tensor = torch.tensor([1, 2, 3])
    assert tensor.mean().item() == 2.0
    assert tensor.sum().item() == 6

def test_jax_compatibility():
    """Test JAX version and basic functionality."""
    assert jax.__version__ >= "0.4.13"
    # Test basic JAX operations
    arr = jnp.array([1, 2, 3])
    assert jnp.mean(arr).item() == 2.0
    assert jnp.sum(arr).item() == 6

def test_gymnasium_compatibility():
    """Test Gymnasium version and basic functionality."""
    assert gym.__version__ >= "0.27.0"
    # Test basic Gymnasium functionality
    env = gym.make('CartPole-v1')
    assert env.action_space.n == 2
    assert env.observation_space.shape == (4,)

def test_pybullet_compatibility():
    """Test PyBullet version and basic functionality."""
    assert p.__version__ >= "3.2.0"
    # Test basic PyBullet functionality
    physicsClient = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    assert p.getGravity() == (0, 0, -9.81)
    p.disconnect()

def test_matplotlib_compatibility():
    """Test Matplotlib version and basic functionality."""
    assert matplotlib.__version__ >= "3.0.0"
    # Test basic Matplotlib functionality
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.close()

def test_pandas_compatibility():
    """Test Pandas version and basic functionality."""
    assert pd.__version__ >= "1.5.0"
    # Test basic Pandas functionality
    df = pd.DataFrame({'A': [1, 2, 3]})
    assert df['A'].mean() == 2.0

def test_yaml_compatibility():
    """Test PyYAML version and basic functionality."""
    assert yaml.__version__ >= "6.0"
    # Test basic YAML functionality
    data = yaml.safe_load("key: value")
    assert data['key'] == 'value'

def test_pillow_compatibility():
    """Test Pillow version and basic functionality."""
    assert Image.__version__ >= "9.0.0"
    # Test basic Pillow functionality
    img = Image.new('RGB', (100, 100), color='red')
    assert img.size == (100, 100)

def test_tqdm_compatibility():
    """Test tqdm version and basic functionality."""
    assert tqdm.__version__ >= "4.64.0"
    # Test basic tqdm functionality
    for _ in tqdm(range(1)):
        pass

def test_seaborn_compatibility():
    """Test Seaborn version and basic functionality."""
    assert sns.__version__ >= "0.11.0"
    # Test basic Seaborn functionality
    data = torch.randn(100)
    sns.histplot(data)
    plt.close() 
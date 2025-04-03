import jax
import jax.numpy as jnp
from typing import Tuple
from jax import jit

@jit
def quaternion_to_euler(quaternion: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quaternion: Quaternion [x, y, z, w]
        
    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    x, y, z, w = quaternion
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = jnp.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    
    return jnp.array([roll, pitch, yaw])

@jit
def quaternion_to_rotation_matrix(quaternion: jnp.ndarray) -> jnp.ndarray:
    """Convert quaternion to rotation matrix.
    
    Args:
        quaternion: Quaternion [x, y, z, w]
        
    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = quaternion
    return jnp.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

@jit
def euler_to_quaternion(euler: jnp.ndarray) -> jnp.ndarray:
    """Convert Euler angles to quaternion.
    
    Args:
        euler: Euler angles [roll, pitch, yaw] in radians
        
    Returns:
        Quaternion [x, y, z, w]
    """
    roll, pitch, yaw = euler
    
    # Roll
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    
    # Pitch
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    
    # Yaw
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    
    # Quaternion components
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    
    return jnp.array([x, y, z, w])

@jit
def normalize_quaternion(quaternion: jnp.ndarray) -> jnp.ndarray:
    """Normalize a quaternion to unit length.
    
    Args:
        quaternion: Quaternion [x, y, z, w]
        
    Returns:
        Normalized quaternion
    """
    norm = jnp.linalg.norm(quaternion)
    return jnp.where(norm == 0,
                    jnp.array([0.0, 0.0, 0.0, 1.0]),
                    quaternion / norm)

@jit
def quaternion_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Multiply two quaternions.
    
    Args:
        q1: First quaternion [x, y, z, w]
        q2: Second quaternion [x, y, z, w]
        
    Returns:
        Product quaternion
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return jnp.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

@jit
def quaternion_inverse(quaternion: jnp.ndarray) -> jnp.ndarray:
    """Compute the inverse of a quaternion.
    
    Args:
        quaternion: Quaternion [x, y, z, w]
        
    Returns:
        Inverse quaternion
    """
    x, y, z, w = quaternion
    norm_squared = w*w + x*x + y*y + z*z
    return jnp.where(norm_squared == 0,
                    jnp.array([0.0, 0.0, 0.0, 1.0]),
                    jnp.array([-x, -y, -z, w]) / norm_squared)

@jit
def quaternion_rotate_vector(quaternion: jnp.ndarray, vector: jnp.ndarray) -> jnp.ndarray:
    """Rotate a vector by a quaternion.
    
    Args:
        quaternion: Quaternion [x, y, z, w]
        vector: 3D vector to rotate
        
    Returns:
        Rotated vector
    """
    # Convert vector to quaternion
    v_quat = jnp.concatenate([vector, jnp.zeros(1)])
    
    # Perform rotation: q * v * q^(-1)
    q_inv = quaternion_inverse(quaternion)
    rotated = quaternion_multiply(quaternion_multiply(quaternion, v_quat), q_inv)
    
    return rotated[:3]

@jit
def skew_symmetric(vector: jnp.ndarray) -> jnp.ndarray:
    """Create skew-symmetric matrix from 3D vector.
    
    Args:
        vector: 3D vector [x, y, z]
        
    Returns:
        3x3 skew-symmetric matrix
    """
    x, y, z = vector
    return jnp.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ])

@jit
def compute_rotation_error(current: jnp.ndarray, desired: jnp.ndarray) -> jnp.ndarray:
    """Compute rotation error between two quaternions.
    
    Args:
        current: Current quaternion [x, y, z, w]
        desired: Desired quaternion [x, y, z, w]
        
    Returns:
        Rotation error as Euler angles [roll, pitch, yaw]
    """
    # Compute error quaternion: q_error = q_desired * q_current^(-1)
    q_current_inv = quaternion_inverse(current)
    q_error = quaternion_multiply(desired, q_current_inv)
    
    # Convert to Euler angles
    return quaternion_to_euler(q_error)

@jit
def compute_position_error(current: jnp.ndarray, desired: jnp.ndarray) -> jnp.ndarray:
    """Compute position error between two 3D vectors.
    
    Args:
        current: Current position
        desired: Desired position
        
    Returns:
        Position error vector
    """
    return desired - current

@jit
def compute_velocity_error(current: jnp.ndarray, desired: jnp.ndarray) -> jnp.ndarray:
    """Compute velocity error between two 3D vectors.
    
    Args:
        current: Current velocity
        desired: Desired velocity
        
    Returns:
        Velocity error vector
    """
    return desired - current 
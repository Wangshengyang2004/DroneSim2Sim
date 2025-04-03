"""Utility functions for drone simulation and control."""

from .math import (
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    euler_to_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_inverse,
    quaternion_rotate_vector,
    skew_matrix,
    compute_rotation_error,
    compute_position_error,
    compute_velocity_error,
    compute_attitude_error,
    compute_angular_velocity_error,
    normalize_angle,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix
)

from .math_jax import (
    quaternion_to_euler as quaternion_to_euler_jax,
    quaternion_to_rotation_matrix as quaternion_to_rotation_matrix_jax,
    euler_to_quaternion as euler_to_quaternion_jax,
    normalize_quaternion as normalize_quaternion_jax,
    quaternion_multiply as quaternion_multiply_jax,
    quaternion_inverse as quaternion_inverse_jax,
    quaternion_rotate_vector as quaternion_rotate_vector_jax,
    skew_symmetric as skew_matrix_jax,
    compute_rotation_error as compute_rotation_error_jax,
    compute_position_error as compute_position_error_jax,
    compute_velocity_error as compute_velocity_error_jax
)

from .urdf import URDFLoader
from .simulator_urdf import PyBulletURDFLoader

__all__ = [
    # PyTorch functions
    'quaternion_to_euler',
    'quaternion_to_rotation_matrix',
    'euler_to_quaternion',
    'normalize_quaternion',
    'quaternion_multiply',
    'quaternion_inverse',
    'quaternion_rotate_vector',
    'skew_matrix',
    'compute_rotation_error',
    'compute_position_error',
    'compute_velocity_error',
    'compute_attitude_error',
    'compute_angular_velocity_error',
    'normalize_angle',
    'rotation_matrix_to_euler',
    'euler_to_rotation_matrix',
    
    # JAX functions
    'quaternion_to_euler_jax',
    'quaternion_to_rotation_matrix_jax',
    'euler_to_quaternion_jax',
    'normalize_quaternion_jax',
    'quaternion_multiply_jax',
    'quaternion_inverse_jax',
    'quaternion_rotate_vector_jax',
    'skew_matrix_jax',
    'compute_rotation_error_jax',
    'compute_position_error_jax',
    'compute_velocity_error_jax',
    
    # URDF loader
    'URDFLoader',
    'PyBulletURDFLoader'
] 
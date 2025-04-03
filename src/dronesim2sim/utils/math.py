import torch
from typing import Tuple

def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:
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
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw])

def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix.
    
    Args:
        quaternion: Quaternion [x, y, z, w]
        
    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = quaternion
    return torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y])
    ])

def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to quaternion.
    
    Args:
        euler: Euler angles [roll, pitch, yaw] in radians
        
    Returns:
        Quaternion [x, y, z, w]
    """
    roll, pitch, yaw = euler
    
    # Roll
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    # Pitch
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    
    # Yaw
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    
    # Quaternion components
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    
    return torch.stack([x, y, z, w])

def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize a quaternion to unit length.
    
    Args:
        quaternion: Quaternion [x, y, z, w]
        
    Returns:
        Normalized quaternion
    """
    norm = torch.norm(quaternion)
    if norm == 0:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=quaternion.device)
    return quaternion / norm

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.
    
    Args:
        q1: First quaternion [x, y, z, w]
        q2: Second quaternion [x, y, z, w]
        
    Returns:
        Product quaternion
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quaternion_inverse(quaternion: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a quaternion.
    
    Args:
        quaternion: Quaternion [x, y, z, w]
        
    Returns:
        Inverse quaternion
    """
    x, y, z, w = quaternion
    norm_squared = w*w + x*x + y*y + z*z
    if norm_squared == 0:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=quaternion.device)
    return torch.stack([-x, -y, -z, w]) / norm_squared

def quaternion_rotate_vector(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion.
    
    Args:
        quaternion: Quaternion [x, y, z, w]
        vector: 3D vector to rotate
        
    Returns:
        Rotated vector
    """
    # Convert vector to quaternion
    v_quat = torch.cat([vector, torch.zeros(1, device=vector.device)])
    
    # Perform rotation: q * v * q^(-1)
    q_inv = quaternion_inverse(quaternion)
    rotated = quaternion_multiply(quaternion_multiply(quaternion, v_quat), q_inv)
    
    return rotated[:3]

def skew_symmetric(vector: torch.Tensor) -> torch.Tensor:
    """Create skew-symmetric matrix from 3D vector.
    
    Args:
        vector: 3D vector [x, y, z]
        
    Returns:
        3x3 skew-symmetric matrix
    """
    x, y, z = vector
    return torch.stack([
        torch.stack([torch.zeros(1, device=vector.device), -z, y]),
        torch.stack([z, torch.zeros(1, device=vector.device), -x]),
        torch.stack([-y, x, torch.zeros(1, device=vector.device)])
    ])

def compute_rotation_error(current: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
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

def compute_position_error(current: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    """Compute position error between two 3D vectors.
    
    Args:
        current: Current position
        desired: Desired position
        
    Returns:
        Position error vector
    """
    return desired - current

def compute_velocity_error(current: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    """Compute velocity error between two 3D vectors.
    
    Args:
        current: Current velocity
        desired: Desired velocity
        
    Returns:
        Velocity error vector
    """
    return desired - current

def compute_attitude_error(current: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    """Compute attitude error between current and desired orientations.
    
    Args:
        current: Current Euler angles [roll, pitch, yaw]
        desired: Desired Euler angles [roll, pitch, yaw]
        
    Returns:
        Attitude error [roll_error, pitch_error, yaw_error]
    """
    # Convert to [-pi, pi] range
    error = current - desired
    error = torch.atan2(torch.sin(error), torch.cos(error))
    return error

def compute_angular_velocity_error(current: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    """Compute angular velocity error.
    
    Args:
        current: Current angular velocity [wx, wy, wz]
        desired: Desired angular velocity [wx, wy, wz]
        
    Returns:
        Angular velocity error
    """
    return current - desired

def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-pi, pi] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))

def skew_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Create skew-symmetric matrix from vector.
    
    Args:
        vector: 3D vector [x, y, z]
        
    Returns:
        3x3 skew-symmetric matrix
    """
    x, y, z = vector
    return torch.stack([
        torch.stack([0.0, -z, y]),
        torch.stack([z, 0.0, -x]),
        torch.stack([-y, x, 0.0])
    ])

def rotation_matrix_to_euler(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to Euler angles.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Euler angles [roll, pitch, yaw]
    """
    # Handle gimbal lock cases
    if torch.abs(R[2, 0]) >= 1:
        yaw = 0
        if R[2, 0] < 0:
            pitch = torch.tensor(torch.pi / 2)
            roll = torch.atan2(R[1, 2], R[1, 1])
        else:
            pitch = torch.tensor(-torch.pi / 2)
            roll = -torch.atan2(R[1, 2], R[1, 1])
    else:
        pitch = -torch.asin(R[2, 0])
        cos_pitch = torch.cos(pitch)
        roll = torch.atan2(R[2, 1] / cos_pitch, R[2, 2] / cos_pitch)
        yaw = torch.atan2(R[1, 0] / cos_pitch, R[0, 0] / cos_pitch)
    
    return torch.stack([roll, pitch, yaw])

def euler_to_rotation_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to rotation matrix.
    
    Args:
        euler: Euler angles [roll, pitch, yaw]
        
    Returns:
        3x3 rotation matrix
    """
    roll, pitch, yaw = euler
    
    # Roll
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    
    # Pitch
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    
    # Yaw
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    
    # Rotation matrix
    R = torch.stack([
        torch.stack([cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy]),
        torch.stack([cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy]),
        torch.stack([-sp, sr*cp, cr*cp])
    ])
    
    return R 
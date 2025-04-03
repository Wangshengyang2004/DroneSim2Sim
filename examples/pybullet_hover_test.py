import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Tuple, List
from drone_env import DroneEnv
from pid_controller import PIDController
from core.tasks import HoverTask
from core.utils import quaternion_to_euler

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run hover tests with different controllers")
    parser.add_argument("--target_height", type=float, default=1.0, help="Target height for hover")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI")
    return parser.parse_args()

def setup_controller(env: DroneEnv, target_height: float) -> PIDController:
    """Setup the PID controller.
    
    Args:
        env: Drone environment
        target_height: Target height for hover
        
    Returns:
        Configured PID controller
    """
    # Define target position
    target_position = torch.tensor([0.0, 0.0, target_height], dtype=torch.float32)
    
    # Define motor limits
    motor_limits = {
        "thrust": 5.0,  # Maximum thrust in Newtons
        "torque": 0.2,  # Maximum torque in Nm
    }
    
    # Create controller
    controller = PIDController(
        dt=1.0 / env.freq,
        target_position=target_position,
        motor_limits=motor_limits,
    )
    
    return controller

def run_hover_test(
    env: DroneEnv,
    controller: PIDController,
    duration: float,
) -> Tuple[List[float], List[Dict[str, torch.Tensor]]]:
    """Run hover test with PID controller.
    
    Args:
        env: Drone environment
        controller: PID controller
        duration: Test duration in seconds
        
    Returns:
        List of rewards and observations
    """
    # Initialize lists to store data
    rewards = []
    observations = []
    
    # Reset environment and controller
    obs, _ = env.reset()
    controller.reset()
    
    # Run simulation
    num_steps = int(duration * env.freq)
    for _ in range(num_steps):
        # Compute control action
        action = controller.compute_control(obs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store data
        rewards.append(reward)
        observations.append(obs)
        
        # Check if episode terminated
        if terminated or truncated:
            print("Episode terminated early!")
            break
    
    return rewards, observations

def plot_results(rewards, observations, target_height):
    """Plot the results of the hover test."""
    # Convert observations to tensors if they're not already
    positions = torch.stack([obs["position"] if torch.is_tensor(obs["position"]) else torch.tensor(obs["position"]) for obs in observations])
    orientations = torch.stack([obs["orientation"] if torch.is_tensor(obs["orientation"]) else torch.tensor(obs["orientation"]) for obs in observations])
    linear_velocities = torch.stack([obs["linear_velocity"] if torch.is_tensor(obs["linear_velocity"]) else torch.tensor(obs["linear_velocity"]) for obs in observations])
    angular_velocities = torch.stack([obs["angular_velocity"] if torch.is_tensor(obs["angular_velocity"]) else torch.tensor(obs["angular_velocity"]) for obs in observations])
    
    # Convert rewards to tensor if it's not already
    rewards = rewards if torch.is_tensor(rewards) else torch.tensor(rewards)
    
    # Create target position tensor
    target_position = torch.tensor([0.0, 0.0, target_height], dtype=torch.float32)
    
    # Compute errors
    position_errors = torch.norm(positions - target_position, dim=1)
    orientation_errors = torch.stack([quaternion_to_euler(quat) for quat in orientations])
    velocity_errors = torch.norm(linear_velocities, dim=1)
    angular_velocity_errors = torch.norm(angular_velocities, dim=1)
    
    # Create time tensor
    time = torch.arange(len(rewards), dtype=torch.float32) / 240.0  # Assuming 240Hz control frequency
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Hover Test Results")
    
    # Plot position error
    axs[0, 0].plot(time.numpy(), position_errors.numpy())
    axs[0, 0].set_title("Position Error")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Error (m)")
    axs[0, 0].grid(True)
    
    # Plot orientation error (roll, pitch, yaw)
    axs[0, 1].plot(time.numpy(), orientation_errors[:, 0].numpy(), label="Roll")
    axs[0, 1].plot(time.numpy(), orientation_errors[:, 1].numpy(), label="Pitch")
    axs[0, 1].plot(time.numpy(), orientation_errors[:, 2].numpy(), label="Yaw")
    axs[0, 1].set_title("Orientation Error")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Error (rad)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot linear velocity error
    axs[1, 0].plot(time.numpy(), velocity_errors.numpy())
    axs[1, 0].set_title("Linear Velocity Error")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Error (m/s)")
    axs[1, 0].grid(True)
    
    # Plot angular velocity error
    axs[1, 1].plot(time.numpy(), angular_velocity_errors.numpy())
    axs[1, 1].set_title("Angular Velocity Error")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Error (rad/s)")
    axs[1, 1].grid(True)
    
    # Plot rewards
    axs[2, 0].plot(time.numpy(), rewards.numpy())
    axs[2, 0].set_title("Rewards")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Reward")
    axs[2, 0].grid(True)
    
    # Plot 3D trajectory
    axs[2, 1].plot(positions[:, 0].numpy(), positions[:, 1].numpy(), positions[:, 2].numpy())
    axs[2, 1].set_title("3D Trajectory")
    axs[2, 1].set_xlabel("X (m)")
    axs[2, 1].set_ylabel("Y (m)")
    axs[2, 1].set_zlabel("Z (m)")
    axs[2, 1].grid(True)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create environment
    env = DroneEnv(
        task_name="hover",
        render_mode="human" if args.gui else None,
        gui=args.gui,
    )
    
    try:
        # Setup controller
        controller = setup_controller(env, args.target_height)
        
        # Run test
        print("Starting hover test...")
        rewards, observations = run_hover_test(env, controller, args.duration)
        
        # Plot results
        print("Plotting results...")
        plot_results(rewards, observations, args.target_height)
        
    finally:
        # Clean up
        env.close()

if __name__ == "__main__":
    main() 
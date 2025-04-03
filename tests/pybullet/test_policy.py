import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from dronesim2sim.core.simulators.pybullet.drone_env import DroneEnv

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test a drone control policy in PyBullet")
    
    # Environment parameters
    parser.add_argument("--task", type=str, default="hover", choices=["hover", "trajectory"], 
                        help="Task to test")
    parser.add_argument("--gui", action="store_true", 
                        help="Use GUI for visualization")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed")
    parser.add_argument("--freq", type=int, default=240, 
                        help="Control frequency (Hz)")
    
    # Policy parameters
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to policy file (.pt)")
    parser.add_argument("--random", action="store_true",
                        help="Use random actions instead of policy")
    
    # Test parameters
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode")
    
    # Rendering parameters
    parser.add_argument("--record", action="store_true",
                        help="Record video of test run")
    parser.add_argument("--output_dir", type=str, default="../results/pybullet",
                        help="Output directory for videos and plots")
    
    args = parser.parse_args()
    
    # Check for incompatible arguments
    if args.policy is None and not args.random:
        parser.error("Either --policy or --random must be specified")
    
    return args

def load_policy(policy_path):
    """Load a trained policy.
    
    Args:
        policy_path: Path to policy file
        
    Returns:
        Loaded policy
    """
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    
    try:
        policy = torch.load(policy_path)
        print(f"Loaded policy from {policy_path}")
        return policy
    except Exception as e:
        raise RuntimeError(f"Failed to load policy: {e}")

def test_policy(args):
    """Test a policy in the PyBullet environment.
    
    Args:
        args: Command line arguments
        
    Returns:
        Test results
    """
    # Create the environment
    render_mode = "rgb_array" if args.record else ("human" if args.gui else None)
    env = DroneEnv(
        task_name=args.task,
        render_mode=render_mode,
        seed=args.seed,
        freq=args.freq,
        gui=args.gui,
        record_video=args.record,
    )
    
    # Load policy if specified
    policy = None
    if not args.random:
        policy = load_policy(args.policy)
    
    # Create output directory if needed
    if args.record or args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run test episodes
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(args.episodes):
        print(f"Episode {episode + 1}/{args.episodes}")
        
        obs, info = env.reset()
        episode_reward = 0
        position_history = []
        
        for step in range(args.max_steps):
            # Choose action based on policy or random
            if args.random:
                action = env.action_space.sample()
            else:
                # Process observation according to the policy's expectations
                # This will depend on the specific policy implementation
                with torch.no_grad():
                    # Convert observation to tensor format expected by policy
                    obs_tensor = {}
                    for key, value in obs.items():
                        obs_tensor[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
                    
                    # Get action from policy
                    action = policy.predict(obs_tensor)[0]
            
            # Apply action and observe next state
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Store position for plotting
            position_history.append(obs["position"])
            
            # Check for episode end
            done = terminated or truncated
            if done:
                break
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # Plot trajectory for hover task
        if args.task == "hover" and len(position_history) > 0:
            plt.figure(figsize=(10, 5))
            
            position_history = torch.stack(position_history).cpu().numpy()
            plt.subplot(1, 2, 1)
            plt.plot(position_history[:, 0], position_history[:, 1])
            plt.title("XY Trajectory")
            plt.xlabel("X position")
            plt.ylabel("Y position")
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(len(position_history)), position_history[:, 2])
            plt.title("Z Trajectory (Height)")
            plt.xlabel("Step")
            plt.ylabel("Z position")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"pybullet_trajectory_episode_{episode}.png"))
            plt.close()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(args.episodes) + 1, episode_rewards, marker='o')
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(args.episodes) + 1, episode_lengths, marker='o')
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pybullet_episode_statistics.png"))
    
    # Save results to file
    results = {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "task": args.task,
        "simulator": "pybullet",
        "policy": args.policy if not args.random else "random",
    }
    
    np.save(os.path.join(args.output_dir, "pybullet_results.npy"), results)
    
    # Clean up
    env.close()
    
    return results

def main():
    """Main function."""
    args = parse_args()
    test_policy(args)

if __name__ == "__main__":
    main() 
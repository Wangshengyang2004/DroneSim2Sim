"""
Transfer Learning Example for DroneSim2Sim.

This script demonstrates how to train a policy in one simulator (PyBullet)
and transfer it to another simulator (in this case simulated - but would be applied to Isaac Sim).
The example uses Proximal Policy Optimization (PPO) for reinforcement learning.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Neural network for both actor and critic functions.
    
    The network takes the drone state as input and outputs:
    1. Mean of action distribution (for actor)
    2. State value (for critic)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the policy network.
        
        Args:
            input_dim: Dimension of the state space
            hidden_dim: Dimension of the hidden layers
            output_dim: Dimension of the action space
        """
        super(PolicyNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, output_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(output_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: The state input
            
        Returns:
            Tuple of (action_mean, action_log_std, value)
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Actor output (action mean and log std)
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        # Critic output (state value)
        value = self.critic(features)
        
        return action_mean, action_log_std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Sample an action from the policy.
        
        Args:
            state: The state input
            deterministic: Whether to return the deterministic action (mean)
            
        Returns:
            Tuple of (action, info)
        """
        # Get action distribution parameters
        action_mean, action_log_std, value = self.forward(state)
        
        if deterministic:
            # Return the mean action
            return action_mean, {"value": value}
        else:
            # Sample from the distribution
            action_std = torch.exp(action_log_std)
            distribution = Normal(action_mean, action_std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)
            
            return action, {"log_prob": log_prob, "value": value}
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the log probability and value of given state-action pairs.
        
        Args:
            state: The state input
            action: The action to evaluate
            
        Returns:
            Tuple of (log_prob, value, entropy)
        """
        # Get distribution parameters
        action_mean, action_log_std, value = self.forward(state)
        action_std = torch.exp(action_log_std)
        
        # Create the distribution
        distribution = Normal(action_mean, action_std)
        
        # Evaluate the action
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        
        return log_prob, value.squeeze(), entropy


class PPO:
    """
    Proximal Policy Optimization implementation for drone control.
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01
    ):
        """
        Initialize the PPO agent.
        
        Args:
            policy: Policy network (actor-critic)
            optimizer: Optimizer for policy
            device: Device to run computations on
            clip_ratio: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            target_kl: Target KL divergence for early stopping
        """
        self.policy = policy
        self.optimizer = optimizer
        self.device = device
        
        # PPO parameters
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        n_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Update the policy network using PPO.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Batch of log probabilities from old policy
            returns: Batch of returns
            advantages: Batch of advantages
            n_epochs: Number of epochs to update the policy
            batch_size: Mini-batch size
            
        Returns:
            Dictionary of training statistics
        """
        # Move data to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training stats
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0
        
        # Training loop
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        
        for epoch in range(n_epochs):
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Mini-batch update
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                if end > dataset_size:
                    end = dataset_size
                
                mb_indices = indices[start:end]
                
                # Get mini-batch
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Evaluate actions
                new_log_probs, values, entropy = self.policy.evaluate_actions(mb_states, mb_actions)
                
                # Policy loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Value loss
                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()
                
                # Entropy bonus
                entropy_bonus = entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
                
                # Approximate KL divergence
                with torch.no_grad():
                    kl = (mb_old_log_probs - new_log_probs).mean().item()
                    if kl > 1.5 * self.target_kl:
                        # Early stopping if KL divergence is too high
                        break
                
                # Gradient update
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Update stats
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
                total_kl += kl
                n_updates += 1
        
        # Average stats
        stats = {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "kl": total_kl / n_updates
        }
        
        return stats


def compute_returns_and_advantages(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute returns and advantages using Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        next_value: Value estimate for the next state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        Tuple of (returns, advantages)
    """
    # Convert lists to numpy arrays
    rewards = np.array(rewards)
    values = np.array(values)
    dones = np.array(dones)
    
    # Initialize arrays
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    
    # Initialize variables
    next_advantage = 0
    next_return = next_value
    
    # Compute returns and advantages (backwards)
    for t in reversed(range(len(rewards))):
        # Calculate returns (discounted rewards)
        if dones[t]:
            returns[t] = rewards[t]
        else:
            returns[t] = rewards[t] + gamma * next_return
        
        # Calculate TD error
        if dones[t]:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * next_value - values[t]
        
        # Calculate advantage with GAE
        advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * next_advantage
        
        # Update next values
        next_return = returns[t]
        next_advantage = advantages[t]
        next_value = values[t]
    
    return returns, advantages


def collect_rollout(
    env,
    policy: PolicyNetwork,
    n_steps: int,
    device: torch.device
) -> Dict[str, List]:
    """
    Collect experience from the environment.
    
    Args:
        env: Gym environment
        policy: Policy network
        n_steps: Number of steps to collect
        device: Device to run computations on
        
    Returns:
        Dictionary of collected experience
    """
    # Initialize data lists
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    
    # Reset environment
    state, _ = env.reset()
    done = False
    
    # Collect experience
    for _ in range(n_steps):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Get action
        with torch.no_grad():
            action, info = policy.get_action(state_tensor)
            value = info["value"]
            log_prob = info.get("log_prob")
        
        # Convert to numpy
        action_np = action.cpu().numpy().squeeze()
        
        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        
        # Store data
        states.append(state)
        actions.append(action_np)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.item() if log_prob is not None else 0)
        values.append(value.item())
        
        # Update state
        state = next_state
        
        # Reset if done
        if done:
            state, _ = env.reset()
    
    # Get the final state value
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        _, info = policy.get_action(state_tensor)
        next_value = info["value"].item()
    
    # Compute returns and advantages
    returns, advantages = compute_returns_and_advantages(
        rewards, values, dones, next_value
    )
    
    # Create dictionary of data
    data = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "log_probs": log_probs,
        "values": values,
        "returns": returns,
        "advantages": advantages
    }
    
    return data


def train_ppo(
    env,
    test_env,
    policy: PolicyNetwork,
    n_epochs: int = 100,
    n_steps: int = 2048,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    log_interval: int = 10,
    checkpoint_dir: str = "checkpoints"
) -> List[Dict]:
    """
    Train a policy using PPO.
    
    Args:
        env: Training environment
        test_env: Testing environment
        policy: Policy network
        n_epochs: Number of training epochs
        n_steps: Number of steps per epoch
        batch_size: Mini-batch size
        lr: Learning rate
        device: Device to run computations on
        log_interval: Interval for logging
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        List of training statistics
    """
    # Create optimizer
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Create PPO agent
    ppo = PPO(policy, optimizer, device)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training stats
    stats_list = []
    
    # Training loop
    for epoch in range(n_epochs):
        # Collect experience
        data = collect_rollout(env, policy, n_steps, device)
        
        # Convert data to tensors
        states = torch.FloatTensor(np.array(data["states"]))
        actions = torch.FloatTensor(np.array(data["actions"]))
        log_probs = torch.FloatTensor(np.array(data["log_probs"]))
        returns = torch.FloatTensor(data["returns"])
        advantages = torch.FloatTensor(data["advantages"])
        
        # Update policy
        update_stats = ppo.update(
            states, actions, log_probs, returns, advantages, 
            n_epochs=10, batch_size=batch_size
        )
        
        # Evaluate policy
        if epoch % log_interval == 0:
            # Test performance
            test_rewards = []
            for _ in range(5):  # 5 test episodes
                state, _ = test_env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action, _ = policy.get_action(state_tensor, deterministic=True)
                    action_np = action.cpu().numpy().squeeze()
                    state, reward, terminated, truncated, _ = test_env.step(action_np)
                    episode_reward += reward
                    done = terminated or truncated
                
                test_rewards.append(episode_reward)
            
            # Calculate average reward
            avg_reward = np.mean(test_rewards)
            
            # Update and log stats
            stats = {
                "epoch": epoch,
                "avg_reward": avg_reward,
                **update_stats
            }
            stats_list.append(stats)
            
            print(f"Epoch {epoch}: Avg Reward: {avg_reward:.2f}, Loss: {update_stats['loss']:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"ppo_policy_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_reward": avg_reward
            }, checkpoint_path)
    
    return stats_list


def plot_training_progress(stats_list: List[Dict], output_dir: str = "results"):
    """
    Plot training progress.
    
    Args:
        stats_list: List of training statistics
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    epochs = [stat["epoch"] for stat in stats_list]
    rewards = [stat["avg_reward"] for stat in stats_list]
    losses = [stat["loss"] for stat in stats_list]
    policy_losses = [stat["policy_loss"] for stat in stats_list]
    value_losses = [stat["value_loss"] for stat in stats_list]
    entropies = [stat["entropy"] for stat in stats_list]
    kls = [stat["kl"] for stat in stats_list]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 3, 1)
    plt.plot(epochs, rewards)
    plt.title("Average Reward")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.grid(True)
    
    # Plot losses
    plt.subplot(2, 3, 2)
    plt.plot(epochs, losses)
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    # Plot policy loss
    plt.subplot(2, 3, 3)
    plt.plot(epochs, policy_losses)
    plt.title("Policy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    # Plot value loss
    plt.subplot(2, 3, 4)
    plt.plot(epochs, value_losses)
    plt.title("Value Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    # Plot entropy
    plt.subplot(2, 3, 5)
    plt.plot(epochs, entropies)
    plt.title("Entropy")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.grid(True)
    
    # Plot KL divergence
    plt.subplot(2, 3, 6)
    plt.plot(epochs, kls)
    plt.title("KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("KL")
    plt.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()


def test_transfer(
    policy: PolicyNetwork,
    source_env,
    target_env,
    n_episodes: int = 10,
    device: torch.device = torch.device("cpu")
) -> Dict[str, float]:
    """
    Test the transfer of a policy from one simulator to another.
    
    Args:
        policy: Trained policy
        source_env: Source environment (training environment)
        target_env: Target environment (testing environment)
        n_episodes: Number of test episodes
        device: Device to run computations on
        
    Returns:
        Dictionary of performance metrics
    """
    # Set policy to evaluation mode
    policy.eval()
    
    # Test environments
    envs = {"source": source_env, "target": target_env}
    results = {"source": [], "target": []}
    
    # Test in both environments
    for env_name, env in envs.items():
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # Get action (deterministic for testing)
                with torch.no_grad():
                    action, _ = policy.get_action(state_tensor, deterministic=True)
                
                # Convert to numpy
                action_np = action.cpu().numpy().squeeze()
                
                # Step environment
                state, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                
                # Update episode stats
                episode_reward += reward
                episode_length += 1
            
            # Store episode stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate environment stats
        results[env_name] = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths)
        }
    
    # Calculate transfer metrics
    transfer_ratio = results["target"]["mean_reward"] / results["source"]["mean_reward"]
    transfer_gap = results["source"]["mean_reward"] - results["target"]["mean_reward"]
    
    # Add transfer metrics to results
    results["transfer_ratio"] = transfer_ratio
    results["transfer_gap"] = transfer_gap
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transfer learning example")
    
    # Environment parameters
    parser.add_argument("--source_env", type=str, default="PyBullet-v0",
                        help="Source environment ID")
    parser.add_argument("--target_env", type=str, default="IsaacSim-v0",
                        help="Target environment ID")
    
    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of steps per epoch")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension of the policy network")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results/transfer",
                        help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/transfer",
                        help="Checkpoint directory")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    
    args = parser.parse_args()
    return args


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create environments
    # For this example, we'll use PyBullet environments - in a real setup,
    # you would use different simulators for source and target
    try:
        # Try to import the custom environments
        from pybullet.drone_env import DroneEnv
        
        source_env = DroneEnv(render=False, control_freq=240, task='hover')
        target_env = DroneEnv(render=False, control_freq=240, task='hover', 
                             sim_params={"use_different_dynamics": True})
    except ImportError:
        # Fall back to gym environments
        print("Custom environments not found. Using gym environments instead.")
        source_env = gym.make("CartPole-v1")
        target_env = gym.make("CartPole-v1")
    
    # Get observation and action dimensions
    if isinstance(source_env.observation_space, gym.spaces.Dict):
        # For Dict observation spaces, we need to flatten the space
        obs_dim = sum(np.prod(source_env.observation_space[key].shape) 
                     for key in source_env.observation_space.spaces)
    else:
        obs_dim = np.prod(source_env.observation_space.shape)
    
    if isinstance(source_env.action_space, gym.spaces.Discrete):
        # For discrete action spaces
        act_dim = source_env.action_space.n
    else:
        # For continuous action spaces
        act_dim = np.prod(source_env.action_space.shape)
    
    # Create policy network
    policy = PolicyNetwork(
        input_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        output_dim=act_dim
    ).to(device)
    
    # Train policy
    print(f"Training policy on {args.source_env}...")
    stats_list = train_ppo(
        source_env,
        source_env,  # Use source env for testing during training
        policy,
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        log_interval=10,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Plot training progress
    plot_training_progress(stats_list, args.output_dir)
    
    # Test transfer
    print(f"Testing transfer from {args.source_env} to {args.target_env}...")
    transfer_results = test_transfer(
        policy,
        source_env,
        target_env,
        n_episodes=10,
        device=device
    )
    
    # Print results
    print("\nTransfer Learning Results:")
    print(f"Source Environment ({args.source_env}):")
    print(f"  Mean Reward: {transfer_results['source']['mean_reward']:.2f} ± {transfer_results['source']['std_reward']:.2f}")
    print(f"  Mean Episode Length: {transfer_results['source']['mean_length']:.2f} ± {transfer_results['source']['std_length']:.2f}")
    
    print(f"Target Environment ({args.target_env}):")
    print(f"  Mean Reward: {transfer_results['target']['mean_reward']:.2f} ± {transfer_results['target']['std_reward']:.2f}")
    print(f"  Mean Episode Length: {transfer_results['target']['mean_length']:.2f} ± {transfer_results['target']['std_length']:.2f}")
    
    print(f"Transfer Ratio: {transfer_results['transfer_ratio']:.2f}")
    print(f"Transfer Gap: {transfer_results['transfer_gap']:.2f}")
    
    # Save results
    import json
    with open(os.path.join(args.output_dir, "transfer_results.json"), "w") as f:
        json.dump(transfer_results, f, indent=2)
    
    # Close environments
    source_env.close()
    target_env.close()


if __name__ == "__main__":
    main() 
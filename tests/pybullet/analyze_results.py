import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze simulation results across simulators")
    
    parser.add_argument("--dir", type=str, default="./results",
                        help="Directory containing results files")
    parser.add_argument("--task", type=str, default="hover", choices=["hover", "trajectory"],
                        help="Task to analyze")
    parser.add_argument("--output", type=str, default="./results/comparison",
                        help="Output directory for analysis results")
    
    return parser.parse_args()

def load_results(results_dir: str, task: str) -> Dict[str, Any]:
    """Load results from NPY files for various simulators.
    
    Args:
        results_dir: Directory containing results
        task: Task name to filter results
        
    Returns:
        Dictionary of results by simulator
    """
    # Find all NPY files
    result_files = glob.glob(os.path.join(results_dir, "**/*.npy"), recursive=True)
    
    # Filter for the specified task
    results_by_simulator = {}
    
    for file_path in result_files:
        try:
            # Load the result file
            result = np.load(file_path, allow_pickle=True).item()
            
            # Check if this is for the requested task
            if result.get("task") == task:
                simulator = result.get("simulator", "unknown")
                results_by_simulator[simulator] = result
                print(f"Loaded {simulator} results from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results_by_simulator

def plot_reward_comparison(results: Dict[str, Any], output_dir: str, task: str):
    """Plot reward comparison across simulators.
    
    Args:
        results: Results by simulator
        output_dir: Output directory for plots
        task: Task name
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for box plot
    simulators = []
    rewards_data = []
    mean_rewards = []
    std_rewards = []
    
    for simulator, result in results.items():
        simulators.append(simulator)
        rewards = result.get("rewards", [])
        rewards_data.append(rewards)
        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))
    
    # Sort simulators by mean reward (descending)
    sorted_indices = np.argsort(mean_rewards)[::-1]
    simulators = [simulators[i] for i in sorted_indices]
    rewards_data = [rewards_data[i] for i in sorted_indices]
    mean_rewards = [mean_rewards[i] for i in sorted_indices]
    std_rewards = [std_rewards[i] for i in sorted_indices]
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(rewards_data, labels=simulators, vert=True, patch_artist=True,
                boxprops=dict(facecolor="lightblue"))
    plt.title(f"Reward Distribution Across Simulators ({task.capitalize()} Task)")
    plt.ylabel("Episode Reward")
    plt.xlabel("Simulator")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{task}_reward_boxplot.png"))
    
    # Create bar plot with error bars
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(simulators)))
    plt.bar(simulators, mean_rewards, yerr=std_rewards, capsize=10, color=colors, alpha=0.7)
    plt.title(f"Mean Reward Across Simulators ({task.capitalize()} Task)")
    plt.ylabel("Mean Episode Reward")
    plt.xlabel("Simulator")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add value labels above bars
    for i, value in enumerate(mean_rewards):
        plt.text(i, value + std_rewards[i] + 0.05, f"{value:.2f}", ha="center")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{task}_mean_reward.png"))

def plot_episode_length_comparison(results: Dict[str, Any], output_dir: str, task: str):
    """Plot episode length comparison across simulators.
    
    Args:
        results: Results by simulator
        output_dir: Output directory for plots
        task: Task name
    """
    # Prepare data
    simulators = []
    lengths_data = []
    mean_lengths = []
    std_lengths = []
    
    for simulator, result in results.items():
        simulators.append(simulator)
        lengths = result.get("lengths", [])
        lengths_data.append(lengths)
        mean_lengths.append(np.mean(lengths))
        std_lengths.append(np.std(lengths))
    
    # Sort simulators by mean episode length (descending)
    sorted_indices = np.argsort(mean_lengths)[::-1]
    simulators = [simulators[i] for i in sorted_indices]
    lengths_data = [lengths_data[i] for i in sorted_indices]
    mean_lengths = [mean_lengths[i] for i in sorted_indices]
    std_lengths = [std_lengths[i] for i in sorted_indices]
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(lengths_data, labels=simulators, vert=True, patch_artist=True,
                boxprops=dict(facecolor="lightgreen"))
    plt.title(f"Episode Length Distribution Across Simulators ({task.capitalize()} Task)")
    plt.ylabel("Episode Length (steps)")
    plt.xlabel("Simulator")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{task}_length_boxplot.png"))
    
    # Create bar plot with error bars
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(simulators)))
    plt.bar(simulators, mean_lengths, yerr=std_lengths, capsize=10, color=colors, alpha=0.7)
    plt.title(f"Mean Episode Length Across Simulators ({task.capitalize()} Task)")
    plt.ylabel("Mean Episode Length (steps)")
    plt.xlabel("Simulator")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add value labels above bars
    for i, value in enumerate(mean_lengths):
        plt.text(i, value + std_lengths[i] + 0.05, f"{value:.1f}", ha="center")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{task}_mean_length.png"))

def generate_comparison_table(results: Dict[str, Any], output_dir: str, task: str):
    """Generate a comparison table of metrics across simulators.
    
    Args:
        results: Results by simulator
        output_dir: Output directory for table
        task: Task name
    """
    # Prepare data
    table_data = []
    headers = ["Simulator", "Mean Reward", "Std Reward", "Mean Length", "Std Length", "Success Rate"]
    
    for simulator, result in results.items():
        rewards = result.get("rewards", [])
        lengths = result.get("lengths", [])
        
        # Calculate success rate (assuming success is not crashing before max steps)
        max_steps = 1000  # This should be extracted from the results ideally
        success_rate = sum(1 for length in lengths if length >= max_steps) / len(lengths) if len(lengths) > 0 else 0
        
        row = [
            simulator,
            f"{np.mean(rewards):.4f}",
            f"{np.std(rewards):.4f}",
            f"{np.mean(lengths):.1f}",
            f"{np.std(lengths):.1f}",
            f"{success_rate * 100:.1f}%",
        ]
        table_data.append(row)
    
    # Sort by mean reward (descending)
    table_data.sort(key=lambda x: float(x[1]), reverse=True)
    
    # Write table to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{task}_comparison.csv"), "w") as f:
        f.write(",".join(headers) + "\n")
        for row in table_data:
            f.write(",".join(row) + "\n")
    
    print(f"Comparison table saved to {os.path.join(output_dir, f'{task}_comparison.csv')}")

def main():
    """Main function."""
    args = parse_args()
    
    # Load results
    results = load_results(args.dir, args.task)
    
    if not results:
        print(f"No results found for task '{args.task}' in directory '{args.dir}'")
        return
    
    # Plot results
    plot_reward_comparison(results, args.output, args.task)
    plot_episode_length_comparison(results, args.output, args.task)
    
    # Generate comparison table
    generate_comparison_table(results, args.output, args.task)
    
    print(f"Analysis complete. Results saved to {args.output}")

if __name__ == "__main__":
    main() 
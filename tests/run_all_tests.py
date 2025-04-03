#!/usr/bin/env python3
"""Script to run all test suites for DroneSim2Sim."""

import os
import subprocess
import sys
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run all test suites for DroneSim2Sim")
    
    parser.add_argument("--core", action="store_true", help="Run only core tests")
    parser.add_argument("--pybullet", action="store_true", help="Run only PyBullet tests")
    parser.add_argument("--isaac_sim", action="store_true", help="Run only Isaac Sim tests")
    parser.add_argument("--mujoco", action="store_true", help="Run only MuJoCo tests")
    parser.add_argument("--gazebo", action="store_true", help="Run only Gazebo tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # If no specific test suite is selected, run all
    if not any([args.core, args.pybullet, args.isaac_sim, args.mujoco, args.gazebo]):
        args.core = args.pybullet = args.isaac_sim = args.mujoco = args.gazebo = True
    
    return args

def run_tests(test_dir, verbose=False):
    """Run pytest on a specific test directory.
    
    Args:
        test_dir: Directory containing tests to run
        verbose: Whether to use verbose output
        
    Returns:
        True if tests passed, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running tests in {test_dir}")
    print(f"{'='*80}")
    
    # Construct command
    cmd = ["pytest", test_dir]
    if verbose:
        cmd.append("-v")
    
    # Run pytest
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Print output
    print(result.stdout)
    if result.stderr:
        print(f"Errors:\n{result.stderr}")
    
    return result.returncode == 0

def main():
    """Main function."""
    args = parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Track test results
    results = {}
    
    # Run selected test suites
    if args.core:
        results["core"] = run_tests(os.path.join(script_dir, "core"), args.verbose)
    
    if args.pybullet:
        results["pybullet"] = run_tests(os.path.join(script_dir, "pybullet"), args.verbose)
    
    if args.isaac_sim:
        results["isaac_sim"] = run_tests(os.path.join(script_dir, "isaac_sim"), args.verbose)
    
    if args.mujoco:
        results["mujoco"] = run_tests(os.path.join(script_dir, "mujoco"), args.verbose)
    
    if args.gazebo:
        results["gazebo"] = run_tests(os.path.join(script_dir, "gazebo"), args.verbose)
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    all_passed = True
    for suite, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{suite:10}: {status}")
        all_passed = all_passed and passed
    
    # Return appropriate exit code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main() 
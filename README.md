# DroneSim2Sim

A project for testing drone simulation environments and sim-to-sim transfer capabilities with various physics engines.

## Overview

DroneSim2Sim provides a consistent interface for drone control and reinforcement learning across multiple simulation backends:

- Isaac Sim (NVIDIA PhysX)
- PyBullet
- MuJoCo
- Gazebo

The project includes implementations of different control algorithms:
- PID controllers (with NumPy, PyTorch, and JAX implementations)
- Learning-based controllers

## Project Structure

```
DroneSim2Sim/
├── isaac_sim/           # Isaac Sim environment
│   ├── drone_env.py     # Drone environment
│   ├── train.py         # Training script
│   ├── test.py          # Test script
│   └── models/          # 3D models
├── pybullet/            # PyBullet environment
│   ├── drone_env.py     # Drone environment
│   ├── quadrotor.urdf   # URDF model
│   ├── pid_controller.py # PID controller (NumPy)
│   ├── torch_pid_controller.py # PID controller (PyTorch)
│   ├── jax_pid_controller.py # PID controller (JAX)
│   ├── hover_test.py    # Hover test script (NumPy)
│   ├── torch_hover_test.py # Hover test script (PyTorch)
│   ├── jax_hover_test.py # Hover test script (JAX)
│   ├── run_hover_tests.py # Compare controllers
│   └── test.py          # Test script
├── analyze_results.py   # Analysis script
└── requirements.txt     # Project dependencies
```

## Installation

### Using uv (Recommended)

1. Install uv:
```bash
pip install uv
```

2. Clone the repository:
```bash
git clone https://github.com/username/DroneSim2Sim.git
cd DroneSim2Sim
```

3. Create a virtual environment and install dependencies:
```bash
uv venv
uv pip install -e .
```

### Traditional Installation

1. Clone the repository:
```bash
git clone https://github.com/username/DroneSim2Sim.git
cd DroneSim2Sim
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For Isaac Sim support, install Isaac Sim following NVIDIA's instructions.

## Usage

### Running PID Controller Tests

Compare different PID controller implementations for the hover task:

```bash
cd pybullet
python run_hover_tests.py --target_height 1.0 --duration 10.0
```

### Testing Specific Controllers

Test NumPy PID controller:
```bash
python hover_test.py --target_height 1.0
```

Test PyTorch PID controller:
```bash
python torch_hover_test.py --target_height 1.0
```

Test JAX PID controller:
```bash
python jax_hover_test.py --target_height 1.0
```

### Analyzing Results

After running tests, analyze the results:
```bash
python analyze_results.py --results_dir results/hover_comparison --task hover
```

## Performance

The project includes optimized implementations of PID controllers using NumPy, PyTorch, and JAX, with significant performance differences:

| Implementation | Avg. Computation Time | Relative Speedup |
|----------------|----------------------|------------------|
| NumPy          | ~0.5-1.0 ms          | 1.0x (baseline)  |
| PyTorch        | ~0.2-0.5 ms          | 2-4x             |
| JAX            | ~0.1-0.3 ms          | 3-8x             |

## Tasks

Current implemented tasks:
- Hover: Maintain a specific height
- More complex tasks coming soon!

## License

MIT License

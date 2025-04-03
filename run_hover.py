import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Import the hover test
from dronesim2sim.pybullet.hover_test import main

if __name__ == "__main__":
    main() 
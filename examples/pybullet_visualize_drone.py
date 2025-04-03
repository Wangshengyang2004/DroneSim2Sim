import os
import time
import argparse
import numpy as np
import pybullet as p
import pybullet_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize the quadrotor model in PyBullet")
    parser.add_argument("--urdf", type=str, default="quadrotor.urdf",
                        help="Path to the URDF file")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration to visualize the model (in seconds)")
    parser.add_argument("--animate", action="store_true", default=True,
                        help="Animate the propellers")
    return parser.parse_args()


def main():
    """Visualize the quadrotor model."""
    # Parse arguments
    args = parse_args()
    
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load plane
    p.loadURDF("plane.urdf")
    
    # Set up camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )
    
    # Load quadrotor model
    urdf_path = args.urdf
    if not os.path.isabs(urdf_path):
        # Make path relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(script_dir, urdf_path)
    
    print(f"Loading URDF from: {urdf_path}")
    drone = p.loadURDF(urdf_path, [0, 0, 1], useFixedBase=False)
    
    # Get number of joints
    num_joints = p.getNumJoints(drone)
    print(f"Number of joints: {num_joints}")
    
    # Print joint info
    print("\nJoint Information:")
    for i in range(num_joints):
        joint_info = p.getJointInfo(drone, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}")
    
    # Calculate simulation duration
    steps = int(args.duration * 240)  # 240 Hz
    
    # Add debug parameter sliders
    motor_sliders = []
    if args.animate:
        for i in range(4):
            motor_sliders.append(
                p.addUserDebugParameter(f"Motor {i+1}", 0, 10, 5)
            )
    
    # Animation loop
    print("\nVisualization started. Close the window or press Ctrl+C to exit.")
    for i in range(steps):
        # Get motor values from sliders if animate is enabled
        if args.animate:
            motor_values = [p.readUserDebugParameter(slider) for slider in motor_sliders]
            
            # Set joint velocities for propellers
            propeller_joints = [1, 3, 5, 7]  # Updated joint indices
            for j, joint in enumerate(propeller_joints):
                # Scale motor values to reasonable angular velocity
                angular_velocity = motor_values[j] * 40
                p.setJointMotorControl2(
                    drone, 
                    joint, 
                    p.VELOCITY_CONTROL, 
                    targetVelocity=angular_velocity,
                    force=10
                )
        
        # Step simulation
        p.stepSimulation()
        
        # Add some lift force based on motor values if animate is enabled
        if args.animate:
            # Apply lift force
            total_thrust = sum(motor_values) * 5
            p.applyExternalForce(
                drone,
                -1,  # Apply to base link
                [0, 0, total_thrust],
                [0, 0, 0],
                p.LINK_FRAME
            )
        
        # Display position
        pos, orn = p.getBasePositionAndOrientation(drone)
        if i % 100 == 0:
            print(f"Position: {pos}, Orientation: {p.getEulerFromQuaternion(orn)}")
        
        # Small delay for visualization
        time.sleep(1/240)
    
    # Disconnect when done
    p.disconnect()


if __name__ == "__main__":
    main() 
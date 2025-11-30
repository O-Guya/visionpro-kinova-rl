#!/usr/bin/env python3
"""Calibration tool for measuring table height.

This script measures the table height for safety workspace limits.

Workflow:
1. Launch robot in manual mode or use existing interface
2. Manually move TCP to touch table surface
3. Press Enter
4. Script reads current TCP Z-coordinate
5. Saves to config/safety_params.yaml
"""

import sys
import time
import yaml
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    ROS2_AVAILABLE = True
except ImportError:
    print("ERROR: ROS2 not available. Please install ROS2 Humble.")
    sys.exit(1)

try:
    from kinova_teleoperation.modules.motion_planner import MotionPlanner
    MOTION_PLANNER_AVAILABLE = True
except ImportError:
    print("Warning: MotionPlanner not available. Will use manual Z input.")
    MOTION_PLANNER_AVAILABLE = False


class TableCalibrationNode(Node):
    """ROS2 node for table calibration."""

    def __init__(self, robot_name: str = "my_gen3"):
        """Initialize calibration node.

        Args:
            robot_name: ROS2 robot namespace
        """
        super().__init__('table_calibration_node')

        self.robot_name = robot_name
        self.current_joints = None
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4',
            'joint_5', 'joint_6', 'joint_7'
        ]

        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            f'/{robot_name}/joint_states',
            self._joint_state_callback,
            10
        )

        self.get_logger().info(f"Waiting for joint states from '{robot_name}'...")

    def _joint_state_callback(self, msg: JointState) -> None:
        """Callback for joint state updates."""
        joints = np.zeros(7)
        for i, joint_name in enumerate(self.joint_names):
            try:
                idx = msg.name.index(joint_name)
                joints[i] = msg.position[idx]
            except (ValueError, IndexError):
                pass

        self.current_joints = joints

    def is_ready(self) -> bool:
        """Check if joint states are being received."""
        return self.current_joints is not None

    def get_current_joints(self) -> np.ndarray:
        """Get current joint positions."""
        return self.current_joints


def calibrate_with_fk(calibration_node: TableCalibrationNode, urdf_path: str) -> float:
    """Calibrate table height using forward kinematics.

    Args:
        calibration_node: Calibration node instance
        urdf_path: Path to robot URDF

    Returns:
        Table height in meters (Z coordinate)
    """
    if not MOTION_PLANNER_AVAILABLE:
        print("ERROR: MotionPlanner not available for FK computation.")
        return None

    # Create motion planner for FK
    planner = MotionPlanner(
        urdf_path=urdf_path,
        base_link="base_link",
        end_effector_link="end_effector_link"
    )

    # Get current joint positions
    joints = calibration_node.get_current_joints()

    # Compute FK
    fk_result = planner.forward_kinematics(joints)

    if fk_result is None:
        print("ERROR: Forward kinematics computation failed.")
        return None

    position, rotation = fk_result

    print(f"\nCurrent end-effector position:")
    print(f"  X: {position[0]:.4f} m")
    print(f"  Y: {position[1]:.4f} m")
    print(f"  Z: {position[2]:.4f} m")

    return position[2]


def calibrate_manual() -> float:
    """Calibrate table height with manual input.

    Returns:
        Table height in meters
    """
    print("\nManual calibration mode.")
    print("Please measure the table height manually and enter it below.")
    print("(Measure from robot base to table surface)")

    while True:
        try:
            z_height = float(input("\nEnter table Z-height (meters): "))
            if z_height < 0 or z_height > 2.0:
                print("Invalid height. Please enter a value between 0 and 2.0 meters.")
                continue
            return z_height
        except ValueError:
            print("Invalid input. Please enter a number.")


def save_config(table_height: float, config_dir: Path) -> None:
    """Save calibration to safety_params.yaml.

    Args:
        table_height: Measured table height (meters)
        config_dir: Configuration directory
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "safety_params.yaml"

    # Default safety configuration
    config = {
        'safety': {
            'table_height_z': float(table_height),
            'z_safety_margin': 0.03,  # 3cm margin above table
            'workspace_radius_xy': 0.6,  # 60cm radius
            'workspace_z_max': 1.0,  # 1m maximum height
            'vision_latency_max': 0.2,  # 200ms
            'ik_failure_threshold': 5,
            'joint_error_max': 0.2,  # 0.2 radians
            'robot_base_x': 0.0,
            'robot_base_y': 0.0,
            'robot_base_z': 0.0,
        }
    }

    # Save to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Configuration saved to: {config_path}")
    print(f"\nSafety parameters:")
    print(f"  Table height: {table_height:.4f} m")
    print(f"  Safety margin: {config['safety']['z_safety_margin']:.4f} m")
    print(f"  Minimum safe Z: {table_height + config['safety']['z_safety_margin']:.4f} m")
    print(f"  XY workspace radius: {config['safety']['workspace_radius_xy']:.2f} m")


def main():
    """Main calibration procedure."""
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate table height for safety limits")
    parser.add_argument('--robot-name', type=str, default='my_gen3',
                        help='ROS2 robot namespace')
    parser.add_argument('--urdf', type=str, default=None,
                        help='Path to robot URDF file (for FK computation)')
    parser.add_argument('--manual', action='store_true',
                        help='Use manual calibration mode (no robot required)')
    parser.add_argument('--config-dir', type=str,
                        default='../config',
                        help='Configuration directory')

    args = parser.parse_args()

    # Resolve config directory
    config_dir = Path(__file__).parent.parent / args.config_dir
    config_dir = config_dir.resolve()

    print("\n" + "="*60)
    print("KINOVA GEN3 TABLE HEIGHT CALIBRATION")
    print("="*60)

    if args.manual:
        # Manual calibration
        table_height = calibrate_manual()

    else:
        # Automatic calibration with robot
        print("\nInitializing ROS2...")
        rclpy.init()

        calibration_node = TableCalibrationNode(robot_name=args.robot_name)

        print("Waiting for robot connection...")
        timeout = 10.0
        start_time = time.time()

        while not calibration_node.is_ready():
            rclpy.spin_once(calibration_node, timeout_sec=0.1)

            if time.time() - start_time > timeout:
                print("\nERROR: Timeout waiting for robot. Is the robot running?")
                print(f"Expected topic: /{args.robot_name}/joint_states")
                rclpy.shutdown()
                sys.exit(1)

        print("✓ Robot connected!\n")

        print("="*60)
        print("CALIBRATION PROCEDURE")
        print("="*60)
        print("1. Manually move the robot's end-effector to touch the table surface")
        print("2. Ensure the TCP is firmly touching the table")
        print("3. Press Enter when ready to capture the position")
        print("="*60)

        input("\nPress Enter when the TCP is touching the table surface...")

        # Spin once more to get latest joint state
        rclpy.spin_once(calibration_node, timeout_sec=0.1)

        # Get table height
        if args.urdf and MOTION_PLANNER_AVAILABLE:
            table_height = calibrate_with_fk(calibration_node, args.urdf)
            if table_height is None:
                print("\nFalling back to manual calibration...")
                table_height = calibrate_manual()
        else:
            print("\nNo URDF provided. Using manual calibration...")
            table_height = calibrate_manual()

        rclpy.shutdown()

    # Save configuration
    print("\n" + "="*60)
    save_config(table_height, config_dir)
    print("="*60)

    print("\n✓ Calibration complete!")
    print(f"\nYou can now run the teleoperation system with:")
    print(f"  --config {config_dir / 'safety_params.yaml'}")


if __name__ == "__main__":
    main()

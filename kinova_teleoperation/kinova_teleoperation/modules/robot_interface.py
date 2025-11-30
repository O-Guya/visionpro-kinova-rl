"""ROS2 Robot Interface for Kinova Gen3.

Handles communication with Kinova Gen3 via ROS2 kortex drivers.
"""

import numpy as np
from typing import Dict, Optional, Any
import time
from scipy.spatial.transform import Rotation
import threading


try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from control_msgs.action import GripperCommand
    from builtin_interfaces.msg import Duration
    ROS2_AVAILABLE = True
except ImportError:
    print("Warning: ROS2 not available. Robot interface will run in simulation mode.")
    ROS2_AVAILABLE = False
    Node = object


class KinovaRobotInterface(Node if ROS2_AVAILABLE else object):
    """ROS2 interface for Kinova Gen3 robot.

    Features:
        - Joint state subscription
        - Joint trajectory command publishing
        - Gripper control via action
        - Thread-safe state access
        - Hold position mode
    """

    def __init__(
        self,
        robot_name: str = "my_gen3",
        joint_names: Optional[list] = None,
        simulation_mode: bool = False
    ):
        """Initialize the robot interface.

        Args:
            robot_name: Robot namespace (e.g., 'my_gen3')
            joint_names: List of joint names
            simulation_mode: If True, runs without ROS2
        """
        self.robot_name = robot_name
        self.simulation_mode = simulation_mode or not ROS2_AVAILABLE

        # Default joint names for Kinova Gen3 7DOF
        if joint_names is None:
            self.joint_names = [
                'joint_1', 'joint_2', 'joint_3', 'joint_4',
                'joint_5', 'joint_6', 'joint_7'
            ]
        else:
            self.joint_names = joint_names

        self.num_joints = len(self.joint_names)

        # Robot state (thread-safe)
        self._state_lock = threading.Lock()
        self._current_joints = np.zeros(self.num_joints)
        self._current_velocities = np.zeros(self.num_joints)
        self._current_ee_pose = np.zeros(7)  # [x, y, z, qx, qy, qz, qw]
        self._last_state_timestamp = None

        # Gripper state
        self._current_gripper_position = 0.0

        if not self.simulation_mode:
            # Initialize ROS2 node
            super().__init__('kinova_teleoperation_interface')

            # Subscribers
            self.joint_state_sub = self.create_subscription(
                JointState,
                f'/{robot_name}/joint_states',
                self._joint_state_callback,
                10
            )

            # Publishers
            self.trajectory_pub = self.create_publisher(
                JointTrajectory,
                f'/{robot_name}/joint_trajectory_controller/joint_trajectory',
                10
            )

            # Gripper action client
            self.gripper_client = ActionClient(
                self,
                GripperCommand,
                f'/{robot_name}/robotiq_gripper_controller/gripper_cmd'
            )

            self.get_logger().info(f"Kinova interface initialized for '{robot_name}'")

        else:
            print(f"[KinovaRobotInterface] Running in SIMULATION mode")

    def _joint_state_callback(self, msg: 'JointState') -> None:
        """Callback for joint state updates.

        Args:
            msg: JointState message
        """
        with self._state_lock:
            # Map joint names to positions
            for i, joint_name in enumerate(self.joint_names):
                try:
                    idx = msg.name.index(joint_name)
                    self._current_joints[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        self._current_velocities[i] = msg.velocity[idx]
                except (ValueError, IndexError):
                    pass

            self._last_state_timestamp = time.time()

    def get_state(self) -> Dict[str, Any]:
        """Get current robot state.

        Returns:
            Dictionary with keys:
                - 'joints': Joint positions (7D)
                - 'velocities': Joint velocities (7D)
                - 'ee_pose': End-effector pose [x,y,z,qx,qy,qz,qw] (7D)
                - 'timestamp': Last update timestamp
                - 'gripper_position': Gripper position (0.0 to 1.0)
        """
        with self._state_lock:
            return {
                'joints': self._current_joints.copy(),
                'velocities': self._current_velocities.copy(),
                'ee_pose': self._current_ee_pose.copy(),
                'timestamp': self._last_state_timestamp,
                'gripper_position': self._current_gripper_position,
            }

    def send_trajectory(self, trajectory: 'JointTrajectory') -> bool:
        """Send joint trajectory command.

        Args:
            trajectory: JointTrajectory message

        Returns:
            True if sent successfully
        """
        if self.simulation_mode:
            # Simulate trajectory execution
            if len(trajectory.points) > 0:
                with self._state_lock:
                    self._current_joints = np.array(trajectory.points[0].positions)
            return True

        try:
            # Set joint names
            trajectory.joint_names = self.joint_names

            # Set header timestamp
            trajectory.header.stamp = self.get_clock().now().to_msg()

            # Publish
            self.trajectory_pub.publish(trajectory)
            return True

        except Exception as e:
            self.get_logger().error(f"Error sending trajectory: {e}")
            return False

    def send_gripper(self, position: float, max_effort: float = 100.0) -> bool:
        """Send gripper command.

        Args:
            position: Gripper position (0.0=open, 1.0=closed)
            max_effort: Maximum gripper effort

        Returns:
            True if command sent successfully
        """
        if self.simulation_mode:
            self._current_gripper_position = position
            return True

        try:
            # Robotiq gripper: 0.0 = open, 0.8 = closed (max 0.8m)
            # Scale our 0-1 to 0-0.8
            gripper_position_m = position * 0.8

            goal_msg = GripperCommand.Goal()
            goal_msg.command.position = gripper_position_m
            goal_msg.command.max_effort = max_effort

            # Send goal (async)
            self.gripper_client.send_goal_async(goal_msg)

            self._current_gripper_position = position
            return True

        except Exception as e:
            self.get_logger().error(f"Error sending gripper command: {e}")
            return False

    def hold_position(self) -> bool:
        """Command robot to hold current position.

        Returns:
            True if command sent successfully
        """
        state = self.get_state()
        current_joints = state['joints']

        # Create single-point trajectory at current position
        if not self.simulation_mode and ROS2_AVAILABLE:
            trajectory = JointTrajectory()
            trajectory.joint_names = self.joint_names

            point = JointTrajectoryPoint()
            point.positions = current_joints.tolist()
            point.velocities = [0.0] * self.num_joints
            point.time_from_start = Duration(sec=0, nanosec=100_000_000)  # 100ms

            trajectory.points.append(point)

            return self.send_trajectory(trajectory)
        else:
            return True

    def update_ee_pose_from_fk(self, fk_solver) -> None:
        """Update end-effector pose using forward kinematics.

        Args:
            fk_solver: Forward kinematics solver (from MotionPlanner)
        """
        state = self.get_state()
        joints = state['joints']

        # Compute FK
        fk_result = fk_solver.forward_kinematics(joints)

        if fk_result is not None:
            position, rotation = fk_result

            # Convert rotation matrix to quaternion
            rot_scipy = Rotation.from_matrix(rotation)
            quat = rot_scipy.as_quat()  # [x, y, z, w]

            with self._state_lock:
                self._current_ee_pose = np.array([
                    position[0], position[1], position[2],
                    quat[0], quat[1], quat[2], quat[3]
                ])

    def get_joint_position_error(self, commanded_joints: np.ndarray) -> float:
        """Compute maximum joint position error.

        Args:
            commanded_joints: Commanded joint positions (7D)

        Returns:
            Maximum absolute joint error (radians)
        """
        state = self.get_state()
        current_joints = state['joints']

        error = np.abs(current_joints - commanded_joints)
        return np.max(error)

    def is_ready(self) -> bool:
        """Check if robot interface is ready.

        Returns:
            True if receiving joint states
        """
        if self.simulation_mode:
            return True

        state = self.get_state()
        if state['timestamp'] is None:
            return False

        # Check if data is recent (within 1 second)
        age = time.time() - state['timestamp']
        return age < 1.0

    def shutdown(self) -> None:
        """Clean shutdown of robot interface."""
        if not self.simulation_mode:
            # Send hold command before shutdown
            self.hold_position()
            self.get_logger().info("Robot interface shutting down")


def spin_ros_node(interface: KinovaRobotInterface) -> None:
    """Spin ROS2 node in background thread.

    Args:
        interface: KinovaRobotInterface instance
    """
    if ROS2_AVAILABLE and not interface.simulation_mode:
        rclpy.spin(interface)


if __name__ == "__main__":
    # Test robot interface
    print("Testing KinovaRobotInterface...")

    if ROS2_AVAILABLE:
        rclpy.init()

    interface = KinovaRobotInterface(simulation_mode=True)

    print(f"\nInterface configuration:")
    print(f"  Robot name: {interface.robot_name}")
    print(f"  Joint names: {interface.joint_names}")
    print(f"  Simulation mode: {interface.simulation_mode}")

    # Test getting state
    state = interface.get_state()
    print(f"\nCurrent state:")
    print(f"  Joints: {state['joints']}")
    print(f"  Gripper: {state['gripper_position']}")

    # Test hold position
    print(f"\nTesting hold position...")
    interface.hold_position()

    # Test gripper
    print(f"Testing gripper command...")
    interface.send_gripper(0.5)

    print("\nTest complete!")

    if ROS2_AVAILABLE:
        rclpy.shutdown()

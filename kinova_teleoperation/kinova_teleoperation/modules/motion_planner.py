"""Motion Planner for IK solving and trajectory generation.

Implements inverse kinematics with KDL and predictive trajectory extrapolation.
"""

import numpy as np
from typing import Optional, List, Tuple
from scipy.spatial.transform import Rotation
import time


try:
    from kdl_parser_py.urdf import treeFromUrdfModel
    from urdf_parser_py.urdf import URDF
    import PyKDL as kdl
    KDL_AVAILABLE = True
except ImportError:
    print("Warning: KDL not available. Install: pip install kdl_parser_py urdf_parser_py PyKDL")
    KDL_AVAILABLE = False


# Check if ROS2 messages are available
try:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    ROS2_AVAILABLE = True
except ImportError:
    print("Warning: ROS2 trajectory messages not available")
    ROS2_AVAILABLE = False


class MotionPlanner:
    """IK solver and trajectory generator for Kinova Gen3.

    Features:
        - KDL-based inverse kinematics
        - Warm-start IK with current joint positions
        - Predictive trajectory extrapolation (3-point window)
        - Velocity clamping for safety
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        base_link: str = "base_link",
        end_effector_link: str = "end_effector_link",
        joint_names: Optional[List[str]] = None,
        max_ik_iterations: int = 20,
        position_tolerance: float = 0.001,  # 1mm
    ):
        """Initialize the motion planner.

        Args:
            urdf_path: Path to robot URDF file
            base_link: Name of base link
            end_effector_link: Name of end-effector link
            joint_names: List of joint names (in order)
            max_ik_iterations: Maximum IK solver iterations
            position_tolerance: Position error tolerance (meters)
        """
        self.base_link = base_link
        self.end_effector_link = end_effector_link
        self.max_ik_iterations = max_ik_iterations
        self.position_tolerance = position_tolerance

        # Default Kinova Gen3 7DOF joint names
        if joint_names is None:
            self.joint_names = [
                'joint_1', 'joint_2', 'joint_3', 'joint_4',
                'joint_5', 'joint_6', 'joint_7'
            ]
        else:
            self.joint_names = joint_names

        self.num_joints = len(self.joint_names)

        # KDL chain and solvers
        self.chain: Optional[kdl.Chain] = None
        self.fk_solver: Optional[kdl.ChainFkSolverPos_recursive] = None
        self.ik_vel_solver: Optional[kdl.ChainIkSolverVel_pinv] = None
        self.ik_solver: Optional[kdl.ChainIkSolverPos_NR] = None

        # Trajectory extrapolation parameters
        self.extrapolation_times = [0.05, 0.10, 0.15]  # seconds
        self.max_velocity = 0.5  # m/s

        # Statistics
        self.stats = {
            'ik_attempts': 0,
            'ik_successes': 0,
            'ik_failures': 0,
        }

        # Initialize KDL
        if urdf_path is not None and KDL_AVAILABLE:
            self._initialize_kdl(urdf_path)
        elif not KDL_AVAILABLE:
            print("[MotionPlanner] Warning: KDL not available. IK disabled.")
        else:
            print("[MotionPlanner] No URDF provided. IK disabled.")

    def _initialize_kdl(self, urdf_path: str) -> None:
        """Initialize KDL chain and solvers from URDF.

        Args:
            urdf_path: Path to URDF file
        """
        try:
            # Load URDF
            robot = URDF.from_xml_file(urdf_path)
            ok, tree = treeFromUrdfModel(robot)

            if not ok:
                print(f"[MotionPlanner] Failed to construct KDL tree from URDF")
                return

            # Extract chain
            self.chain = tree.getChain(self.base_link, self.end_effector_link)

            print(f"[MotionPlanner] KDL chain initialized:")
            print(f"  Segments: {self.chain.getNrOfSegments()}")
            print(f"  Joints: {self.chain.getNrOfJoints()}")

            # Create solvers
            self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
            self.ik_vel_solver = kdl.ChainIkSolverVel_pinv(self.chain)
            self.ik_solver = kdl.ChainIkSolverPos_NR(
                self.chain,
                self.fk_solver,
                self.ik_vel_solver,
                self.max_ik_iterations,
                self.position_tolerance
            )

            print(f"[MotionPlanner] IK solver ready.")

        except Exception as e:
            print(f"[MotionPlanner] Error initializing KDL: {e}")
            self.chain = None

    def solve_ik(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        seed_joints: np.ndarray
    ) -> Optional[np.ndarray]:
        """Solve inverse kinematics for target pose.

        Args:
            target_position: Target position [x, y, z] in meters
            target_rotation: Target rotation as 3x3 matrix
            seed_joints: Seed joint angles for warm start (7D array)

        Returns:
            Joint angles (7D array) or None if IK fails
        """
        if self.ik_solver is None:
            print("[MotionPlanner] IK solver not initialized.")
            return None

        self.stats['ik_attempts'] += 1

        try:
            # Convert target pose to KDL Frame
            kdl_rotation = kdl.Rotation(
                target_rotation[0, 0], target_rotation[0, 1], target_rotation[0, 2],
                target_rotation[1, 0], target_rotation[1, 1], target_rotation[1, 2],
                target_rotation[2, 0], target_rotation[2, 1], target_rotation[2, 2]
            )
            kdl_position = kdl.Vector(
                target_position[0],
                target_position[1],
                target_position[2]
            )
            target_frame = kdl.Frame(kdl_rotation, kdl_position)

            # Convert seed to KDL JntArray
            seed_array = kdl.JntArray(self.num_joints)
            for i in range(self.num_joints):
                seed_array[i] = seed_joints[i]

            # Solve IK
            result_array = kdl.JntArray(self.num_joints)
            ret = self.ik_solver.CartToJnt(seed_array, target_frame, result_array)

            if ret >= 0:  # Success
                # Convert back to numpy
                joint_angles = np.array([result_array[i] for i in range(self.num_joints)])
                self.stats['ik_successes'] += 1
                return joint_angles
            else:
                self.stats['ik_failures'] += 1
                return None

        except Exception as e:
            print(f"[MotionPlanner] IK error: {e}")
            self.stats['ik_failures'] += 1
            return None

    def forward_kinematics(self, joint_angles: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Compute forward kinematics.

        Args:
            joint_angles: Joint angles (7D array)

        Returns:
            (position, rotation) tuple or None if FK fails
        """
        if self.fk_solver is None:
            return None

        try:
            # Convert to KDL JntArray
            jnt_array = kdl.JntArray(self.num_joints)
            for i in range(self.num_joints):
                jnt_array[i] = joint_angles[i]

            # Compute FK
            end_frame = kdl.Frame()
            ret = self.fk_solver.JntToCart(jnt_array, end_frame)

            if ret >= 0:
                # Extract position
                position = np.array([
                    end_frame.p.x(),
                    end_frame.p.y(),
                    end_frame.p.z()
                ])

                # Extract rotation
                rotation = np.array([
                    [end_frame.M[0, 0], end_frame.M[0, 1], end_frame.M[0, 2]],
                    [end_frame.M[1, 0], end_frame.M[1, 1], end_frame.M[1, 2]],
                    [end_frame.M[2, 0], end_frame.M[2, 1], end_frame.M[2, 2]]
                ])

                return position, rotation
            else:
                return None

        except Exception as e:
            print(f"[MotionPlanner] FK error: {e}")
            return None

    def generate_trajectory_window(
        self,
        target_joint_positions: np.ndarray,
        hand_velocity: np.ndarray,
        current_joints: np.ndarray,
        use_extrapolation: bool = True
    ) -> Optional['JointTrajectory']:
        """Generate trajectory with predictive extrapolation.

        Args:
            target_joint_positions: Target joint positions (7D)
            hand_velocity: Hand velocity in Cartesian space [vx, vy, vz] m/s
            current_joints: Current joint positions (7D)
            use_extrapolation: If True, generate 3-point window

        Returns:
            JointTrajectory message or None
        """
        if not ROS2_AVAILABLE:
            print("[MotionPlanner] ROS2 not available. Cannot create trajectory.")
            return None

        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        # Point 1: Current target (t+50ms)
        point1 = JointTrajectoryPoint()
        point1.positions = target_joint_positions.tolist()
        point1.time_from_start = Duration(sec=0, nanosec=50_000_000)  # 50ms

        # Compute velocities via finite difference
        dt = 0.05
        velocities = (target_joint_positions - current_joints) / dt
        point1.velocities = velocities.tolist()

        trajectory.points.append(point1)

        if use_extrapolation and self.fk_solver is not None:
            # Extrapolate future points using hand velocity
            for i, t_future in enumerate([0.10, 0.15], start=2):
                # Clamp velocity
                vel_clamped = np.clip(hand_velocity, -self.max_velocity, self.max_velocity)

                # Get current end-effector pose
                fk_result = self.forward_kinematics(target_joint_positions)
                if fk_result is None:
                    break

                ee_pos, ee_rot = fk_result

                # Extrapolate position
                delta_t = t_future - 0.05
                extrapolated_pos = ee_pos + vel_clamped * delta_t

                # Solve IK for extrapolated pose
                extrapolated_joints = self.solve_ik(
                    extrapolated_pos,
                    ee_rot,
                    seed_joints=target_joint_positions
                )

                if extrapolated_joints is not None:
                    point = JointTrajectoryPoint()
                    point.positions = extrapolated_joints.tolist()
                    point.velocities = velocities.tolist()  # Reuse velocities
                    point.time_from_start = Duration(
                        sec=0,
                        nanosec=int(t_future * 1e9)
                    )
                    trajectory.points.append(point)
                else:
                    break  # Stop if extrapolation IK fails

        return trajectory

    def get_success_rate(self) -> float:
        """Get IK success rate.

        Returns:
            Success rate (0.0 to 1.0)
        """
        if self.stats['ik_attempts'] == 0:
            return 0.0
        return self.stats['ik_successes'] / self.stats['ik_attempts']

    def print_statistics(self) -> None:
        """Print IK statistics."""
        print("\n[MotionPlanner] Statistics:")
        print(f"  IK attempts: {self.stats['ik_attempts']}")
        print(f"  IK successes: {self.stats['ik_successes']}")
        print(f"  IK failures: {self.stats['ik_failures']}")
        print(f"  Success rate: {self.get_success_rate():.2%}")


if __name__ == "__main__":
    # Test the motion planner (requires URDF)
    print("Testing MotionPlanner...")

    # This is a minimal test - full test requires actual URDF
    planner = MotionPlanner()

    print("\nTest configuration:")
    print(f"  Joint names: {planner.joint_names}")
    print(f"  Max IK iterations: {planner.max_ik_iterations}")
    print(f"  Position tolerance: {planner.position_tolerance}m")

    print("\nNote: Full IK testing requires URDF file.")
    print("Test complete!")

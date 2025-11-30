"""Main Control Loop - 20Hz System Orchestrator.

Integrates all modules for real-time teleoperation of Kinova Gen3.
"""

import numpy as np
import time
from typing import Optional
from dataclasses import dataclass
from collections import deque
import sys
import signal
import threading

# Import VisionPro streamer
try:
    sys.path.append('/home/user/visionpro-kinova-rl/VisionProTeleop')
    from avp_stream import VisionProStreamer
    VISION_PRO_AVAILABLE = True
except ImportError:
    print("Warning: VisionProStreamer not available")
    VISION_PRO_AVAILABLE = False

# Import ROS2
try:
    import rclpy
    ROS2_AVAILABLE = True
except ImportError:
    print("Warning: ROS2 not available")
    ROS2_AVAILABLE = False

# Import our modules
from .modules.reference_frame_manager import ReferenceFrameManager
from .modules.input_aggregator import InputAggregator
from .modules.safety_monitor import SafetyMonitor, SafetyConfig
from .modules.motion_planner import MotionPlanner
from .modules.data_logger import DataLogger
from .modules.robot_interface import KinovaRobotInterface


@dataclass
class ControlState:
    """State machine for teleoperation control."""
    anchor_hand_pose: Optional[np.ndarray] = None
    anchor_robot_pose: Optional[np.ndarray] = None
    is_clutched: bool = False
    ik_success_history: deque = None

    def __post_init__(self):
        if self.ik_success_history is None:
            self.ik_success_history = deque(maxlen=10)


class TeleoperationSystem:
    """Complete teleoperation system for Kinova Gen3 with Vision Pro.

    Orchestrates all modules at 20Hz for real-time control.
    """

    def __init__(
        self,
        vision_pro_ip: str,
        robot_name: str = "my_gen3",
        config_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        simulation_mode: bool = False,
        enable_data_logging: bool = True,
    ):
        """Initialize the teleoperation system.

        Args:
            vision_pro_ip: IP address of Vision Pro device
            robot_name: ROS2 robot namespace
            config_path: Path to safety configuration YAML
            urdf_path: Path to robot URDF file
            simulation_mode: If True, runs without real robot
            enable_data_logging: If True, enables demonstration recording
        """
        self.vision_pro_ip = vision_pro_ip
        self.robot_name = robot_name
        self.simulation_mode = simulation_mode
        self.enable_data_logging = enable_data_logging

        # Control loop parameters
        self.control_rate_hz = 20.0
        self.control_period = 1.0 / self.control_rate_hz

        # Modules
        self.vision_pro: Optional[VisionProStreamer] = None
        self.reference_frame_mgr: Optional[ReferenceFrameManager] = None
        self.input_aggregator: Optional[InputAggregator] = None
        self.safety_monitor: Optional[SafetyMonitor] = None
        self.motion_planner: Optional[MotionPlanner] = None
        self.data_logger: Optional[DataLogger] = None
        self.robot_interface: Optional[KinovaRobotInterface] = None

        # Control state
        self.state = ControlState()

        # Scaling factors
        self.scaling_factors = {'x': 1.5, 'y': 1.5, 'z': 1.0}

        # Running flag
        self._running = False
        self._ros_thread: Optional[threading.Thread] = None

        # Initialize all modules
        self._initialize_modules(config_path, urdf_path)

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("\n" + "="*60)
        print("KINOVA GEN3 TELEOPERATION SYSTEM")
        print("="*60)
        print(f"Vision Pro IP: {vision_pro_ip}")
        print(f"Robot: {robot_name}")
        print(f"Control Rate: {self.control_rate_hz} Hz")
        print(f"Simulation Mode: {simulation_mode}")
        print("="*60 + "\n")

    def _initialize_modules(self, config_path: Optional[str], urdf_path: Optional[str]) -> None:
        """Initialize all system modules.

        Args:
            config_path: Path to safety config
            urdf_path: Path to URDF file
        """
        print("[TeleoperationSystem] Initializing modules...")

        # 1. Vision Pro Streamer
        if VISION_PRO_AVAILABLE:
            print("  - Connecting to Vision Pro...")
            self.vision_pro = VisionProStreamer(ip=self.vision_pro_ip, record=False)
        else:
            print("  - Vision Pro streamer not available (will use dummy data)")

        # 2. Reference Frame Manager
        print("  - Initializing reference frame manager...")
        self.reference_frame_mgr = ReferenceFrameManager(
            min_cutoff=1.0,
            beta=0.05,
            d_cutoff=1.0
        )

        # 3. Input Aggregator
        print("  - Initializing gamepad input...")
        self.input_aggregator = InputAggregator(deadband=0.1, fast_mode=True)

        # 4. Safety Monitor
        print("  - Initializing safety monitor...")
        if config_path:
            self.safety_monitor = SafetyMonitor(config_path=config_path)
        else:
            print("    Warning: No config path provided. Using default safety config.")
            self.safety_monitor = SafetyMonitor()

        # 5. Motion Planner
        print("  - Initializing motion planner...")
        self.motion_planner = MotionPlanner(
            urdf_path=urdf_path,
            base_link="base_link",
            end_effector_link="end_effector_link"
        )

        # 6. Data Logger
        if self.enable_data_logging:
            print("  - Initializing data logger...")
            self.data_logger = DataLogger(
                output_dir="./demonstrations",
                buffer_size=50
            )

        # 7. Robot Interface
        print("  - Initializing robot interface...")
        if ROS2_AVAILABLE and not self.simulation_mode:
            rclpy.init()

        self.robot_interface = KinovaRobotInterface(
            robot_name=self.robot_name,
            simulation_mode=self.simulation_mode
        )

        # Start ROS2 spinning in background thread
        if ROS2_AVAILABLE and not self.simulation_mode:
            self._ros_thread = threading.Thread(
                target=rclpy.spin,
                args=(self.robot_interface,),
                daemon=True
            )
            self._ros_thread.start()

        print("[TeleoperationSystem] All modules initialized!\n")

    def calibrate_world_frame(self) -> None:
        """Calibrate the world reference frame using current head pose."""
        if self.vision_pro is None:
            print("[TeleoperationSystem] Vision Pro not available for calibration")
            return

        print("[TeleoperationSystem] Calibrating world frame...")
        print("  Please hold your head still and press Enter...")
        input()

        latest = self.vision_pro.get_latest()
        if latest is None:
            print("  Error: No Vision Pro data available")
            return

        head_pose = latest['head']
        self.reference_frame_mgr.calibrate_world_frame(head_pose)

        print("  World frame calibrated successfully!\n")

    def run(self) -> None:
        """Main control loop at 20Hz."""
        if not self.reference_frame_mgr.is_calibrated():
            print("\n[TeleoperationSystem] World frame not calibrated!")
            self.calibrate_world_frame()

        # Wait for robot to be ready
        if not self.simulation_mode:
            print("[TeleoperationSystem] Waiting for robot connection...")
            while not self.robot_interface.is_ready():
                time.sleep(0.1)
            print("  Robot connected!\n")

        print("\n" + "="*60)
        print("TELEOPERATION CONTROLS")
        print("="*60)
        print("  A Button:      Engage clutch (hold to control robot)")
        print("  Y Button:      Toggle scaling mode (fast/precision)")
        print("  Right Trigger: Gripper control (0=open, 1=closed)")
        print("  B Button:      Emergency stop")
        print("  Ctrl+C:        Shutdown system")
        print("="*60 + "\n")

        print("[TeleoperationSystem] Starting main loop at 20Hz...\n")

        self._running = True
        loop_count = 0

        try:
            while self._running:
                loop_start = time.time()

                # Run one control iteration
                self._control_iteration()

                # Timing control
                loop_count += 1
                if loop_count % 100 == 0:
                    elapsed = time.time() - loop_start
                    if elapsed > self.control_period:
                        print(f"⚠ Loop overrun: {elapsed*1000:.1f}ms (target: {self.control_period*1000:.1f}ms)")

                # Sleep to maintain rate
                elapsed = time.time() - loop_start
                sleep_time = self.control_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[TeleoperationSystem] Received Ctrl+C")

        finally:
            self.shutdown()

    def _control_iteration(self) -> None:
        """Single iteration of control loop."""

        # === 1. SENSE ===
        # Get Vision Pro data
        if self.vision_pro:
            latest = self.vision_pro.get_latest()
            if latest is None:
                return

            head_pose = latest['head']
            hand_pose_raw = latest['right_wrist']
        else:
            # Dummy data for testing
            head_pose = np.eye(4)
            hand_pose_raw = np.eye(4)

        # Transform and filter
        hand_filtered, hand_rot = self.reference_frame_mgr.get_filtered_hand_pose(
            head_pose, hand_pose_raw
        )
        hand_velocity = self.reference_frame_mgr.get_hand_velocity()

        # Get gamepad input
        gamepad_state = self.input_aggregator.get_state()

        # Get robot state
        robot_state = self.robot_interface.get_state()

        # === 2. MODE SWITCHING ===
        if gamepad_state.button_y_just_pressed:
            self.scaling_factors = self.input_aggregator.get_scaling_factors()

        # === 3. SAFETY CHECK ===
        vision_timestamp = self.reference_frame_mgr.get_last_timestamp()
        system_healthy = self.safety_monitor.check_system_health(
            vision_timestamp=vision_timestamp,
            ik_success_history=list(self.state.ik_success_history),
            joint_error=0.0  # TODO: Compute from robot state
        )

        if not system_healthy:
            if self.state.is_clutched:
                print(f"⚠ SAFETY VIOLATION: {self.safety_monitor.get_violation_reason()}")
                self.state.is_clutched = False
                self.robot_interface.hold_position()
            return

        # === 4. CONTROL LOGIC ===
        if gamepad_state.clutch_pressed and system_healthy:

            # CRITICAL: Set anchors ONLY on rising edge
            if gamepad_state.clutch_just_pressed:
                self.state.anchor_hand_pose = hand_filtered.copy()
                # Use current end-effector position as anchor
                self.state.anchor_robot_pose = robot_state['ee_pose'][:3].copy()
                self.state.is_clutched = True
                print(">>> CLUTCH ENGAGED <<<")

            # Compute relative target
            delta_hand = hand_filtered - self.state.anchor_hand_pose
            delta_scaled = np.array([
                delta_hand[0] * self.scaling_factors['x'],
                delta_hand[1] * self.scaling_factors['y'],
                delta_hand[2] * self.scaling_factors['z']
            ])

            target_position = self.state.anchor_robot_pose + delta_scaled

            # Apply safety constraints
            safe_position, _ = self.safety_monitor.clamp_to_workspace(target_position)

            # IK + Trajectory Generation
            joint_target = self.motion_planner.solve_ik(
                safe_position,
                hand_rot,
                seed_joints=robot_state['joints']
            )

            if joint_target is not None:
                trajectory = self.motion_planner.generate_trajectory_window(
                    joint_target,
                    hand_velocity,
                    robot_state['joints'],
                    use_extrapolation=True
                )

                if trajectory is not None:
                    self.robot_interface.send_trajectory(trajectory)
                    self.state.ik_success_history.append(True)
                else:
                    self.robot_interface.hold_position()
                    self.state.ik_success_history.append(False)
            else:
                self.robot_interface.hold_position()
                self.state.ik_success_history.append(False)
                print("⚠ IK Failed")

            # Gripper Control
            gripper_position = gamepad_state.trigger_val
            self.robot_interface.send_gripper(gripper_position)

            # Data Logging
            if self.enable_data_logging and self.data_logger:
                if not self.data_logger.is_recording():
                    # Auto-start recording on first clutch
                    task_name = f"demo_{int(time.time())}"
                    self.data_logger.start_recording(task_name)

                # Log frame
                self.data_logger.log_frame(
                    robot_state=robot_state,
                    action_delta=delta_scaled,
                )

        else:
            # Clutch disengaged or system unhealthy
            self.robot_interface.hold_position()

            if self.state.is_clutched:
                print(">>> CLUTCH RELEASED <<<")
                self.state.is_clutched = False

                # Stop recording
                if self.enable_data_logging and self.data_logger and self.data_logger.is_recording():
                    filepath = self.data_logger.stop_recording()
                    print(f"  Demonstration saved: {filepath}")

        # Emergency stop
        if gamepad_state.button_b_pressed:
            print("!!! EMERGENCY STOP ACTIVATED !!!")
            self.robot_interface.hold_position()
            self._running = False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n[TeleoperationSystem] Received signal {signum}")
        self._running = False

    def shutdown(self) -> None:
        """Clean shutdown of all modules."""
        print("\n[TeleoperationSystem] Shutting down...")

        self._running = False

        # Stop recording if active
        if self.data_logger and self.data_logger.is_recording():
            self.data_logger.stop_recording()

        # Stop input aggregator
        if self.input_aggregator:
            self.input_aggregator.stop()

        # Shutdown robot interface
        if self.robot_interface:
            self.robot_interface.shutdown()

        # Shutdown ROS2
        if ROS2_AVAILABLE and not self.simulation_mode:
            rclpy.shutdown()

        # Print statistics
        if self.safety_monitor:
            self.safety_monitor.print_statistics()

        if self.motion_planner:
            self.motion_planner.print_statistics()

        print("\n[TeleoperationSystem] Shutdown complete.\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Kinova Gen3 Teleoperation System")
    parser.add_argument('--vision-pro-ip', type=str, required=True,
                        help='Vision Pro IP address')
    parser.add_argument('--robot-name', type=str, default='my_gen3',
                        help='ROS2 robot namespace')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to safety configuration YAML')
    parser.add_argument('--urdf', type=str, default=None,
                        help='Path to robot URDF file')
    parser.add_argument('--sim', action='store_true',
                        help='Run in simulation mode')
    parser.add_argument('--no-logging', action='store_true',
                        help='Disable data logging')

    args = parser.parse_args()

    # Create system
    system = TeleoperationSystem(
        vision_pro_ip=args.vision_pro_ip,
        robot_name=args.robot_name,
        config_path=args.config,
        urdf_path=args.urdf,
        simulation_mode=args.sim,
        enable_data_logging=not args.no_logging,
    )

    # Run main loop
    system.run()


if __name__ == "__main__":
    main()

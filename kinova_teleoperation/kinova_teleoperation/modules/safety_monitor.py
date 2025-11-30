"""Safety Monitor for comprehensive system health monitoring and workspace constraints.

Implements watchdog conditions and workspace limits to ensure safe operation.
"""

import time
import numpy as np
from typing import Dict, Optional, List
from collections import deque
from dataclasses import dataclass
import yaml


@dataclass
class SafetyConfig:
    """Safety configuration parameters."""
    # Watchdog thresholds
    vision_latency_max: float = 0.2  # seconds
    ik_failure_threshold: int = 5  # consecutive failures
    joint_error_max: float = 0.2  # radians
    tracking_confidence_min: str = "High"  # Not implemented yet

    # Workspace limits
    table_height_z: float = 0.0  # meters, from robot base
    z_safety_margin: float = 0.03  # meters above table
    workspace_radius_xy: float = 0.6  # meters from robot base
    workspace_z_max: float = 1.0  # meters from base

    # Robot base position (world frame)
    robot_base_x: float = 0.0
    robot_base_y: float = 0.0
    robot_base_z: float = 0.0


class SafetyMonitor:
    """Comprehensive system health monitoring and workspace constraints.

    Features:
        - Vision latency watchdog
        - IK failure tracking
        - Joint error detection (jam detection)
        - Workspace boundary enforcement
        - Safety state management
    """

    def __init__(self, config: Optional[SafetyConfig] = None, config_path: Optional[str] = None):
        """Initialize the safety monitor.

        Args:
            config: SafetyConfig object
            config_path: Path to YAML config file (overrides config if provided)
        """
        if config_path is not None:
            self.config = self._load_config(config_path)
        elif config is not None:
            self.config = config
        else:
            self.config = SafetyConfig()
            print("[SafetyMonitor] Warning: Using default safety config. Run calibration first!")

        # Watchdog state
        self._ik_failure_history = deque(maxlen=self.config.ik_failure_threshold * 2)
        self._last_vision_timestamp: Optional[float] = None
        self._safety_violated = False
        self._violation_reason = ""

        # Statistics
        self.stats = {
            'total_checks': 0,
            'vision_latency_violations': 0,
            'ik_failure_violations': 0,
            'joint_error_violations': 0,
            'workspace_violations': 0,
        }

        print(f"[SafetyMonitor] Initialized with config:")
        print(f"  Table height: {self.config.table_height_z:.3f}m")
        print(f"  Z safety margin: {self.config.z_safety_margin:.3f}m")
        print(f"  XY workspace radius: {self.config.workspace_radius_xy:.3f}m")

    def _load_config(self, config_path: str) -> SafetyConfig:
        """Load safety configuration from YAML file.

        Args:
            config_path: Path to safety_params.yaml

        Returns:
            SafetyConfig object
        """
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)

            safety_data = data.get('safety', {})

            return SafetyConfig(
                vision_latency_max=safety_data.get('vision_latency_max', 0.2),
                ik_failure_threshold=safety_data.get('ik_failure_threshold', 5),
                joint_error_max=safety_data.get('joint_error_max', 0.2),
                table_height_z=safety_data.get('table_height_z', 0.0),
                z_safety_margin=safety_data.get('z_safety_margin', 0.03),
                workspace_radius_xy=safety_data.get('workspace_radius_xy', 0.6),
                workspace_z_max=safety_data.get('workspace_z_max', 1.0),
                robot_base_x=safety_data.get('robot_base_x', 0.0),
                robot_base_y=safety_data.get('robot_base_y', 0.0),
                robot_base_z=safety_data.get('robot_base_z', 0.0),
            )

        except FileNotFoundError:
            print(f"[SafetyMonitor] Config file not found: {config_path}")
            print("[SafetyMonitor] Using default config.")
            return SafetyConfig()
        except Exception as e:
            print(f"[SafetyMonitor] Error loading config: {e}")
            print("[SafetyMonitor] Using default config.")
            return SafetyConfig()

    def check_system_health(
        self,
        vision_timestamp: Optional[float] = None,
        ik_success_history: Optional[List[bool]] = None,
        joint_error: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> bool:
        """Check overall system health.

        Args:
            vision_timestamp: Timestamp of latest vision data
            ik_success_history: Recent IK success/failure boolean list
            joint_error: Maximum joint position error (radians)
            current_time: Current time (if None, uses time.time())

        Returns:
            True if system is healthy, False if watchdog triggered
        """
        if current_time is None:
            current_time = time.time()

        self.stats['total_checks'] += 1

        # Reset violation state
        self._safety_violated = False
        self._violation_reason = ""

        # Check 1: Vision latency
        if vision_timestamp is not None:
            vision_latency = current_time - vision_timestamp
            if vision_latency > self.config.vision_latency_max:
                self._safety_violated = True
                self._violation_reason = f"Vision latency too high: {vision_latency:.3f}s"
                self.stats['vision_latency_violations'] += 1
                return False

        # Check 2: IK divergence
        if ik_success_history is not None and len(ik_success_history) > 0:
            # Count consecutive failures
            consecutive_failures = 0
            for success in reversed(ik_success_history):
                if not success:
                    consecutive_failures += 1
                else:
                    break

            if consecutive_failures >= self.config.ik_failure_threshold:
                self._safety_violated = True
                self._violation_reason = f"IK failures: {consecutive_failures} consecutive"
                self.stats['ik_failure_violations'] += 1
                return False

        # Check 3: Joint error (jam detection)
        if joint_error is not None:
            if joint_error > self.config.joint_error_max:
                self._safety_violated = True
                self._violation_reason = f"Joint error too high: {joint_error:.3f} rad"
                self.stats['joint_error_violations'] += 1
                return False

        # All checks passed
        return True

    def clamp_to_workspace(
        self,
        target_position: np.ndarray,
        target_rotation: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Clamp target pose to safe workspace.

        Args:
            target_position: Target position [x, y, z] in robot base frame
            target_rotation: Optional target rotation (3x3 matrix)

        Returns:
            (clamped_position, target_rotation) tuple
        """
        position = target_position.copy()
        clamped = False

        # Robot base position in world frame
        base_x = self.config.robot_base_x
        base_y = self.config.robot_base_y
        base_z = self.config.robot_base_z

        # Convert to robot-relative coordinates
        rel_x = position[0] - base_x
        rel_y = position[1] - base_y
        rel_z = position[2] - base_z

        # Check Z lower limit (table height)
        min_z = self.config.table_height_z + self.config.z_safety_margin
        if rel_z < min_z:
            rel_z = min_z
            clamped = True

        # Check Z upper limit
        if rel_z > self.config.workspace_z_max:
            rel_z = self.config.workspace_z_max
            clamped = True

        # Check XY cylindrical workspace
        xy_distance = np.sqrt(rel_x**2 + rel_y**2)
        if xy_distance > self.config.workspace_radius_xy:
            # Clamp to boundary
            scale = self.config.workspace_radius_xy / xy_distance
            rel_x *= scale
            rel_y *= scale
            clamped = True

        # Convert back to world frame
        position[0] = rel_x + base_x
        position[1] = rel_y + base_y
        position[2] = rel_z + base_z

        if clamped:
            self.stats['workspace_violations'] += 1

        return position, target_rotation

    def is_position_safe(self, position: np.ndarray) -> bool:
        """Check if a position is within safe workspace.

        Args:
            position: Position [x, y, z] in robot base frame

        Returns:
            True if position is safe
        """
        clamped_pos, _ = self.clamp_to_workspace(position)
        return np.allclose(position, clamped_pos, atol=1e-6)

    def get_table_height(self) -> float:
        """Get configured table height.

        Returns:
            Table height in meters from robot base
        """
        return self.config.table_height_z

    def get_safe_z_minimum(self) -> float:
        """Get minimum safe Z coordinate.

        Returns:
            Minimum safe Z in meters from robot base
        """
        return self.config.table_height_z + self.config.z_safety_margin

    def is_safety_violated(self) -> bool:
        """Check if safety is currently violated.

        Returns:
            True if watchdog has triggered
        """
        return self._safety_violated

    def get_violation_reason(self) -> str:
        """Get reason for safety violation.

        Returns:
            Human-readable violation reason
        """
        return self._violation_reason

    def reset_watchdog(self) -> None:
        """Reset watchdog state (e.g., after user re-engages clutch)."""
        self._safety_violated = False
        self._violation_reason = ""
        self._ik_failure_history.clear()

    def get_statistics(self) -> Dict:
        """Get safety statistics.

        Returns:
            Dictionary with violation counts
        """
        return self.stats.copy()

    def print_statistics(self) -> None:
        """Print safety statistics to console."""
        print("\n[SafetyMonitor] Statistics:")
        print(f"  Total health checks: {self.stats['total_checks']}")
        print(f"  Vision latency violations: {self.stats['vision_latency_violations']}")
        print(f"  IK failure violations: {self.stats['ik_failure_violations']}")
        print(f"  Joint error violations: {self.stats['joint_error_violations']}")
        print(f"  Workspace violations: {self.stats['workspace_violations']}")


if __name__ == "__main__":
    # Test the safety monitor
    print("Testing SafetyMonitor...")

    config = SafetyConfig(
        table_height_z=0.15,
        z_safety_margin=0.03,
        workspace_radius_xy=0.6,
    )

    monitor = SafetyMonitor(config=config)

    # Test 1: Vision latency
    print("\nTest 1: Vision latency check")
    old_timestamp = time.time() - 0.5
    healthy = monitor.check_system_health(vision_timestamp=old_timestamp)
    print(f"  Result: {'PASS' if not healthy else 'FAIL'}")
    print(f"  Reason: {monitor.get_violation_reason()}")

    # Test 2: IK failures
    print("\nTest 2: IK failure check")
    ik_history = [False, False, False, False, False, False]
    healthy = monitor.check_system_health(ik_success_history=ik_history)
    print(f"  Result: {'PASS' if not healthy else 'FAIL'}")
    print(f"  Reason: {monitor.get_violation_reason()}")

    # Test 3: Workspace clamping
    print("\nTest 3: Workspace clamping")
    test_positions = [
        np.array([0.3, 0.0, 0.5]),   # Safe
        np.array([0.8, 0.0, 0.5]),   # Too far XY
        np.array([0.3, 0.0, 0.05]),  # Below table
    ]

    for pos in test_positions:
        clamped, _ = monitor.clamp_to_workspace(pos)
        print(f"  Original: {pos} -> Clamped: {clamped}")

    monitor.print_statistics()
    print("\nTest complete!")

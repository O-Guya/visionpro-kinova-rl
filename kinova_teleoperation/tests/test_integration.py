#!/usr/bin/env python3
"""Integration test for teleoperation system.

Tests all modules in simulation mode without requiring hardware.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinova_teleoperation.modules.reference_frame_manager import ReferenceFrameManager
from kinova_teleoperation.modules.input_aggregator import InputAggregator, GamepadState
from kinova_teleoperation.modules.safety_monitor import SafetyMonitor, SafetyConfig
from kinova_teleoperation.modules.motion_planner import MotionPlanner
from kinova_teleoperation.modules.data_logger import DataLogger
from kinova_teleoperation.modules.robot_interface import KinovaRobotInterface


class DummyInputAggregator:
    """Dummy input aggregator for testing without gamepad."""

    def __init__(self):
        self._clutch_state = False
        self._trigger = 0.0
        self._mode_toggle_count = 0

    def get_state(self) -> GamepadState:
        """Get simulated gamepad state."""
        state = GamepadState()
        state.clutch_pressed = self._clutch_state
        state.clutch_just_pressed = False
        state.trigger_val = self._trigger
        state.button_y_just_pressed = False
        state.button_b_pressed = False
        state.connected = True
        return state

    def set_clutch(self, pressed: bool):
        """Simulate clutch press."""
        self._clutch_state = pressed

    def set_trigger(self, value: float):
        """Simulate trigger input."""
        self._trigger = value

    def get_scaling_factors(self):
        """Get scaling factors."""
        return {'x': 1.5, 'y': 1.5, 'z': 1.0}

    def stop(self):
        """Stop (no-op for dummy)."""
        pass


def test_reference_frame_manager():
    """Test ReferenceFrameManager."""
    print("\n" + "="*60)
    print("TEST 1: ReferenceFrameManager")
    print("="*60)

    manager = ReferenceFrameManager(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)

    # Calibrate with initial head pose
    H_init = np.eye(4)
    H_init[:3, 3] = [0, 0, 1.5]  # Head at 1.5m height
    manager.calibrate_world_frame(H_init)

    assert manager.is_calibrated(), "Manager should be calibrated"

    # Test transformation and filtering
    for i in range(5):
        t = i * 0.05

        # Simulate head movement
        H_current = np.eye(4)
        H_current[:3, 3] = [0.01 * i, 0, 1.5]

        # Simulate hand
        P_hand_rel = np.eye(4)
        P_hand_rel[:3, 3] = [0.3, 0, -0.4]

        # Get filtered pose
        pos, rot = manager.get_filtered_hand_pose(H_current, P_hand_rel, timestamp=t)
        vel = manager.get_hand_velocity()

        assert pos.shape == (3,), "Position should be 3D"
        assert rot.shape == (3, 3), "Rotation should be 3x3"
        assert vel.shape == (3,), "Velocity should be 3D"

        print(f"  Frame {i}: pos={pos[:2]}, vel_magnitude={np.linalg.norm(vel):.4f}")

    print("  ✓ ReferenceFrameManager test passed")
    return True


def test_input_aggregator():
    """Test InputAggregator."""
    print("\n" + "="*60)
    print("TEST 2: InputAggregator (Dummy Mode)")
    print("="*60)

    aggregator = DummyInputAggregator()

    # Test clutch
    aggregator.set_clutch(True)
    state = aggregator.get_state()
    assert state.clutch_pressed, "Clutch should be pressed"

    # Test trigger
    aggregator.set_trigger(0.7)
    state = aggregator.get_state()
    assert abs(state.trigger_val - 0.7) < 0.1, "Trigger value should be ~0.7"

    # Test scaling factors
    scaling = aggregator.get_scaling_factors()
    assert 'x' in scaling and 'y' in scaling and 'z' in scaling

    print(f"  Clutch: {state.clutch_pressed}")
    print(f"  Trigger: {state.trigger_val:.2f}")
    print(f"  Scaling: {scaling}")
    print("  ✓ InputAggregator test passed")
    return True


def test_safety_monitor():
    """Test SafetyMonitor."""
    print("\n" + "="*60)
    print("TEST 3: SafetyMonitor")
    print("="*60)

    config = SafetyConfig(
        table_height_z=0.15,
        z_safety_margin=0.03,
        workspace_radius_xy=0.6,
    )

    monitor = SafetyMonitor(config=config)

    # Test 1: Vision latency violation
    old_timestamp = time.time() - 0.5
    healthy = monitor.check_system_health(vision_timestamp=old_timestamp)
    assert not healthy, "Should detect vision latency violation"
    print(f"  Vision latency check: {monitor.get_violation_reason()}")

    # Test 2: IK failure violation
    ik_history = [False] * 6
    healthy = monitor.check_system_health(ik_success_history=ik_history)
    assert not healthy, "Should detect IK failure violation"
    print(f"  IK failure check: {monitor.get_violation_reason()}")

    # Test 3: Workspace clamping
    test_positions = [
        np.array([0.3, 0.0, 0.5]),   # Safe
        np.array([0.8, 0.0, 0.5]),   # Too far XY
        np.array([0.3, 0.0, 0.05]),  # Below table
    ]

    for pos in test_positions:
        clamped, _ = monitor.clamp_to_workspace(pos)
        is_safe = monitor.is_position_safe(pos)
        print(f"  Position {pos} -> Safe: {is_safe}, Clamped: {clamped}")

    print("  ✓ SafetyMonitor test passed")
    return True


def test_motion_planner():
    """Test MotionPlanner."""
    print("\n" + "="*60)
    print("TEST 4: MotionPlanner")
    print("="*60)

    planner = MotionPlanner()

    print(f"  Joint names: {planner.joint_names}")
    print(f"  Max IK iterations: {planner.max_ik_iterations}")

    # Note: IK testing requires URDF
    if planner.ik_solver is None:
        print("  (Skipping IK test - no URDF provided)")
    else:
        print("  IK solver initialized")

    print("  ✓ MotionPlanner test passed")
    return True


def test_data_logger():
    """Test DataLogger."""
    print("\n" + "="*60)
    print("TEST 5: DataLogger")
    print("="*60)

    logger = DataLogger(output_dir="./test_demonstrations")

    # Start recording
    logger.start_recording(task_name="test_task", robot_ip="192.168.1.10")
    assert logger.is_recording(), "Should be recording"

    # Log frames
    for i in range(10):
        t = time.time()

        # Simulate image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        logger.buffer_image(image, timestamp=t)

        # Simulate robot state
        robot_state = {
            'joints': np.random.randn(7),
            'velocities': np.random.randn(7),
            'ee_pose': np.random.randn(7),
            'timestamp': t
        }

        action_delta = np.random.randn(6)
        logger.log_frame(robot_state, action_delta, timestamp=t)

        time.sleep(0.05)

    # Stop and verify
    filepath = logger.stop_recording()
    assert filepath is not None, "Should return filepath"

    stats = logger.get_stats()
    print(f"  Frames logged: {stats['total_frames']}")
    print(f"  Dropped frames: {stats['dropped_frames']}")
    print(f"  File saved: {filepath}")

    # Cleanup
    import os
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"  (Cleaned up test file)")

    print("  ✓ DataLogger test passed")
    return True


def test_robot_interface():
    """Test RobotInterface."""
    print("\n" + "="*60)
    print("TEST 6: RobotInterface (Simulation Mode)")
    print("="*60)

    interface = KinovaRobotInterface(simulation_mode=True)

    assert interface.simulation_mode, "Should be in simulation mode"

    # Test getting state
    state = interface.get_state()
    assert 'joints' in state
    assert 'velocities' in state
    assert 'ee_pose' in state

    print(f"  Robot name: {interface.robot_name}")
    print(f"  Joint names: {interface.joint_names}")
    print(f"  Current state: joints={state['joints'][:3]}...")

    # Test hold position
    success = interface.hold_position()
    assert success, "Hold position should succeed in sim mode"

    # Test gripper
    success = interface.send_gripper(0.5)
    assert success, "Gripper command should succeed in sim mode"

    print("  ✓ RobotInterface test passed")
    return True


def test_full_integration():
    """Test full integration of all modules."""
    print("\n" + "="*60)
    print("TEST 7: Full Integration")
    print("="*60)

    # Initialize all modules
    reference_mgr = ReferenceFrameManager()
    input_agg = DummyInputAggregator()
    safety_monitor = SafetyMonitor(config=SafetyConfig(table_height_z=0.15))
    motion_planner = MotionPlanner()
    robot_interface = KinovaRobotInterface(simulation_mode=True)

    # Calibrate world frame
    H_init = np.eye(4)
    H_init[:3, 3] = [0, 0, 1.5]
    reference_mgr.calibrate_world_frame(H_init)

    # Simulate control loop
    print("\n  Running 5 control iterations...")

    for i in range(5):
        # Simulate Vision Pro data
        H_current = np.eye(4)
        H_current[:3, 3] = [0, 0, 1.5]

        P_hand_rel = np.eye(4)
        P_hand_rel[:3, 3] = [0.3, 0.1 * i, -0.4]

        # Transform and filter
        hand_pos, hand_rot = reference_mgr.get_filtered_hand_pose(H_current, P_hand_rel)
        hand_vel = reference_mgr.get_hand_velocity()

        # Get gamepad input
        input_agg.set_clutch(True)
        input_agg.set_trigger(0.5)
        gamepad_state = input_agg.get_state()

        # Safety check
        vision_timestamp = reference_mgr.get_last_timestamp()
        healthy = safety_monitor.check_system_health(
            vision_timestamp=vision_timestamp,
            ik_success_history=[True] * 3
        )

        # Get robot state
        robot_state = robot_interface.get_state()

        # Apply safety clamping
        safe_pos, _ = safety_monitor.clamp_to_workspace(hand_pos)

        # Hold position (IK would go here if URDF available)
        robot_interface.hold_position()

        print(f"  Iteration {i}: healthy={healthy}, hand_pos={hand_pos[:2]}, gripper={gamepad_state.trigger_val}")

        time.sleep(0.05)

    print("\n  ✓ Full integration test passed")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "INTEGRATION TEST SUITE" + " "*21 + "║")
    print("╚" + "="*58 + "╝")

    tests = [
        ("ReferenceFrameManager", test_reference_frame_manager),
        ("InputAggregator", test_input_aggregator),
        ("SafetyMonitor", test_safety_monitor),
        ("MotionPlanner", test_motion_planner),
        ("DataLogger", test_data_logger),
        ("RobotInterface", test_robot_interface),
        ("Full Integration", test_full_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n  ✗ {name} test FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

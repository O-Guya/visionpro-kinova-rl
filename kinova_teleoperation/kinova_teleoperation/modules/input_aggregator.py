"""Input Aggregator for gamepad control with state machine logic.

Handles Xbox/PlayStation gamepad input with edge detection and scaling modes.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional
import threading


try:
    from inputs import get_gamepad, UnpluggedError
except ImportError:
    print("Warning: 'inputs' library not installed. Run: pip install inputs")
    get_gamepad = None
    UnpluggedError = Exception


@dataclass
class GamepadState:
    """Current gamepad state."""
    clutch_pressed: bool = False
    clutch_just_pressed: bool = False
    trigger_val: float = 0.0
    button_y_just_pressed: bool = False
    button_b_pressed: bool = False  # Emergency stop
    connected: bool = False


class InputAggregator:
    """Unified gamepad input handling with state machine logic.

    Gamepad Mapping (Xbox Controller):
        - Button A: Clutch (hold to engage robot following)
        - Right Trigger: Gripper control (0.0=open, 1.0=closed)
        - Button Y: Toggle scaling mode
        - Button B: Emergency stop (optional)

    Features:
        - Edge detection for clutch and mode toggle
        - 10% deadband on trigger
        - Dual scaling modes (fast/precision)
    """

    # Xbox controller event codes
    BTN_SOUTH = 'BTN_SOUTH'  # A button
    BTN_NORTH = 'BTN_NORTH'  # Y button
    BTN_EAST = 'BTN_EAST'    # B button
    ABS_RZ = 'ABS_RZ'        # Right trigger

    def __init__(self, deadband: float = 0.1, fast_mode: bool = True):
        """Initialize the input aggregator.

        Args:
            deadband: Trigger deadband threshold (0.0 to 1.0)
            fast_mode: If True, start in fast mode; else precision mode
        """
        self.deadband = deadband

        # Current state
        self.state = GamepadState()

        # Previous state for edge detection
        self._prev_clutch = False
        self._prev_button_y = False

        # Scaling modes
        self._fast_mode = fast_mode
        self.scaling_fast = {'x': 1.5, 'y': 1.5, 'z': 1.0}
        self.scaling_precision = {'x': 0.5, 'y': 0.5, 'z': 0.5}

        # Gamepad input thread
        self._running = False
        self._gamepad_thread: Optional[threading.Thread] = None

        # Raw input values
        self._raw_clutch = False
        self._raw_trigger = 0.0
        self._raw_button_y = False
        self._raw_button_b = False

        # Thread lock
        self._lock = threading.Lock()

        # Start monitoring gamepad
        self.start()

    def start(self) -> None:
        """Start gamepad monitoring thread."""
        if get_gamepad is None:
            print("[InputAggregator] Warning: gamepad library not available. Running in dummy mode.")
            self.state.connected = False
            return

        self._running = True
        self._gamepad_thread = threading.Thread(target=self._gamepad_loop, daemon=True)
        self._gamepad_thread.start()
        print("[InputAggregator] Gamepad monitoring started.")

    def stop(self) -> None:
        """Stop gamepad monitoring thread."""
        self._running = False
        if self._gamepad_thread is not None:
            self._gamepad_thread.join(timeout=1.0)
        print("[InputAggregator] Gamepad monitoring stopped.")

    def _gamepad_loop(self) -> None:
        """Background thread to read gamepad events."""
        retry_interval = 2.0
        last_retry = 0.0

        while self._running:
            try:
                events = get_gamepad()
                with self._lock:
                    self.state.connected = True

                for event in events:
                    with self._lock:
                        self._process_event(event)

            except (UnpluggedError, OSError) as e:
                # Gamepad disconnected
                current_time = time.time()
                if current_time - last_retry > retry_interval:
                    print(f"[InputAggregator] Gamepad disconnected: {e}. Retrying...")
                    last_retry = current_time

                with self._lock:
                    self.state.connected = False
                    self._reset_inputs()

                time.sleep(0.5)

            except Exception as e:
                print(f"[InputAggregator] Unexpected error: {e}")
                time.sleep(0.1)

    def _process_event(self, event) -> None:
        """Process individual gamepad event (called within lock)."""
        event_code = event.code
        event_state = event.state

        # Button A (Clutch)
        if event_code == self.BTN_SOUTH:
            self._raw_clutch = bool(event_state)

        # Button Y (Toggle scaling)
        elif event_code == self.BTN_NORTH:
            self._raw_button_y = bool(event_state)

        # Button B (Emergency)
        elif event_code == self.BTN_EAST:
            self._raw_button_b = bool(event_state)

        # Right Trigger (Gripper)
        elif event_code == self.ABS_RZ:
            # Normalize trigger: 0-255 -> 0.0-1.0
            normalized = event_state / 255.0
            self._raw_trigger = normalized

    def _reset_inputs(self) -> None:
        """Reset all raw inputs (called within lock)."""
        self._raw_clutch = False
        self._raw_trigger = 0.0
        self._raw_button_y = False
        self._raw_button_b = False

    def get_state(self) -> GamepadState:
        """Get current gamepad state with edge detection.

        Returns:
            GamepadState with current values and edge flags
        """
        with self._lock:
            # Copy raw values
            current_clutch = self._raw_clutch
            current_trigger = self._raw_trigger
            current_button_y = self._raw_button_y
            current_button_b = self._raw_button_b

        # Edge detection
        clutch_just_pressed = current_clutch and not self._prev_clutch
        button_y_just_pressed = current_button_y and not self._prev_button_y

        # Toggle scaling mode on button Y press
        if button_y_just_pressed:
            self._fast_mode = not self._fast_mode
            mode_name = "Fast (1.5x)" if self._fast_mode else "Precision (0.5x)"
            print(f"[InputAggregator] Switched to {mode_name} mode")

        # Process trigger with deadband
        processed_trigger = 0.0
        if abs(current_trigger) > self.deadband:
            # Scale to full range after deadband
            processed_trigger = (current_trigger - self.deadband) / (1.0 - self.deadband)
            processed_trigger = max(0.0, min(1.0, processed_trigger))

        # Update state
        self.state.clutch_pressed = current_clutch
        self.state.clutch_just_pressed = clutch_just_pressed
        self.state.trigger_val = processed_trigger
        self.state.button_y_just_pressed = button_y_just_pressed
        self.state.button_b_pressed = current_button_b

        # Update previous state
        self._prev_clutch = current_clutch
        self._prev_button_y = current_button_y

        return self.state

    def get_scaling_factors(self) -> Dict[str, float]:
        """Get current scaling factors based on mode.

        Returns:
            Dictionary with 'x', 'y', 'z' scaling factors
        """
        return self.scaling_fast if self._fast_mode else self.scaling_precision

    def is_connected(self) -> bool:
        """Check if gamepad is connected."""
        return self.state.connected

    def set_scaling_modes(
        self,
        fast: Dict[str, float],
        precision: Dict[str, float]
    ) -> None:
        """Configure custom scaling modes.

        Args:
            fast: Fast mode scaling {'x': 1.5, 'y': 1.5, 'z': 1.0}
            precision: Precision mode scaling {'x': 0.5, 'y': 0.5, 'z': 0.5}
        """
        self.scaling_fast = fast
        self.scaling_precision = precision
        print(f"[InputAggregator] Scaling modes updated: fast={fast}, precision={precision}")


if __name__ == "__main__":
    # Test the input aggregator
    print("Testing InputAggregator...")
    print("Controls:")
    print("  A button: Clutch")
    print("  Y button: Toggle scaling mode")
    print("  Right Trigger: Gripper control")
    print("  B button: Emergency stop")
    print("\nPress Ctrl+C to exit\n")

    aggregator = InputAggregator()

    try:
        while True:
            state = aggregator.get_state()
            scaling = aggregator.get_scaling_factors()

            if state.clutch_just_pressed:
                print(">>> CLUTCH ENGAGED <<<")

            if state.clutch_pressed:
                print(f"Clutch: ON | Trigger: {state.trigger_val:.2f} | Scaling: {scaling}")

            if state.button_b_pressed:
                print("!!! EMERGENCY STOP !!!")

            time.sleep(0.05)  # 20Hz

    except KeyboardInterrupt:
        print("\nStopping...")
        aggregator.stop()
        print("Test complete!")

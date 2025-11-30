"""Reference Frame Manager for Vision Pro hand tracking.

Transforms hand poses from head-relative to world-fixed coordinates and applies filtering.
"""

import numpy as np
import time
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation

from ..utils.one_euro_filter import OneEuroFilter3D


class ReferenceFrameManager:
    """Manages coordinate frame transformations and filtering for Vision Pro tracking.

    Transforms hand poses from head-relative coordinates to a world-fixed reference frame
    established at initialization. Applies One Euro Filter for smooth motion.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        """Initialize the reference frame manager.

        Args:
            min_cutoff: Minimum cutoff frequency for OneEuroFilter (Hz)
            beta: Cutoff slope for adaptive filtering
            d_cutoff: Cutoff frequency for derivative estimation (Hz)
        """
        self.H_init: Optional[np.ndarray] = None  # Initial head pose (4x4)
        self.H_init_inv: Optional[np.ndarray] = None  # Inverse for efficient computation

        # Filters for position (x, y, z)
        self.position_filter = OneEuroFilter3D(min_cutoff, beta, d_cutoff)

        # Cached values
        self._filtered_position: Optional[np.ndarray] = None
        self._filtered_rotation: Optional[np.ndarray] = None
        self._hand_velocity: Optional[np.ndarray] = None
        self._last_timestamp: Optional[float] = None

        self._calibrated = False

    def calibrate_world_frame(self, head_pose: np.ndarray) -> None:
        """Calibrate the world reference frame using the current head pose.

        Args:
            head_pose: 4x4 transformation matrix representing initial head pose
        """
        if head_pose.shape != (4, 4):
            # Handle (1, 4, 4) format from VisionProStreamer
            if head_pose.shape == (1, 4, 4):
                head_pose = head_pose[0]
            else:
                raise ValueError(f"Invalid head_pose shape: {head_pose.shape}. Expected (4, 4) or (1, 4, 4)")

        self.H_init = head_pose.copy()
        self.H_init_inv = np.linalg.inv(head_pose)
        self._calibrated = True

        # Reset filters on calibration
        self.position_filter.reset()

        print(f"[ReferenceFrameManager] World frame calibrated.")
        print(f"  Initial head position: {head_pose[:3, 3]}")

    def is_calibrated(self) -> bool:
        """Check if world frame has been calibrated."""
        return self._calibrated

    def transform_to_world_frame(
        self,
        head_pose: np.ndarray,
        hand_pose_relative: np.ndarray
    ) -> np.ndarray:
        """Transform hand pose from head-relative to world-fixed coordinates.

        Args:
            head_pose: Current head pose (4x4 or 1x4x4)
            hand_pose_relative: Hand pose relative to current head (4x4 or 1x4x4)

        Returns:
            Hand pose in world frame (4x4)
        """
        if not self._calibrated:
            raise RuntimeError("World frame not calibrated. Call calibrate_world_frame() first.")

        # Handle (1, 4, 4) format from VisionProStreamer
        if head_pose.shape == (1, 4, 4):
            head_pose = head_pose[0]
        if hand_pose_relative.shape == (1, 4, 4):
            hand_pose_relative = hand_pose_relative[0]

        # P_world = H_init^-1 × H_current × P_raw
        P_world = self.H_init_inv @ head_pose @ hand_pose_relative

        return P_world

    def get_filtered_hand_pose(
        self,
        head_pose: np.ndarray,
        hand_pose_relative: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get filtered hand pose in world coordinates.

        Args:
            head_pose: Current head pose (4x4 or 1x4x4)
            hand_pose_relative: Hand pose relative to head (4x4 or 1x4x4)
            timestamp: Optional timestamp (seconds). If None, uses current time.

        Returns:
            (position, rotation) tuple:
                - position: 3D position [x, y, z] in meters
                - rotation: 3x3 rotation matrix
        """
        # Transform to world frame
        P_world = self.transform_to_world_frame(head_pose, hand_pose_relative)

        # Extract position and rotation
        position = P_world[:3, 3]
        rotation = P_world[:3, :3]

        # Apply filtering to position
        if timestamp is None:
            timestamp = time.time()

        filtered_position, velocity = self.position_filter(position, timestamp)

        # Cache results
        self._filtered_position = filtered_position
        self._filtered_rotation = rotation  # Not filtered currently
        self._hand_velocity = velocity
        self._last_timestamp = timestamp

        return filtered_position, rotation

    def get_hand_velocity(self) -> np.ndarray:
        """Get current hand velocity estimate from filter.

        Returns:
            3D velocity vector [vx, vy, vz] in m/s
        """
        if self._hand_velocity is None:
            return np.zeros(3)
        return self._hand_velocity

    def get_last_timestamp(self) -> Optional[float]:
        """Get timestamp of last processed data."""
        return self._last_timestamp

    def reset(self) -> None:
        """Reset filters but preserve calibration."""
        self.position_filter.reset()
        self._filtered_position = None
        self._filtered_rotation = None
        self._hand_velocity = None
        self._last_timestamp = None

    def recalibrate(self, head_pose: np.ndarray) -> None:
        """Recalibrate world frame and reset filters.

        Args:
            head_pose: New initial head pose (4x4)
        """
        self.calibrate_world_frame(head_pose)
        self.reset()


if __name__ == "__main__":
    # Test the reference frame manager
    print("Testing ReferenceFrameManager...")

    manager = ReferenceFrameManager(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)

    # Simulate initial head pose
    H_init = np.eye(4)
    H_init[:3, 3] = [0, 0, 1.5]  # Head at 1.5m height

    manager.calibrate_world_frame(H_init)

    # Simulate hand tracking data
    for i in range(10):
        t = i * 0.05  # 20Hz

        # Simulate head moving
        H_current = np.eye(4)
        H_current[:3, 3] = [0.01 * i, 0, 1.5]

        # Simulate hand relative to head
        P_hand_rel = np.eye(4)
        P_hand_rel[:3, 3] = [0.3, 0, -0.4]  # Hand 30cm forward, 40cm down from head

        # Get filtered pose
        pos, rot = manager.get_filtered_hand_pose(H_current, P_hand_rel, timestamp=t)
        vel = manager.get_hand_velocity()

        print(f"Frame {i}: pos={pos}, vel={vel}")

    print("\nTest complete!")

"""Data Logger for multi-modal data recording with timestamp alignment.

Records demonstration data in HDF5 format compatible with HIL-SERL.
"""

import h5py
import numpy as np
from typing import Dict, Optional, Any
from collections import deque
from dataclasses import dataclass
import time
from datetime import datetime
from pathlib import Path


@dataclass
class TimestampedData:
    """Data with timestamp."""
    timestamp: float
    data: Any


class DataLogger:
    """Multi-modal data logger with strict timestamp alignment.

    Records:
        - Joint positions/velocities
        - End-effector poses
        - Camera images
        - Commanded actions
        - Metadata

    HDF5 Structure:
        observations/images: (N, H, W, 3) uint8
        observations/qpos: (N, 7) float32
        observations/qvel: (N, 7) float32
        observations/ee_pose: (N, 7) float32 [x,y,z,qx,qy,qz,qw]
        actions/cartesian_delta: (N, 6) float32 [dx,dy,dz,drx,dry,drz]
        actions/joint_positions: (N, 7) float32
        metadata: dict
    """

    def __init__(
        self,
        output_dir: str = "./demonstrations",
        buffer_size: int = 50,
        alignment_threshold_ms: float = 50.0,
        interpolation_threshold_ms: float = 100.0,
    ):
        """Initialize the data logger.

        Args:
            output_dir: Directory to save HDF5 files
            buffer_size: Size of circular buffers for alignment
            alignment_threshold_ms: Time tolerance for direct save (ms)
            interpolation_threshold_ms: Time threshold for interpolation vs drop (ms)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.alignment_threshold = alignment_threshold_ms / 1000.0  # Convert to seconds
        self.interpolation_threshold = interpolation_threshold_ms / 1000.0

        # Circular buffers
        self.image_buffer = deque(maxlen=buffer_size)
        self.robot_state_buffer = deque(maxlen=buffer_size)
        self.action_buffer = deque(maxlen=buffer_size)

        # Recording state
        self._recording = False
        self._current_file: Optional[h5py.File] = None
        self._current_filepath: Optional[Path] = None

        # Data lists (accumulated during recording)
        self._images = []
        self._qpos = []
        self._qvel = []
        self._ee_poses = []
        self._cartesian_deltas = []
        self._joint_actions = []
        self._timestamps = []

        # Metadata
        self._metadata = {}

        # Statistics
        self.stats = {
            'total_frames': 0,
            'dropped_frames': 0,
            'interpolated_frames': 0,
            'direct_frames': 0,
        }

        print(f"[DataLogger] Initialized. Output directory: {self.output_dir}")

    def start_recording(
        self,
        task_name: str,
        robot_ip: Optional[str] = None,
        **extra_metadata
    ) -> None:
        """Start a new recording session.

        Args:
            task_name: Name of the task being demonstrated
            robot_ip: Robot IP address (optional)
            **extra_metadata: Additional metadata to save
        """
        if self._recording:
            print("[DataLogger] Warning: Already recording. Stopping previous session.")
            self.stop_recording()

        # Create filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_name}_{timestamp_str}.hdf5"
        self._current_filepath = self.output_dir / filename

        # Initialize metadata
        self._metadata = {
            'task_name': task_name,
            'start_time': datetime.now().isoformat(),
            'robot_ip': robot_ip if robot_ip else "unknown",
            **extra_metadata
        }

        # Clear data lists
        self._images.clear()
        self._qpos.clear()
        self._qvel.clear()
        self._ee_poses.clear()
        self._cartesian_deltas.clear()
        self._joint_actions.clear()
        self._timestamps.clear()

        # Clear buffers
        self.image_buffer.clear()
        self.robot_state_buffer.clear()
        self.action_buffer.clear()

        # Reset stats
        for key in self.stats:
            self.stats[key] = 0

        self._recording = True

        print(f"[DataLogger] Recording started: {filename}")
        print(f"  Task: {task_name}")

    def log_frame(
        self,
        robot_state: Dict[str, Any],
        action_delta: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """Log a single frame of data with timestamp alignment.

        Args:
            robot_state: Dict with keys: 'joints' (7D), 'velocities' (7D),
                        'ee_pose' (7D [x,y,z,qx,qy,qz,qw]), 'timestamp'
            action_delta: Commanded action [dx,dy,dz,drx,dry,drz] (6D)
            image: Camera image (H, W, 3)
            timestamp: Optional override timestamp
        """
        if not self._recording:
            return

        if timestamp is None:
            timestamp = time.time()

        # Extract robot state
        robot_timestamp = robot_state.get('timestamp', timestamp)
        qpos = robot_state.get('joints', np.zeros(7))
        qvel = robot_state.get('velocities', np.zeros(7))
        ee_pose = robot_state.get('ee_pose', np.zeros(7))

        # Find matching image (aligned by timestamp)
        aligned_image = self._find_aligned_image(robot_timestamp)

        if aligned_image is None:
            # No matching image - drop this frame
            self.stats['dropped_frames'] += 1
            return

        # Append data
        self._images.append(aligned_image)
        self._qpos.append(qpos)
        self._qvel.append(qvel)
        self._ee_poses.append(ee_pose)

        # Action (use zeros if not provided)
        if action_delta is not None:
            self._cartesian_deltas.append(action_delta)
        else:
            self._cartesian_deltas.append(np.zeros(6))

        # Joint action (use current qpos if not provided)
        self._joint_actions.append(qpos)

        self._timestamps.append(robot_timestamp)

        self.stats['total_frames'] += 1
        self.stats['direct_frames'] += 1

    def _find_aligned_image(self, target_timestamp: float) -> Optional[np.ndarray]:
        """Find image aligned with target timestamp.

        Alignment strategy:
            - Δt < 50ms: Use directly
            - 50ms ≤ Δt < 100ms: Interpolate (not implemented yet, uses nearest)
            - Δt ≥ 100ms: Drop frame

        Args:
            target_timestamp: Target timestamp to align to

        Returns:
            Aligned image or None if dropped
        """
        if len(self.image_buffer) == 0:
            return None

        # Find nearest image
        min_delta = float('inf')
        nearest_image = None

        for img_data in self.image_buffer:
            delta = abs(img_data.timestamp - target_timestamp)
            if delta < min_delta:
                min_delta = delta
                nearest_image = img_data.data

        # Check alignment threshold
        if min_delta < self.alignment_threshold:
            # Direct use
            return nearest_image

        elif min_delta < self.interpolation_threshold:
            # Interpolation (for now, just use nearest)
            # TODO: Implement proper interpolation for image data
            self.stats['interpolated_frames'] += 1
            return nearest_image

        else:
            # Drop frame
            return None

    def buffer_image(self, image: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Add image to circular buffer.

        Args:
            image: Image array (H, W, 3)
            timestamp: Image timestamp (if None, uses current time)
        """
        if timestamp is None:
            timestamp = time.time()

        self.image_buffer.append(TimestampedData(timestamp, image))

    def stop_recording(self) -> Optional[str]:
        """Stop recording and save HDF5 file.

        Returns:
            Path to saved file or None if error
        """
        if not self._recording:
            print("[DataLogger] Not currently recording.")
            return None

        self._recording = False

        if self.stats['total_frames'] == 0:
            print("[DataLogger] No frames recorded. Not saving file.")
            return None

        # Convert lists to numpy arrays
        images = np.array(self._images, dtype=np.uint8)
        qpos = np.array(self._qpos, dtype=np.float32)
        qvel = np.array(self._qvel, dtype=np.float32)
        ee_poses = np.array(self._ee_poses, dtype=np.float32)
        cartesian_deltas = np.array(self._cartesian_deltas, dtype=np.float32)
        joint_actions = np.array(self._joint_actions, dtype=np.float32)

        # Add end time to metadata
        self._metadata['end_time'] = datetime.now().isoformat()
        self._metadata['num_frames'] = self.stats['total_frames']
        self._metadata['dropped_frames'] = self.stats['dropped_frames']

        # Save to HDF5
        try:
            with h5py.File(self._current_filepath, 'w') as f:
                # Observations group
                obs_group = f.create_group('observations')
                obs_group.create_dataset('images', data=images, compression='gzip')
                obs_group.create_dataset('qpos', data=qpos)
                obs_group.create_dataset('qvel', data=qvel)
                obs_group.create_dataset('ee_pose', data=ee_poses)

                # Actions group
                action_group = f.create_group('actions')
                action_group.create_dataset('cartesian_delta', data=cartesian_deltas)
                action_group.create_dataset('joint_positions', data=joint_actions)

                # Metadata
                meta_group = f.create_group('metadata')
                for key, value in self._metadata.items():
                    meta_group.attrs[key] = value

            print(f"[DataLogger] Recording saved: {self._current_filepath}")
            print(f"  Frames: {self.stats['total_frames']}")
            print(f"  Dropped: {self.stats['dropped_frames']}")
            print(f"  Duration: {images.shape[0] / 20.0:.1f}s (at 20Hz)")

            return str(self._current_filepath)

        except Exception as e:
            print(f"[DataLogger] Error saving file: {e}")
            return None

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def get_stats(self) -> Dict:
        """Get recording statistics.

        Returns:
            Dictionary with frame counts and alignment stats
        """
        stats = self.stats.copy()
        if self.stats['total_frames'] > 0:
            stats['alignment_histogram'] = {
                'direct': self.stats['direct_frames'],
                'interpolated': self.stats['interpolated_frames'],
                'dropped': self.stats['dropped_frames'],
            }
        return stats


if __name__ == "__main__":
    # Test the data logger
    print("Testing DataLogger...")

    logger = DataLogger(output_dir="./test_demonstrations")

    # Start recording
    logger.start_recording(
        task_name="test_task",
        robot_ip="192.168.1.10"
    )

    # Simulate data logging
    for i in range(20):
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

        # Log frame
        logger.log_frame(robot_state, action_delta, timestamp=t)

        time.sleep(0.05)  # 20Hz

    # Stop and save
    filepath = logger.stop_recording()

    print(f"\nSaved to: {filepath}")
    print("\nStatistics:")
    for key, value in logger.get_stats().items():
        print(f"  {key}: {value}")

    print("\nTest complete!")

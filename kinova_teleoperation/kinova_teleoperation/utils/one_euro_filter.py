"""One Euro Filter for low-latency smoothing with velocity estimation.

Reference: Géry Casiez, Nicolas Roussel, and Daniel Vogel. 2012.
1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.
"""

import time
import numpy as np
from typing import Optional


class LowPassFilter:
    """First-order low-pass filter."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.y = None
        self.s = None

    def filter(self, x: float, alpha: Optional[float] = None) -> float:
        """Apply filter to input x."""
        if alpha is None:
            alpha = self.alpha

        if self.y is None:
            self.s = x
        else:
            self.s = alpha * x + (1.0 - alpha) * self.s

        self.y = x
        return self.s

    def filter_with_alpha(self, x: float, alpha: float) -> float:
        """Apply filter with custom alpha."""
        return self.filter(x, alpha)


class OneEuroFilter:
    """One Euro Filter for adaptive smoothing based on signal velocity.

    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz) - controls smoothing at low speeds
        beta: Cutoff slope - controls adaptation to velocity changes
        d_cutoff: Cutoff frequency for derivative (Hz)
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter(self._alpha(min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(d_cutoff))

        self.last_time = None
        self.last_value = None

    def _alpha(self, cutoff: float) -> float:
        """Compute smoothing factor alpha from cutoff frequency."""
        te = 1.0 / cutoff
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp: Optional[float] = None) -> tuple[float, float]:
        """Filter input x and return (filtered_value, velocity).

        Args:
            x: Input value to filter
            timestamp: Optional timestamp (seconds). If None, uses current time.

        Returns:
            (filtered_value, estimated_velocity)
        """
        if timestamp is None:
            timestamp = time.time()

        # Initialize on first call
        if self.last_time is None:
            self.last_time = timestamp
            self.last_value = x
            self.x_filter.filter(x)
            self.dx_filter.filter(0.0)
            return x, 0.0

        # Compute time delta
        dt = timestamp - self.last_time
        if dt <= 0:
            dt = 1e-6  # Avoid division by zero

        # Estimate derivative (velocity)
        dx = (x - self.last_value) / dt

        # Filter derivative
        edx = self.dx_filter.filter_with_alpha(dx, self._alpha(self.d_cutoff))

        # Adaptive cutoff frequency based on derivative magnitude
        cutoff = self.min_cutoff + self.beta * abs(edx)

        # Filter value
        filtered_x = self.x_filter.filter_with_alpha(x, self._alpha(cutoff))

        # Update state
        self.last_time = timestamp
        self.last_value = x

        return filtered_x, edx

    def reset(self):
        """Reset filter state."""
        self.x_filter = LowPassFilter(self._alpha(self.min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(self.d_cutoff))
        self.last_time = None
        self.last_value = None


class OneEuroFilter3D:
    """3D One Euro Filter for position vectors."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.filters = [
            OneEuroFilter(min_cutoff, beta, d_cutoff),
            OneEuroFilter(min_cutoff, beta, d_cutoff),
            OneEuroFilter(min_cutoff, beta, d_cutoff)
        ]

    def __call__(self, position: np.ndarray, timestamp: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """Filter 3D position and return (filtered_position, velocity).

        Args:
            position: 3D position array [x, y, z]
            timestamp: Optional timestamp (seconds)

        Returns:
            (filtered_position, velocity) - both as 3D numpy arrays
        """
        if timestamp is None:
            timestamp = time.time()

        filtered_pos = np.zeros(3)
        velocity = np.zeros(3)

        for i in range(3):
            filtered_pos[i], velocity[i] = self.filters[i](position[i], timestamp)

        return filtered_pos, velocity

    def reset(self):
        """Reset all filters."""
        for f in self.filters:
            f.reset()

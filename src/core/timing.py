"""Wall-clock timing utilities for the live update loop.

This module is deliberately framework-free so it can be unit-tested and reused
by the DSP engine in later stages without importing Streamlit.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional


def monotonic() -> float:
    """Return the current time on the monotonic clock, in seconds."""
    return time.monotonic()


class RuntimeMetrics:
    """Measures the health of the live update loop.

    Records every completed update cycle and derives elapsed runtime, the
    total update count, and both instantaneous (rolling) and average update
    rates. Elapsed time uses the monotonic clock so it is immune to system
    clock adjustments.
    """

    def __init__(self, rolling_window: int = 20) -> None:
        self._rolling_window: int = rolling_window
        self._started_at: Optional[float] = None
        self._total_ticks: int = 0
        self._recent_ticks: Deque[float] = deque(maxlen=rolling_window)

    def start(self) -> None:
        """Reset all metrics and pin the start time."""
        self._started_at = monotonic()
        self._total_ticks = 0
        self._recent_ticks.clear()

    def tick(self) -> None:
        """Record the completion of one update cycle."""
        self._total_ticks += 1
        self._recent_ticks.append(monotonic())

    @property
    def started(self) -> bool:
        """True once :meth:`start` has been called at least once."""
        return self._started_at is not None

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since :meth:`start`, or 0 before start."""
        if self._started_at is None:
            return 0.0
        return monotonic() - self._started_at

    @property
    def total_ticks(self) -> int:
        """Total number of recorded update cycles."""
        return self._total_ticks

    @property
    def average_fps(self) -> float:
        """Average update rate over the whole runtime, in updates per second."""
        if not self.started or self.elapsed_seconds <= 0.0:
            return 0.0
        return self._total_ticks / self.elapsed_seconds

    @property
    def rolling_fps(self) -> float:
        """Instantaneous update rate over the rolling window, in updates/s."""
        recent = list(self._recent_ticks)
        if len(recent) < 2:
            return self.average_fps
        span = recent[-1] - recent[0]
        if span <= 0.0:
            return 0.0
        return (len(recent) - 1) / span

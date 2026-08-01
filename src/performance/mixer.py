"""Bounded, overlapping-stroke audio mixer.

A single mono ring buffer holds *future* audio (scheduler writes ahead of the
playhead) and *recent past* audio (the device callback reads the playhead
region). Strokes are mixed at write time, so the device callback is a pure,
lock-guarded copy that never touches the DSP engine.

Memory is bounded: the ring holds ``ring_seconds`` of audio regardless of how
long a performance runs. A lock serialises the two writers (scheduler and
device callback); both critical sections are short (a copy / an O(length) add).
"""

from __future__ import annotations

import threading

import numpy as np

from src.synthesis.tabla import soft_limit


class StrokeMixer:
    """Mono ring buffer with a playhead advancing in device blocks."""

    def __init__(
        self,
        *,
        fs: float = 44100.0,
        block_size: int = 256,
        ring_seconds: float = 2.5,
        output_ceiling: float = 0.95,
    ) -> None:
        self._fs = float(fs)
        self._block = max(1, int(block_size))
        self._ceiling = max(1e-9, float(output_ceiling))
        self._capacity = max(
            self._block * 8, int(round(float(ring_seconds) * self._fs))
        )
        self._ring = np.zeros(self._capacity, dtype=np.float64)
        #: Absolute sample time each ring slot was last written at. Used to tell
        #: "unplayed overlap" (sum) from "already-played previous lap" (replace),
        #: so a long performance never lets stale audio accumulate in the ring.
        self._last_write_abs = np.full(self._capacity, -np.inf)
        self._lock = threading.Lock()
        self._playhead = 0
        self._write_extent = 0
        self._recording = False
        self._recorded_blocks: list[np.ndarray] = []

    # -- accessors -------------------------------------------------------------

    @property
    def playhead(self) -> int:
        """Absolute sample index of the next block the device will read."""
        with self._lock:
            return self._playhead

    @property
    def block_size(self) -> int:
        """Samples per output block."""
        return self._block

    @property
    def fs(self) -> float:
        """Sample rate, in samples per second."""
        return self._fs

    @property
    def write_extent(self) -> int:
        """Furthest absolute sample that has audio written into the ring."""
        with self._lock:
            return self._write_extent

    @property
    def prepared_seconds(self) -> float:
        """Seconds of scheduled audio currently ahead of the playhead."""
        with self._lock:
            return max(0.0, (self._write_extent - self._playhead) / self._fs)

    # -- writing (scheduler thread) -------------------------------------------

    def write_stroke(self, time_samples: int, buffer: np.ndarray) -> int:
        """Mix *buffer* into the ring starting at *time_samples*.

        The write position is clamped to at least one full block ahead of the
        playhead so it can never race the device callback. Returns the furthest
        absolute sample written.

        Overlapping strokes (written before the playhead reaches them) are
        summed; content left from a *previous* ring lap was already played and
        is replaced, never re-summed, so audio cannot pile up over time.
        """
        signal = np.asarray(buffer, dtype=np.float64)
        if signal.size == 0:
            return self._write_extent
        with self._lock:
            start = max(int(time_samples), self._playhead + self._block)
            if start + signal.size > self._write_extent + self._capacity:
                start = max(0, self._write_extent + self._capacity - signal.size)
            indices = np.arange(start, start + signal.size) % self._capacity
            absolute = start + np.arange(signal.size)
            stale = self._last_write_abs[indices] < self._playhead
            region = np.where(stale, signal, self._ring[indices] + signal)
            self._ring[indices] = soft_limit(region, ceiling=self._ceiling)
            self._last_write_abs[indices] = absolute
            self._write_extent = max(self._write_extent, start + signal.size)
            return self._write_extent

    # -- recording ------------------------------------------------------------

    def start_recording(self) -> None:
        """Begin capturing the exact mixed blocks consumed by playback."""
        with self._lock:
            self._recorded_blocks = []
            self._recording = True

    def stop_recording(self) -> np.ndarray:
        """Stop capture and return the recorded mono signal."""
        with self._lock:
            self._recording = False
            if not self._recorded_blocks:
                return np.zeros(0, dtype=np.float64)
            recording = np.concatenate(self._recorded_blocks).astype(
                np.float64, copy=False
            )
            self._recorded_blocks = []
            return recording

    # -- reading (device / simulated callback) --------------------------------

    def consume_block(self) -> np.ndarray:
        """Return the next output block and advance the playhead.

        Returns a float64 mono array of length ``block_size``.
        """
        with self._lock:
            out = self._read_region(self._playhead, self._playhead + self._block)
            if self._recording:
                self._recorded_blocks.append(out.copy())
            self._playhead += self._block
            return out

    def recent_audio(self, count: int) -> np.ndarray:
        """Return the most recent *count* played samples ending at the playhead."""
        with self._lock:
            start = self._playhead - max(0, int(count))
            return self._read_region(start, self._playhead)

    # -- internals -------------------------------------------------------------

    def _read_region(self, start: int, end: int) -> np.ndarray:
        length = max(0, end - start)
        if length <= 0:
            return np.zeros(0, dtype=np.float64)
        indices = np.arange(start, end) % self._capacity
        return self._ring[indices].copy()

"""Bit stream manager: continuous, non-blocking supply of random bits.

Owns a shared :class:`BitBuffer`, a background refill thread and the running
stats. The dashboard thread reads bits through :meth:`take` (never blocking),
while the refill thread tops the reservoir up toward the configured target.
This decouples slow entropy sources from fast consumers.
"""

from __future__ import annotations

import threading
from typing import Optional

from src.core.randomness.base import ProviderStatus, RandomnessProvider
from src.core.randomness.buffer import BitBuffer
from src.core.timing import monotonic


class BufferStats:
    """Lifetime statistics for a :class:`BitStreamManager`."""

    def __init__(self) -> None:
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._generated: int = 0
        self._consumed: int = 0
        self._refills: int = 0
        self._last_take: str = ""

    def start(self) -> None:
        """Begin (or resume) the measurement window."""
        if self._started_at is None:
            self._started_at = monotonic()
        self._stopped_at = None

    def stop(self) -> None:
        """Freeze the measurement window so rates stop decaying."""
        if self._started_at is not None and self._stopped_at is None:
            self._stopped_at = monotonic()

    def record_generated(self, count: int) -> None:
        self._generated += count
        self._refills += 1

    def record_consumed(self, count: int, bits: str) -> None:
        self._consumed += count
        if bits:
            self._last_take = bits

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        end = self._stopped_at if self._stopped_at is not None else monotonic()
        return max(0.0, end - self._started_at)

    @property
    def total_generated(self) -> int:
        return self._generated

    @property
    def total_consumed(self) -> int:
        return self._consumed

    @property
    def refill_count(self) -> int:
        return self._refills

    @property
    def last_take(self) -> str:
        return self._last_take

    @property
    def bits_per_second(self) -> float:
        """Average consumption rate in bits per second over the window."""
        elapsed = self.elapsed_seconds
        if elapsed <= 0.0:
            return 0.0
        return self._consumed / elapsed


class BitStreamManager:
    """Thread-safe reservoir that stays refilled from a randomness provider."""

    def __init__(
        self,
        provider: RandomnessProvider,
        *,
        target_bits: int,
        refill_batch_bits: int,
        refill_interval_s: float,
    ) -> None:
        self._provider: RandomnessProvider = provider
        self._buffer: BitBuffer = BitBuffer()
        self._target_bits: int = max(1, int(target_bits))
        self._refill_batch_bits: int = max(1, int(refill_batch_bits))
        self._interval_s: float = max(0.001, refill_interval_s)
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stats = BufferStats()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Begin refilling the reservoir in the background."""
        with self._lock:
            self._stop_event.clear()
            self._stats.start()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._refill_loop,
                    name="qtabla-refill",
                    daemon=True,
                )
                self._thread.start()

    def stop(self) -> None:
        """Signal the refill thread to stop and wait for it to finish."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._stats.stop()

    # -- consumption ----------------------------------------------------------

    def take(self, count: int) -> str:
        """Remove and return up to *count* bits (never blocks).

        Returns fewer bits than requested when the reservoir is temporarily
        drained; the refill thread tops it up on the next cycle.
        """
        with self._lock:
            bits = self._buffer.take(count)
        self._stats.record_consumed(len(bits), bits)
        return bits

    def available(self) -> int:
        """Number of bits immediately available in the reservoir."""
        with self._lock:
            return self._buffer.available

    def preview(self, count: int) -> str:
        """Peek at the first *count* bits without consuming them."""
        with self._lock:
            return self._buffer.preview(count)

    def clear(self) -> None:
        """Discard all buffered bits (the refill thread tops up afterwards)."""
        with self._lock:
            self._buffer.clear()

    # -- reporting ------------------------------------------------------------

    def status(self) -> ProviderStatus:
        """Status of the underlying randomness provider."""
        return self._provider.status()

    @property
    def stats(self) -> BufferStats:
        """Lifetime statistics for this manager."""
        return self._stats

    @property
    def running(self) -> bool:
        """True while the background refill thread is active."""
        return self._thread is not None and self._thread.is_alive()

    # -- internals ------------------------------------------------------------

    def _refill_loop(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self._refill()

    def _refill(self) -> None:
        with self._lock:
            deficit = self._target_bits - self._buffer.available
            if deficit <= 0:
                return
            batch = min(deficit, self._refill_batch_bits)
            bits = self._provider.request_bits(batch)
            if not bits:
                return
            self._buffer.append(bits)
            self._stats.record_generated(len(bits))

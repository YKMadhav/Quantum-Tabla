"""Thread-safe, continuously growing bit reservoir.

Stores bits as chunks of ``'0'``/``'1'`` strings in a deque and consumes from
the front, so both append and take are O(1) per chunk and allocations stay
small. All public methods are guarded by a lock, making the buffer safe to
share between a background refill thread and the dashboard thread.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque

from src.utils.bits import validate_bitstring


class BitBuffer:
    """A lock-protected FIFO reservoir of random bits."""

    def __init__(self) -> None:
        self._chunks: Deque[str] = deque()
        self._head: int = 0
        self._count: int = 0
        self._lock: threading.RLock = threading.RLock()

    @property
    def available(self) -> int:
        """Number of bits currently in the buffer."""
        with self._lock:
            return self._count

    def append(self, bits: str) -> None:
        """Append *bits* to the tail of the reservoir.

        Raises:
            ValueError: if *bits* is not a valid binary string.
        """
        if not validate_bitstring(bits):
            raise ValueError(f"Invalid binary string: {bits!r}")
        if not bits:
            return
        with self._lock:
            self._chunks.append(bits)
            self._count += len(bits)

    def take(self, count: int) -> str:
        """Remove and return up to *count* bits from the front.

        Returns fewer bits (possibly ``""``) when the buffer is drained —
        it never blocks.
        """
        if count <= 0:
            return ""
        with self._lock:
            count = min(count, self._count)
            parts: list[str] = []
            remaining = count
            while remaining > 0 and self._chunks:
                chunk = self._chunks[0]
                start = self._head
                take_len = min(remaining, len(chunk) - start)
                parts.append(chunk[start:start + take_len])
                remaining -= take_len
                self._count -= take_len
                self._head += take_len
                if self._head >= len(chunk):
                    self._chunks.popleft()
                    self._head = 0
            return "".join(parts)

    def preview(self, count: int) -> str:
        """Return the first *count* bits without consuming them."""
        if count <= 0:
            return ""
        with self._lock:
            count = min(count, self._count)
            parts: list[str] = []
            remaining = count
            offset = self._head
            for chunk in self._chunks:
                if remaining <= 0:
                    break
                available = len(chunk) - offset
                take_len = min(remaining, available)
                parts.append(chunk[offset:offset + take_len])
                remaining -= take_len
                offset = 0
            return "".join(parts)

    def clear(self) -> None:
        """Discard every buffered bit."""
        with self._lock:
            self._chunks.clear()
            self._head = 0
            self._count = 0

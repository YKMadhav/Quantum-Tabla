"""Classical randomness provider (deterministic PRNG).

Sources unbiased bits from NumPy's PCG64 PRNG. It conforms to the same
``RandomnessProvider`` contract as the quantum provider, so no other module
knows or cares where the bits come from. As a PRNG it is deterministic when
seeded and reproducible on any machine; it is *not* an entropy source.
"""

from __future__ import annotations

import threading

import numpy as np

from src.core.randomness.base import (
    ProviderStats,
    ProviderStatus,
    RandomnessProvider,
)
from src.core.randomness.buffer import BitBuffer


class ClassicalRandomProvider(RandomnessProvider):
    """A NumPy-backed provider with an internal bit reservoir.

    Bits are generated as independent, uniform ``0``/``1`` integers and
    buffered internally so :meth:`request_bits` can be served instantly up to
    the configured buffer size. The buffer refills lazily when depleted.
    """

    def __init__(self, seed: int | None = None, buffer_size: int = 4096) -> None:
        self._rng = np.random.default_rng(seed)
        self._buffer = BitBuffer()
        self._buffer_size = max(1, int(buffer_size))
        self._lock = threading.RLock()
        self._generated = 0
        self._zeros = 0
        self._ones = 0

    @property
    def name(self) -> str:
        return "numpy-classical"

    def request_bits(self, count: int) -> str:
        """Return exactly *count* bits, generating more as needed."""
        if count <= 0:
            return ""
        with self._lock:
            if self._buffer.available < count:
                needed = max(count, self._buffer_size)
                self._buffer.append(self._generate(needed))
            return self._buffer.take(count)

    def available_bits(self) -> int:
        return self._buffer.available

    def clear_buffer(self) -> None:
        with self._lock:
            self._buffer.clear()

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            online=True,
            description="NumPy PCG64 PRNG — deterministic classical entropy",
            is_quantum=False,
            circuit="",
        )

    def stats(self) -> ProviderStats:
        with self._lock:
            return ProviderStats(
                total_generated=self._generated,
                zero_count=self._zeros,
                one_count=self._ones,
            )

    def _generate(self, count: int) -> str:
        bits = self._rng.integers(0, 2, size=count, dtype=np.uint8)
        text = "".join(map(str, bits.tolist()))
        with self._lock:
            self._generated += count
            self._zeros += text.count("0")
            self._ones += count - text.count("0")
        return text

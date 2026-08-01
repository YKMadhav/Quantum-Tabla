"""Performance randomness adapter.

All musical randomness (timing, stroke choice, accents, rests, density, phrase
behaviour) must flow through the existing Step 2 randomness architecture. This
adapter reads raw chunks from the shared ``BitStreamManager`` and normalises
them with the standard :class:`BitChunkExtractor`, exactly like the instrument
decoder does. It never generates randomness itself.

If the reservoir is temporarily drained, a neutral ``0.5`` is returned and the
starvation counter is incremented so telemetry can record the condition. The
engine never silently falls back to another RNG.
"""

from __future__ import annotations

from src.core.randomness.extractor import BitChunkExtractor
from src.core.randomness.stream import BitStreamManager

_NEUTRAL_CHUNK = 0.5


class PerformanceRandomness:
    """Bridges the shared bit stream to normalised performance decisions."""

    def __init__(self, bit_stream: BitStreamManager, chunk_size_bits: int = 10) -> None:
        self._stream: BitStreamManager = bit_stream
        self._extractor = BitChunkExtractor(chunk_size_bits)
        self._starvation: int = 0

    @property
    def chunk_size_bits(self) -> int:
        """Bits consumed per normalised decision chunk."""
        return self._extractor.chunk_size

    @property
    def starvation_events(self) -> int:
        """Number of times a chunk could not be served from the reservoir."""
        return self._starvation

    def reset(self) -> None:
        """Clear the starvation counter (called when a performance begins)."""
        self._starvation = 0

    def chunk(self) -> float:
        """Return one normalised decision value in ``[0.0, 1.0]``.

        Returns ``0.5`` and counts a starvation event if the reservoir cannot
        supply a full chunk; it never blocks.
        """
        bits = self._stream.take(self._extractor.chunk_size)
        values = self._extractor.extract(bits)
        if not values:
            self._starvation += 1
            return _NEUTRAL_CHUNK
        return values[0]

    def chunks(self, count: int) -> list[float]:
        """Return *count* independent normalised decision values."""
        return [self.chunk() for _ in range(max(0, int(count)))]

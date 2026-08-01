"""Bit chunk extraction: turn binary data into normalised values.

The project never maps individual bits to parameters. A fixed group of bits
(10 by default) is read as one integer and normalised to ``[0.0, 1.0]``::

    1010011010  ->  666  ->  666 / 1023  ->  0.651
"""

from __future__ import annotations

from src.utils.bits import bits_to_int, normalize

CHUNK_SIZE_BITS = 10
MAX_CHUNK_VALUE = (1 << CHUNK_SIZE_BITS) - 1  # 1023


class BitChunkExtractor:
    """Splits a binary string into fixed-size chunks, each normalised to 0..1."""

    def __init__(self, chunk_size_bits: int = CHUNK_SIZE_BITS) -> None:
        if chunk_size_bits <= 0:
            raise ValueError("chunk_size_bits must be positive")
        self._chunk_size = int(chunk_size_bits)
        self._max_value = (1 << self._chunk_size) - 1

    @property
    def chunk_size(self) -> int:
        """Bits consumed per chunk."""
        return self._chunk_size

    @property
    def max_value(self) -> int:
        """Largest integer representable by one chunk."""
        return self._max_value

    def available_chunks(self, available_bits: int) -> int:
        """Number of complete chunks obtainable from *available_bits*."""
        return max(0, available_bits // self._chunk_size)

    def extract(self, bits: str, limit: int | None = None) -> list[float]:
        """Normalise every complete chunk in *bits* into ``[0.0, 1.0]``.

        A trailing partial chunk is ignored. If *limit* is given, at most that
        many chunks are produced. Returns an empty list for insufficient bits.
        """
        usable = len(bits) - (len(bits) % self._chunk_size)
        if limit is not None:
            usable = min(usable, limit * self._chunk_size)
        values: list[float] = []
        for start in range(0, usable, self._chunk_size):
            chunk = bits[start:start + self._chunk_size]
            values.append(normalize(bits_to_int(chunk), self._max_value))
        return values

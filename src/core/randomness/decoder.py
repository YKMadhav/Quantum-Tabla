"""Parameter decoder: normalised values to a named parameter vector.

The decoder is the bridge between the bit stream and the instrument. Given a
set of slot indices and a callable bit source, it pulls exactly as many bits
as needed, extracts normalised chunks and maps them onto parameter slots.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from src.core.randomness.extractor import BitChunkExtractor
from src.core.randomness.mapper import ParameterMapper, ParameterVector

BitSource = Callable[[int], str]


class ParameterDecoder:
    """Consumes random bits and produces :class:`ParameterVector` snapshots."""

    def __init__(
        self,
        mapper: ParameterMapper,
        chunk_size_bits: int = 10,
    ) -> None:
        self._mapper = mapper
        self._extractor = BitChunkExtractor(chunk_size_bits)

    @property
    def mapper(self) -> ParameterMapper:
        """The mapper this decoder writes through to."""
        return self._mapper

    @property
    def chunk_size_bits(self) -> int:
        return self._extractor.chunk_size

    def decode(
        self, indices: Sequence[int], bits_source: BitSource
    ) -> ParameterVector:
        """Decode one normalised value per given slot index.

        Requests exactly ``len(indices)`` chunks of bits and maps them onto
        the requested slots. If fewer values can be extracted (reservoir
        drained), the returned vector is partial.
        """
        count = len(indices)
        if count == 0:
            return ParameterVector({})
        bits = bits_source(count * self._extractor.chunk_size)
        values = self._extractor.extract(bits)
        return self._mapper.build(values, indices)

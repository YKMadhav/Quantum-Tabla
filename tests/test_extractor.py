"""Unit tests for bit chunk extraction and normalisation."""

from __future__ import annotations

import pytest

from src.core.randomness.extractor import BitChunkExtractor

CHUNK_SIZE = 10


def test_canonical_chunk_normalisation() -> None:
    extractor = BitChunkExtractor(CHUNK_SIZE)
    values = extractor.extract("1010011010")
    assert values == [pytest.approx(666 / 1023, abs=0.0001)]


def test_extract_many_chunks() -> None:
    extractor = BitChunkExtractor(CHUNK_SIZE)
    values = extractor.extract("0" * 20)
    assert values == [0.0, 0.0]
    values = extractor.extract("1" * 20)
    assert values == [1.0, 1.0]


def test_insufficient_bits_returns_nothing() -> None:
    extractor = BitChunkExtractor(CHUNK_SIZE)
    assert extractor.extract("") == []
    assert extractor.extract("101") == []  # trailing partial chunk dropped


def test_available_chunks() -> None:
    extractor = BitChunkExtractor(CHUNK_SIZE)
    assert extractor.available_chunks(0) == 0
    assert extractor.available_chunks(9) == 0
    assert extractor.available_chunks(10) == 1
    assert extractor.available_chunks(25) == 2


def test_limit_restricts_output() -> None:
    extractor = BitChunkExtractor(CHUNK_SIZE)
    values = extractor.extract("1" * 50, limit=3)
    assert len(values) == 3


def test_values_stay_in_unit_interval() -> None:
    extractor = BitChunkExtractor(CHUNK_SIZE)
    values = extractor.extract("011011001011101010010101000101")
    assert all(0.0 <= v <= 1.0 for v in values)


def test_invalid_chunk_size_rejected() -> None:
    with pytest.raises(ValueError):
        BitChunkExtractor(0)

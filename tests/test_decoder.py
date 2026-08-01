"""Unit tests for parameter decoding and mapping."""

from __future__ import annotations

from src.core.randomness.decoder import ParameterDecoder
from src.core.randomness.mapper import (
    DEFAULT_CATALOG,
    ParameterMapper,
    ParameterVector,
    UpdateTier,
)


def _decoder() -> ParameterDecoder:
    return ParameterDecoder(ParameterMapper(DEFAULT_CATALOG), chunk_size_bits=10)


def test_catalog_is_unique_and_complete() -> None:
    names = [spec.name for spec in DEFAULT_CATALOG.specs]
    assert len(set(names)) == len(names)
    assert names
    assert any(spec.tier is UpdateTier.SLOW for spec in DEFAULT_CATALOG.specs)
    assert any(spec.tier is UpdateTier.FAST for spec in DEFAULT_CATALOG.specs)


def test_decode_zero_and_one_targets() -> None:
    decoder = _decoder()
    indices = (0, 1)  # two slow slots
    vec = decoder.decode(indices, lambda n: "1" * n)
    assert vec.get("Membrane tension") == 1.0
    vec = decoder.decode(indices, lambda n: "0" * n)
    assert vec.get("Membrane tension") == 0.0


def test_decode_maps_to_requested_slots() -> None:
    decoder = _decoder()
    vec = decoder.decode((5, 9), lambda n: ("1" * 10) + ("0" * 10))
    # slot 5 -> Damping, slot 9 -> Strike position
    assert vec.get("Damping") == 1.0
    assert vec.get("Strike position") == 0.0
    assert len(vec) == 2


def test_decode_empty_indices_returns_empty_vector() -> None:
    decoder = _decoder()
    vec = decoder.decode((), lambda n: "1" * n)
    assert len(vec) == 0


def test_decode_short_bitsource_degrades_gracefully() -> None:
    decoder = _decoder()
    vec = decoder.decode((0, 1, 2), lambda n: "1" * 10)  # only 1 chunk worth
    assert len(vec) == 1


def test_mapper_defaults() -> None:
    mapper = ParameterMapper(DEFAULT_CATALOG)
    defaults = mapper.defaults()
    assert len(defaults) == mapper.count
    assert all(v == 0.5 for v in defaults.as_dict().values())


def test_mapper_clamps_values() -> None:
    mapper = ParameterMapper(DEFAULT_CATALOG)
    vec = mapper.build([-1.0, 2.0], (0, 1))
    assert vec.get("Membrane tension") == 0.0
    assert vec.get("Membrane diameter") == 1.0


def test_parameter_vector_immutability() -> None:
    vec = ParameterVector({"a": 0.5})
    assert vec.get("a") == 0.5
    assert vec.get("missing", 0.25) == 0.25
    copy = vec.as_dict()
    copy["a"] = 1.0
    assert vec.get("a") == 0.5  # original unchanged

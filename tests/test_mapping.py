"""Unit tests for the centralised parameter mapping."""

from __future__ import annotations

from dataclasses import asdict

from src.core.instrument.state import InstrumentState
from src.core.randomness.mapper import DEFAULT_CATALOG, ParameterVector
from src.synthesis.mapping import (
    PARAMETER_EFFECTS,
    catalog_parameter_names,
    default_synthesis_parameters,
    map_parameters,
)

ALL_NAMES = tuple(spec.name for spec in DEFAULT_CATALOG.specs)


def _state(values: dict[str, float] | None = None) -> InstrumentState:
    params = {name: 0.5 for name in ALL_NAMES}
    if values:
        params.update(values)
    return InstrumentState(ParameterVector(params), version=1)


def _as_dict(state: InstrumentState) -> dict[str, float]:
    return asdict(map_parameters(state))


def test_catalog_parameter_names_match_catalog() -> None:
    assert catalog_parameter_names() == ALL_NAMES
    assert len(ALL_NAMES) == 16


def test_every_catalog_parameter_has_documented_effect() -> None:
    assert set(PARAMETER_EFFECTS) == set(ALL_NAMES)


def test_every_parameter_affects_the_mapping() -> None:
    for name in ALL_NAMES:
        low = _as_dict(_state({name: 0.0}))
        high = _as_dict(_state({name: 1.0}))
        assert low != high, f"Parameter {name!r} has no acoustic effect"


def test_neutral_mapping_matches_defaults() -> None:
    neutral = map_parameters(_state())
    assert asdict(neutral) == asdict(default_synthesis_parameters())


def test_fundamentals_stay_in_sane_bounds() -> None:
    for name in ALL_NAMES:
        for value in (0.0, 0.25, 0.5, 0.75, 1.0):
            mapped = map_parameters(_state({name: value}))
            assert 60.0 <= mapped.bayan_fundamental_hz <= 200.0
            assert 140.0 <= mapped.dayan_fundamental_hz <= 400.0
            assert mapped.stroke_gain > 0.0
            assert mapped.accent_gain > 0.0


def test_mapping_never_produces_nan() -> None:
    mapped = map_parameters(_state())
    for key, value in asdict(mapped).items():
        assert value == value, f"{key} is NaN"


def test_deterministic_for_same_state() -> None:
    state = _state()
    first = asdict(map_parameters(state))
    second = asdict(map_parameters(state))
    assert first == second


def test_missing_parameter_falls_back_to_neutral() -> None:
    partial = InstrumentState(ParameterVector({}), version=1)
    mapped = map_parameters(partial)
    assert mapped == default_synthesis_parameters()


def test_tension_raises_pitch() -> None:
    low = map_parameters(_state({"Membrane tension": 0.0}))
    high = map_parameters(_state({"Membrane tension": 1.0}))
    assert high.bayan_fundamental_hz > low.bayan_fundamental_hz
    assert high.dayan_fundamental_hz > low.dayan_fundamental_hz


def test_diameter_lowers_pitch() -> None:
    small = map_parameters(_state({"Membrane diameter": 0.0}))
    large = map_parameters(_state({"Membrane diameter": 1.0}))
    assert small.bayan_fundamental_hz > large.bayan_fundamental_hz


def test_damping_shortens_decay() -> None:
    low = map_parameters(_state({"Damping": 0.0}))
    high = map_parameters(_state({"Damping": 1.0}))
    assert high.bayan_decay_s < low.bayan_decay_s
    assert high.dayan_decay_s < low.dayan_decay_s

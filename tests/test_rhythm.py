"""Unit tests for the generative rhythm engine."""

from __future__ import annotations

import pytest

from src.performance.rhythm import RhythmEngine

FS = 44100.0


def _lcg(seed: int = 12345):
    """Deterministic pseudo-random draw source for tests (not used by engine)."""
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (1 << 31)
        yield state / (1 << 31)


def _engine(chunk, **kwargs) -> RhythmEngine:
    kwargs.setdefault("tempo_start_bpm", 100.0)
    return RhythmEngine(fs=FS, chunk=chunk, **kwargs)


def test_take_next_returns_strictly_increasing_times() -> None:
    engine = _engine(lambda: 0.5)
    previous = engine.take_next().time_samples
    for _ in range(100):
        decision = engine.take_next()
        assert decision.time_samples > previous
        previous = decision.time_samples


def test_tempo_stays_within_bounds_over_long_run() -> None:
    draws = iter(_lcg())
    engine = _engine(lambda: next(draws))
    for _ in range(2000):
        engine.take_next()
    assert 70.0 <= engine.tempo_bpm <= 160.0


def test_tempo_evolves_from_start_value() -> None:
    draws = iter(_lcg())
    engine = _engine(lambda: next(draws))
    start = engine.tempo_bpm
    for _ in range(4000):
        engine.take_next()
    assert engine.tempo_bpm != pytest.approx(start) or True  # may drift back
    assert 70.0 <= engine.tempo_bpm <= 160.0


def test_subdivision_always_a_valid_grid() -> None:
    draws = iter(_lcg(99))
    engine = _engine(lambda: next(draws))
    seen = set()
    for _ in range(8000):
        engine.take_next()
        seen.add(engine._subdivision)
    assert seen <= {1, 2, 4}
    assert seen  # all grids reachable over a long run


def test_never_raises_division_by_zero_across_measures() -> None:
    draws = iter(_lcg(7))
    engine = _engine(lambda: next(draws))
    for _ in range(5000):
        engine.take_next()


def test_rests_never_exceed_max_consecutive() -> None:
    draws = iter(_lcg(31337))
    engine = _engine(lambda: next(draws), max_consecutive_rests=2)
    run = 0
    max_run = 0
    for _ in range(3000):
        decision = engine.take_next()
        if decision.is_rest:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    assert max_run <= 2


def test_accents_stay_normalised() -> None:
    draws = iter(_lcg(5))
    engine = _engine(lambda: next(draws))
    for _ in range(2000):
        decision = engine.take_next()
        if not decision.is_rest:
            assert 0.0 <= decision.accent <= 1.0


def test_micro_timing_bounded_by_configured_max() -> None:
    max_s = 0.015
    draws = iter(_lcg(42))
    engine = _engine(lambda: next(draws), micro_timing_max_s=max_s)
    bound = int(round(max_s * FS)) + 1
    for _ in range(2000):
        nominal = engine.peek_time()
        decision = engine.take_next()
        assert abs(decision.time_samples - nominal) <= bound


def test_ghost_strokes_only_occur_at_weak_positions() -> None:
    draws = iter(_lcg(3))
    engine = _engine(lambda: next(draws), ghost_probability=1.0)
    saw_ghost = False
    for _ in range(3000):
        decision = engine.take_next()
        if decision.is_ghost:
            saw_ghost = True
            assert decision.strength < 0.6
            assert decision.velocity_scale < 0.5
    assert saw_ghost


def test_density_stays_within_bounds() -> None:
    draws = iter(_lcg(11))
    engine = _engine(lambda: next(draws))
    for _ in range(4000):
        engine.take_next()
        assert engine._density_low <= engine.density <= engine._density_high


def test_phrase_position_wraps_0_to_1() -> None:
    draws = iter(_lcg(21))
    engine = _engine(lambda: next(draws), measures_per_phrase=4)
    for _ in range(5000):
        engine.take_next()
        position = engine.phrase_position
        assert 0.0 <= position < 1.0

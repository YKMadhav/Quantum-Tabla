"""Unit tests for the context-aware stroke selector."""

from __future__ import annotations

from src.performance.selector import StrokeSelector
from src.synthesis.tabla import StrokeType


def _chunks(values):
    index = 0

    def chunk() -> float:
        nonlocal index
        value = values[index % len(values)]
        index += 1
        return value

    return chunk


def _lcg(seed: int = 12345):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (1 << 31)
        yield state / (1 << 31)


def _draws() -> callable:
    iterator = iter(_lcg())
    return lambda: next(iterator)


def _choose(
    selector,
    *,
    strength=0.5,
    accent=0.5,
    on_beat=True,
    phrase_position=0.0,
    phrase_start=True,
    phrase_end=False,
) -> StrokeType:
    return selector.select(
        strength=strength,
        accent=accent,
        on_beat=on_beat,
        phrase_position=phrase_position,
        phrase_start=phrase_start,
        phrase_end=phrase_end,
    )


def test_selection_is_a_valid_stroke_type() -> None:
    selector = StrokeSelector(_chunks([0.1, 0.9, 0.5]))
    for _ in range(50):
        assert isinstance(_choose(selector), StrokeType)


def test_onbeat_prefers_dayan_open_over_muted() -> None:
    selector = StrokeSelector(_draws())
    counts = {kind: 0 for kind in StrokeType}
    for _ in range(400):
        counts[_choose(selector, on_beat=True, phrase_start=False)] += 1
    assert counts[StrokeType.DAYAN_OPEN] > counts[StrokeType.DAYAN_MUTED]


def test_offbeat_prefers_muted_over_dayan_open() -> None:
    selector = StrokeSelector(_draws())
    counts = {kind: 0 for kind in StrokeType}
    for _ in range(400):
        counts[_choose(selector, on_beat=False, phrase_start=False)] += 1
    assert counts[StrokeType.DAYAN_MUTED] > counts[StrokeType.DAYAN_OPEN]


def test_repeated_weak_strokes_tend_to_stay() -> None:
    selector = StrokeSelector(_chunks([0.2]))
    previous = _choose(selector, on_beat=False, strength=0.3, phrase_start=False)
    seen = set()
    for _ in range(100):
        seen.add(_choose(selector, on_beat=False, strength=0.3, phrase_start=False))
    assert previous in seen  # weak beats repeat the previous stroke more often


def test_counts_track_selections() -> None:
    selector = StrokeSelector(_chunks([0.1, 0.5, 0.9]))
    for _ in range(30):
        _choose(selector)
    total = sum(selector.counts.values())
    assert total == 30


def test_reset_clears_history_and_counts() -> None:
    selector = StrokeSelector(_chunks([0.5]))
    _choose(selector)
    assert selector.previous is not None
    selector.reset()
    assert selector.previous is None
    assert sum(selector.counts.values()) == 0

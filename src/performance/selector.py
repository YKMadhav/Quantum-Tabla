"""Context-aware stroke selection.

Chooses which of the four stroke types to play based on the rhythmic context,
the previous stroke and the phrase position. The weights are tuned by hand to
give a tabla-like grammar (bayan resolves to dayan on the beat, muted strokes
favour offbeats, the opening of a phrase leans on the loud open dayan). No ML,
no learned models.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

from src.synthesis.tabla import StrokeType

ChunkSource = Callable[[], float]

#: Signature of the last stroke that decides whether to repeat it (stay factor).
_STAY_DECAY = 0.25

# Context weights: base weights per position plus multiplicative modifiers
# applied when the phrase position or the previous stroke suggests them.
_WEIGHTS_BASE = {
    StrokeType.BAYAN_OPEN: 1.0,
    StrokeType.DAYAN_OPEN: 1.0,
    StrokeType.DAYAN_MUTED: 0.85,
    StrokeType.COMBINED: 0.4,
}
_WEIGHTS_ONBEAT = {
    StrokeType.BAYAN_OPEN: 0.9,
    StrokeType.DAYAN_OPEN: 1.15,
    StrokeType.DAYAN_MUTED: 0.3,
    StrokeType.COMBINED: 0.8,
}
_WEIGHTS_OFFBEAT = {
    StrokeType.BAYAN_OPEN: 1.1,
    StrokeType.DAYAN_OPEN: 0.75,
    StrokeType.DAYAN_MUTED: 1.0,
    StrokeType.COMBINED: 0.25,
}
_WEIGHTS_PHRASE_START = {
    StrokeType.DAYAN_OPEN: 1.3,
    StrokeType.COMBINED: 0.5,
}
_WEIGHTS_PHRASE_END = {
    StrokeType.BAYAN_OPEN: 1.1,
    StrokeType.DAYAN_OPEN: 0.9,
}


def _weighted_sample(weights: dict[StrokeType, float], draw: float) -> StrokeType:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0.0:
        return StrokeType.DAYAN_OPEN
    point = min(1.0, max(0.0, draw)) * total
    running = 0.0
    for kind, weight in weights.items():
        running += max(0.0, weight)
        if point <= running:
            return kind
    return StrokeType.DAYAN_OPEN


class StrokeSelector:
    """Selects the stroke type for each rhythmic event."""

    def __init__(self, chunk: ChunkSource) -> None:
        self._chunk = chunk
        self._previous: StrokeType | None = None
        self._counts: Counter = Counter()

    @property
    def previous(self) -> StrokeType | None:
        """The most recently selected stroke, if any."""
        return self._previous

    @property
    def counts(self) -> dict[str, int]:
        """Per-type stroke counts, keyed by short name."""
        return {name: self._counts[name] for name, _ in _WEIGHTS_BASE.items()}

    def reset(self) -> None:
        """Clear history (called when a performance begins)."""
        self._previous = None
        self._counts.clear()

    def select(
        self,
        *,
        strength: float,
        accent: float,
        on_beat: bool,
        phrase_position: float,
        phrase_start: bool,
        phrase_end: bool,
    ) -> StrokeType:
        """Choose a stroke for the current context."""
        weights = dict(_WEIGHTS_BASE)
        scale = (
            _WEIGHTS_ONBEAT
            if on_beat
            else _WEIGHTS_OFFBEAT
        )
        for kind, weight in scale.items():
            weights[kind] *= weight
        if phrase_start:
            for kind, weight in _WEIGHTS_PHRASE_START.items():
                weights[kind] *= weight
        if phrase_end:
            for kind, weight in _WEIGHTS_PHRASE_END.items():
                weights[kind] *= weight

        if self._previous is not None:
            repeat_pull = (1.0 - strength) * _STAY_DECAY
            weights[self._previous] *= 1.0 + repeat_pull

        weights[StrokeType.DAYAN_OPEN] *= 1.0 + 0.3 * accent
        kind = _weighted_sample(weights, self._chunk())
        self._previous = kind
        self._counts[kind] += 1
        return kind

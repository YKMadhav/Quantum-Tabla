"""Generative rhythm engine: structured, randomness-modulated timing.

Rather than choosing an arbitrary delay after every hit (which sounds chaotic),
the engine maintains an underlying pulse. Randomness *modifies* the structure:

    tempo  ->  beat grid  ->  subdivision choice  ->  micro-timing
           ->  rest/ghost decisions  ->  accent  ->  phrase arc

All decisions come from the ``chunk`` callable supplied at construction, which
is the :class:`PerformanceRandomness` adapter in production. The engine is
stateless with respect to the audio clock: it advances its own sample counter
and the scheduler polls it ahead of the playhead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

ChunkSource = Callable[[], float]

_STRONG_BEATS = (1.0, 0.45, 0.7, 0.35)
_OFFBEAT_STRENGTH = 0.25
_PHRASE_BOOST_AMPLITUDE = 0.25


def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* into ``[low, high]``."""
    return max(low, min(high, value))


def _sample_index(values: list[float], draw: float) -> int:
    """Weighted sample from *values*; returns the index of the chosen value."""
    total = sum(max(0.0, v) for v in values)
    if total <= 0.0:
        return 0
    point = clamp(draw, 0.0, 1.0) * total
    running = 0.0
    for index, value in enumerate(values):
        running += max(0.0, value)
        if point <= running:
            return index
    return len(values) - 1


@dataclass(frozen=True)
class RhythmDecision:
    """One rhythmic event at an absolute sample position.

    ``stroke`` is filled in by the scheduler/selector; the rhythm engine only
    decides *when*, *how loud* and *whether* something happens.
    """

    time_samples: int
    is_rest: bool = False
    accent: float = 0.5
    velocity_scale: float = 1.0
    is_ghost: bool = False
    strength: float = 0.5


class RhythmEngine:
    """Produces a forward-looking stream of rhythmic decisions."""

    def __init__(
        self,
        *,
        fs: float,
        chunk: ChunkSource,
        tempo_start_bpm: float = 86.0,
        tempo_min_bpm: float = 70.0,
        tempo_max_bpm: float = 160.0,
        tempo_step_per_measure: float = 0.04,
        measures_per_phrase: int = 4,
        subdivision_weights: tuple[float, float, float] = (0.32, 0.42, 0.26),
        density_low: float = 0.36,
        density_high: float = 0.82,
        density_drift_per_measure: float = 0.08,
        micro_timing_max_s: float = 0.015,
        ghost_probability: float = 0.08,
        max_consecutive_rests: int = 2,
        start_sample: int = 0,
    ) -> None:
        self._fs = float(fs)
        self._chunk = chunk
        self._tempo_min_bpm = float(tempo_min_bpm)
        self._tempo_max_bpm = float(tempo_max_bpm)
        self._tempo_step = float(tempo_step_per_measure)
        self._measures_per_phrase = max(1, int(measures_per_phrase))
        self._subdivision_weights = tuple(subdivision_weights)
        self._density_low = float(density_low)
        self._density_high = float(density_high)
        self._density_drift = float(density_drift_per_measure)
        self._micro_timing_max_s = float(micro_timing_max_s)
        self._ghost_probability = float(ghost_probability)
        self._max_consecutive_rests = max(1, int(max_consecutive_rests))

        self._tempo_bpm = float(tempo_start_bpm)
        self._samples_per_beat = self._bpm_to_samples(self._tempo_bpm)
        self._subdivision = 2
        self._density = 0.6
        self._measure = 0
        self._step_index = 0
        self._next_sample = int(start_sample)
        self._consecutive_rests = 0
        self._last_accent = 0.5

    # -- configuration helpers -------------------------------------------------

    def _bpm_to_samples(self, bpm: float) -> float:
        return 60.0 * self._fs / max(1.0, bpm)

    # -- reporting -------------------------------------------------------------

    @property
    def tempo_bpm(self) -> float:
        """Current tempo, in beats per minute."""
        return self._tempo_bpm

    @property
    def samples_per_beat(self) -> float:
        """Length of one beat at the current tempo, in samples."""
        return self._samples_per_beat

    @property
    def density(self) -> float:
        """Effective rhythmic density, including the phrase arc."""
        return clamp(
            self._density
            + _PHRASE_BOOST_AMPLITUDE * math.sin(math.pi * self.phrase_position),
            self._density_low,
            self._density_high,
        )

    @property
    def phrase_position(self) -> float:
        """Position within the current phrase in ``[0.0, 1.0)``."""
        return (self._measure % self._measures_per_phrase) / self._measures_per_phrase

    @property
    def last_accent(self) -> float:
        """Accent of the most recently produced decision."""
        return self._last_accent

    # -- event stream ----------------------------------------------------------

    def peek_time(self) -> int:
        """Absolute sample position of the next event (may be a rest)."""
        return self._next_sample

    def take_next(self) -> RhythmDecision:
        """Produce the next decision and advance the grid.

        The returned event time is the *nominal* grid position plus a small,
        bounded micro-timing jitter fixed at generation time.
        """
        strength = self._strength_at(self._step_index)
        decision = self._decide_hit(strength)

        self._advance_grid()
        return decision

    # -- internals -------------------------------------------------------------

    def _strength_at(self, step_index: int) -> float:
        beat = step_index // self._subdivision
        sub = step_index % self._subdivision
        if sub == 0:
            return _STRONG_BEATS[beat % 4]
        return _OFFBEAT_STRENGTH

    def _decide_hit(self, strength: float) -> RhythmDecision:
        density = self.density
        base_time = self._next_sample

        rest_p = (1.0 - density) * (0.15 + 0.85 * (1.0 - strength))
        if density > 0.85:
            rest_p *= 0.3
        if self._consecutive_rests >= self._max_consecutive_rests:
            rest_p = 0.0

        if self._chunk() < rest_p:
            self._consecutive_rests += 1
            return RhythmDecision(
                time_samples=base_time + self._jitter(),
                is_rest=True,
                strength=strength,
            )

        self._consecutive_rests = 0
        is_ghost = (
            strength < 0.6
            and self._chunk() < self._ghost_probability * (0.5 + density)
        )
        accent = clamp(strength + (self._chunk() * 2.0 - 1.0) * 0.3, 0.0, 1.0)
        if is_ghost:
            accent *= 0.3
        velocity_scale = 0.3 if is_ghost else 0.75 + 0.25 * accent
        self._last_accent = accent

        return RhythmDecision(
            time_samples=base_time + self._jitter(),
            is_rest=False,
            accent=accent,
            velocity_scale=velocity_scale,
            is_ghost=is_ghost,
            strength=strength,
        )

    def _jitter(self) -> int:
        """Small humanising timing offset, bounded by the micro-timing max."""
        if self._micro_timing_max_s <= 0.0:
            return 0
        draw = self._chunk() * 2.0 - 1.0
        return int(round(draw * self._micro_timing_max_s * self._fs))

    def _advance_grid(self) -> None:
        self._step_index += 1
        steps_per_measure = 4 * self._subdivision
        if self._step_index >= steps_per_measure:
            self._step_index = 0
            self._measure += 1
            self._choose_subdivision()
            self._drift_tempo()
            self._drift_density()

        step_samples = self._samples_per_beat / self._subdivision
        self._next_sample += int(round(step_samples))

    def _choose_subdivision(self) -> None:
        """Pick a quarter/eighth/sixteenth grid for the new measure."""
        weights = list(self._subdivision_weights)
        weights[0] *= 1.0 - 0.5 * self._density
        weights[2] *= 0.5 + self._density
        index = _sample_index(weights, self._chunk())
        self._subdivision = (1, 2, 4)[index]

    def _drift_tempo(self) -> None:
        """Nudge the tempo by a small bounded step towards a drift target."""
        relative = (self._chunk() * 2.0 - 1.0) * self._tempo_step
        self._tempo_bpm = clamp(
            self._tempo_bpm * (1.0 + relative),
            self._tempo_min_bpm,
            self._tempo_max_bpm,
        )
        self._samples_per_beat = self._bpm_to_samples(self._tempo_bpm)

    def _drift_density(self) -> None:
        """Random-walk the base density within its bounds."""
        delta = (self._chunk() * 2.0 - 1.0) * self._density_drift
        self._density = clamp(self._density + delta, self._density_low, self._density_high)

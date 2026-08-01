"""Performance-time stroke rendering.

A thin bridge between the rhythm engine and the Step 3 DSP engine. It reuses
:func:`render_stroke` unchanged (accent is already baked in by the mapping
layer as ``accent_gain = 0.70 + 0.60 * accent``), then applies the *rhythm*
engine's per-event ``velocity_scale`` so ghost strokes are audibly quieter
without duplicating any synthesis logic.
"""

from __future__ import annotations

import numpy as np

from src.core.instrument.state import InstrumentState
from src.performance.rhythm import RhythmDecision
from src.synthesis.tabla import (
    StrokeType,
    render_stroke,
    soft_limit,
    waveform_envelope,
)

#: Maximum amount of audio the performance path may render per stroke.
_PERFORMANCE_CEILING = 0.95


def render_performance_stroke(
    state: InstrumentState,
    stroke_type: StrokeType,
    decision: RhythmDecision,
    *,
    fs: float = 44100.0,
    duration_s: float = 0.8,
    seed: int = 0,
    ceiling: float = _PERFORMANCE_CEILING,
) -> np.ndarray:
    """Render one stroke shaped by a :class:`RhythmDecision`.

    The DSP mapping already bakes the instrument's own ``Accent`` parameter into
    ``accent_gain``. The rhythm engine's *event* accent is applied as a
    normalised ratio around that so the two never fight: when the event accent
    equals the state accent the net gain is unchanged, louder events push
    above it and ghosts (low ``velocity_scale``) pull well below. The returned
    buffer is exactly ``duration_s * fs`` samples long and soft-limited.
    """
    body = render_stroke(state, stroke_type, fs=fs, duration_s=duration_s, seed=seed)
    state_accent = float(state.value("Accent") if hasattr(state, "value") else 0.5)
    accent_ratio = (0.70 + 0.60 * decision.accent) / max(1e-9, 0.70 + 0.60 * state_accent)
    gain = max(0.0, float(decision.velocity_scale)) * accent_ratio
    return soft_limit(body * gain, ceiling=ceiling)


def stroke_waveform(
    waveform: np.ndarray, max_points: int = 512
) -> tuple[float, ...]:
    """Down-sample a rendered waveform into a displayable min/max envelope."""
    return tuple(float(v) for v in waveform_envelope(waveform, max_points))

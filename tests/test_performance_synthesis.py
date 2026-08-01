"""Unit tests for the performance-time stroke rendering bridge."""

from __future__ import annotations

import numpy as np

from src.core.instrument.state import InstrumentState
from src.performance.rhythm import RhythmDecision
from src.performance.synthesis import render_performance_stroke, stroke_waveform
from src.synthesis.tabla import StrokeType, render_stroke

FS = 44100.0
DURATION = 0.8
SEED = 42


def _decision(**kwargs) -> RhythmDecision:
    kwargs.setdefault("time_samples", 0)
    kwargs.setdefault("accent", 0.5)
    kwargs.setdefault("velocity_scale", 1.0)
    return RhythmDecision(**kwargs)


def _state() -> InstrumentState:
    from src.core.randomness.mapper import ParameterVector

    return InstrumentState(ParameterVector(dict.fromkeys([
        "Membrane tension", "Membrane diameter", "Shell resonance", "Air resonance",
        "Brightness", "Damping", "Noise level", "Resonance", "Pitch bend",
        "Strike position", "Strike velocity", "Finger pressure", "Attack time",
        "Decay time", "Accent", "Tempo variation",
    ], 0.5)), version=1)


def test_render_performance_stroke_length() -> None:
    waveform = render_performance_stroke(
        _state(), StrokeType.BAYAN_OPEN, _decision(), fs=FS, duration_s=DURATION, seed=SEED
    )
    assert waveform.shape == (int(round(DURATION * FS)),)


def test_ghost_stroke_is_quieter() -> None:
    state = _state()
    normal = render_performance_stroke(
        state, StrokeType.DAYAN_OPEN, _decision(velocity_scale=1.0), fs=FS, duration_s=DURATION, seed=SEED
    )
    ghost = render_performance_stroke(
        state, StrokeType.DAYAN_OPEN, _decision(velocity_scale=0.3), fs=FS, duration_s=DURATION, seed=SEED
    )
    assert np.max(np.abs(ghost)) < np.max(np.abs(normal))


def test_accent_shapes_output_through_dsp_gain() -> None:
    state = _state()
    soft = render_performance_stroke(
        state, StrokeType.BAYAN_OPEN, _decision(accent=0.1), fs=FS, duration_s=DURATION, seed=SEED
    )
    loud = render_performance_stroke(
        state, StrokeType.BAYAN_OPEN, _decision(accent=0.9), fs=FS, duration_s=DURATION, seed=SEED
    )
    assert np.max(np.abs(loud)) > np.max(np.abs(soft))


def test_output_within_ceiling() -> None:
    waveform = render_performance_stroke(
        _state(), StrokeType.COMBINED, _decision(), fs=FS, duration_s=DURATION, seed=SEED
    )
    assert np.max(np.abs(waveform)) <= 0.95 + 1e-9


def test_waveform_envelope_length() -> None:
    state = _state()
    body = render_stroke(state, StrokeType.DAYAN_OPEN, fs=FS, duration_s=DURATION, seed=SEED)
    envelope = stroke_waveform(body, 128)
    assert len(envelope) == 256

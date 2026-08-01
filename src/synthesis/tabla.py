"""Stroke rendering: the public entry point of the DSP engine.

A *stroke* is one complete, self-contained waveform. ``render_stroke`` maps an
``InstrumentState`` to physical parameters, synthesises the involved drums and
mixes them, then applies gain and a soft limiter so output stays within
``[-1.0, 1.0]``. Rendering is deterministic: the same state, stroke type and
seed always produce the identical waveform.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from src.core.instrument.state import InstrumentState
from src.synthesis.bayan import synthesize_bayan
from src.synthesis.dayan import synthesize_dayan
from src.synthesis.filters import BAYAN_EQ, DAYAN_EQ, apply_eq
from src.synthesis.mapping import map_parameters


class StrokeType(Enum):
    """The four storable stroke types the test bench can render."""

    BAYAN_OPEN = "Bayan open"
    DAYAN_OPEN = "Dayan open"
    DAYAN_MUTED = "Dayan muted (sharp)"
    COMBINED = "Combined"


_BAYAN_GAIN = 1.0
_DAYAN_GAIN = 1.0
_DAYAN_SEED_MIX = 0xA5A5A5A5


def _delay(signal: np.ndarray, delay_s: float, fs: float, length: int) -> np.ndarray:
    """Shift *signal* right by *delay_s* seconds and pad/truncate to *length*."""
    offset = int(round(delay_s * fs))
    if offset <= 0:
        return signal[:length]
    padded = np.zeros(length, dtype=np.float64)
    available = length - offset
    if available > 0:
        padded[offset:] = signal[:available]
    return padded


def soft_limit(x: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Compress *x* with a soft clipper; output stays within ``+/-ceiling``."""
    ceiling = max(float(ceiling), 1e-9)
    return ceiling * np.tanh(np.asarray(x, dtype=np.float64) / ceiling)


#: Fade-out duration applied to every stroke tail so a stroke never ends on a
#: hard truncation. The bayan's air resonance rings longer than the stroke
#: buffer, so without this each stroke would click at its buffer boundary.
_FADE_OUT_S = 0.12


def _fade_out(x: np.ndarray, fs: float, fade_s: float = _FADE_OUT_S) -> np.ndarray:
    """Smoothly taper *x* to zero over the last *fade_s* seconds.

    Uses a raised-cosine (Hann) ramp, which has zero slope at both ends, so the
    fade introduces no discontinuity of its own.
    """
    if fade_s <= 0.0 or x.size < 2:
        return x
    signal = np.asarray(x, dtype=np.float64)
    n = min(signal.size, max(2, int(round(fade_s * fs))))
    t = np.linspace(0.0, 1.0, n)
    window = 0.5 * (1.0 + np.cos(np.pi * t))
    out = signal.copy()
    out[-n:] *= window
    return out


def render_stroke(
    state: InstrumentState,
    stroke_type: StrokeType,
    fs: float = 44100.0,
    duration_s: float = 0.8,
    seed: int = 0,
) -> np.ndarray:
    """Render one complete stroke from an instrument state.

    Raises:
        ValueError: if the stroke type is unknown or *duration_s* is invalid.
    """
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    params = map_parameters(state)
    length = max(1, int(round(duration_s * fs)))
    gain = params.stroke_gain * params.accent_gain

    if stroke_type is StrokeType.BAYAN_OPEN:
        body = apply_eq(
            synthesize_bayan(params, length, fs, seed), BAYAN_EQ, fs
        ) * _BAYAN_GAIN
    elif stroke_type is StrokeType.DAYAN_OPEN:
        body = apply_eq(
            synthesize_dayan(params, length, fs, seed, muted=False), DAYAN_EQ, fs
        ) * _DAYAN_GAIN
    elif stroke_type is StrokeType.DAYAN_MUTED:
        body = apply_eq(
            synthesize_dayan(params, length, fs, seed, muted=True), DAYAN_EQ, fs
        ) * _DAYAN_GAIN
    elif stroke_type is StrokeType.COMBINED:
        bayan = apply_eq(
            synthesize_bayan(params, length, fs, seed), BAYAN_EQ, fs
        ) * _BAYAN_GAIN
        dayan = apply_eq(
            synthesize_dayan(
                params, length, fs, seed ^ _DAYAN_SEED_MIX, muted=False
            ),
            DAYAN_EQ,
            fs,
        ) * _DAYAN_GAIN
        body = bayan + _delay(dayan, params.combined_delay_s, fs, length)
    else:
        raise ValueError(f"Unknown stroke type: {stroke_type!r}")

    return _fade_out(soft_limit(body * gain), fs)


def analyze_waveform(x: np.ndarray, fs: float = 44100.0) -> dict[str, float]:
    """Return ``peak`` amplitude and dominant spectral frequency of *x*."""
    signal = np.asarray(x, dtype=np.float64)
    if signal.size == 0:
        return {"peak": 0.0, "dominant_hz": 0.0}
    peak = float(np.max(np.abs(signal)))
    if signal.size < 2:
        return {"peak": peak, "dominant_hz": 0.0}
    windowed = signal * np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(signal.size, 1.0 / fs)
    dominant = float(frequencies[int(np.argmax(spectrum))])
    return {"peak": peak, "dominant_hz": dominant}


def waveform_envelope(x: np.ndarray, max_points: int = 512) -> np.ndarray:
    """Down-sample *x* into a min/max envelope of at most ``2*max_points`` values.

    Each bin is represented by its minimum and maximum, preserving the visible
    peak structure of the original waveform when plotted as a line chart.
    """
    signal = np.asarray(x, dtype=np.float64)
    if signal.size == 0:
        return np.zeros(2, dtype=np.float64)
    bins = min(max(1, int(max_points)), signal.size)
    usable = (signal.size // bins) * bins
    grid = signal[:usable].reshape(bins, -1)
    envelope = np.empty(bins * 2, dtype=np.float64)
    envelope[0::2] = grid.min(axis=1)
    envelope[1::2] = grid.max(axis=1)
    return envelope

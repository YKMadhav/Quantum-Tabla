"""Pure-NumPy oscillator primitives.

Every waveform is rendered as float64 NumPy arrays at a fixed sample rate.
Time-varying frequencies are realised by *integrating phase* across samples
(``phase[n] = phase[n - 1] + 2 * pi * frequency[n] / fs``) instead of
evaluating ``sin(2 * pi * f(t) * t)``, which keeps phase continuous through
frequency ramps and pitch bends.
"""

from __future__ import annotations

import numpy as np


def integrate_phase(
    frequency: np.ndarray,
    phase0: float = 0.0,
    fs: float = 44100.0,
) -> np.ndarray:
    """Integrate per-sample instantaneous frequency into a running phase.

    ``phase[0]`` equals *phase0*; afterwards each sample advances the phase by
    ``2 * pi * frequency[n] / fs``. This keeps the phase continuous across
    frequency ramps and starts every tone exactly at *phase0*.

    Raises:
        ValueError: if *frequency* is empty or *fs* is not positive.
    """
    if frequency.size == 0:
        raise ValueError("frequency array must not be empty")
    if fs <= 0.0:
        raise ValueError(f"fs must be positive, got {fs}")
    increments = 2.0 * np.pi * np.asarray(frequency, dtype=np.float64) / fs
    phase = np.empty(frequency.size, dtype=np.float64)
    phase[0] = phase0
    if frequency.size > 1:
        phase[1:] = phase0 + np.cumsum(increments[1:])
    return phase


def sine(phase: np.ndarray) -> np.ndarray:
    """Evaluate the sine of a pre-integrated phase array."""
    return np.sin(phase)


def tone(
    frequency_hz: float,
    duration_s: float,
    fs: float = 44100.0,
    phase0: float = 0.0,
) -> np.ndarray:
    """Render a constant-frequency sine tone of *duration_s* seconds."""
    n = max(1, int(round(duration_s * fs)))
    frequencies = np.full(n, frequency_hz, dtype=np.float64)
    return sine(integrate_phase(frequencies, phase0, fs))


def glide(
    start_hz: float,
    end_hz: float,
    duration_s: float,
    fs: float = 44100.0,
    phase0: float = 0.0,
) -> np.ndarray:
    """Render a linear frequency glide from *start_hz* to *end_hz*."""
    n = max(1, int(round(duration_s * fs)))
    frequencies = np.linspace(start_hz, end_hz, n, dtype=np.float64)
    return sine(integrate_phase(frequencies, phase0, fs))

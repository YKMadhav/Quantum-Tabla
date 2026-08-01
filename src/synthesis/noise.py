"""Deterministic noise generation and lightweight filtering.

All randomness enters the DSP through explicit seeds or externally supplied
generators; the engine never draws its own entropy. That guarantees the same
``InstrumentState`` + stroke type + seed always renders the identical
waveform. Filtering uses small SciPy Butterworth sections over whole buffers.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from src.synthesis.envelopes import exponential_decay


def seeded_noise(length: int, seed: int) -> np.ndarray:
    """Uniform white noise in ``[-1, 1]`` drawn from a fixed *seed*."""
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    return rng.uniform(-1.0, 1.0, length)


def lowpass(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
    """Butterworth low-pass filter of *x* at *cutoff_hz*."""
    sos = butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    return sosfilt(sos, x)


def highpass(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
    """Butterworth high-pass filter of *x* at *cutoff_hz*."""
    sos = butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return sosfilt(sos, x)


def bandpass(
    x: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = 2,
) -> np.ndarray:
    """Butterworth band-pass filter of *x* between *low_hz* and *high_hz*."""
    sos = butter(
        order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos"
    )
    return sosfilt(sos, x)


def transient(
    length: int,
    fs: float,
    seed: int,
    low_hz: float = 500.0,
    high_hz: float = 12000.0,
    decay_s: float = 0.02,
) -> np.ndarray:
    """A short filtered noise burst shaped by an exponential decay."""
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    noise = seeded_noise(length, seed)
    filtered = bandpass(noise, low_hz, high_hz, fs)
    envelope = exponential_decay(length, decay_s, fs)
    return filtered * envelope


def contact_transient(
    length: int,
    fs: float,
    seed: int,
    *,
    low_hz: float,
    high_hz: float,
    decay_s: float,
    attack_s: float = 0.00035,
) -> np.ndarray:
    """A compact, finger-on-skin strike transient.

    Unlike :func:`transient`, this adds a tiny attack ramp and normalises the
    burst, so it reads as a controlled membrane hit instead of a bed of hiss
    between consecutive strokes.
    """
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    burst = transient(length, fs, seed, low_hz=low_hz, high_hz=high_hz, decay_s=decay_s)
    attack_n = min(max(1, int(round(attack_s * fs))), length)
    burst = burst.copy()
    burst[:attack_n] *= np.linspace(0.0, 1.0, attack_n)
    gate_n = min(length, max(attack_n + 2, int(round(decay_s * fs * 5.0))))
    if gate_n < length:
        release_n = min(length - gate_n, max(2, int(round(0.004 * fs))))
        burst[gate_n : gate_n + release_n] *= np.linspace(1.0, 0.0, release_n)
        burst[gate_n + release_n :] = 0.0
    peak = float(np.max(np.abs(burst)))
    if peak > 1e-12:
        burst /= peak
    return burst

"""Resonant modal synthesis.

A *mode* is a sinusoid with a per-sample frequency (so pitch bends are
supported) that decays exponentially. Tabla-like bodies are approximated as
sets of such modes — low, slowly decaying modes for the shell and air, plus
faster-decaying inharmonic partials for the membrane.
"""

from __future__ import annotations

import numpy as np

from src.synthesis.envelopes import exponential_decay
from src.synthesis.oscillator import integrate_phase, sine

_DECAY_FLOOR = 1e-3


def resonant_mode(
    frequency_hz: np.ndarray,
    amplitude: float,
    tau_s: float,
    length: int,
    fs: float = 44100.0,
    phase0: float = 0.0,
) -> np.ndarray:
    """Render one damped mode whose frequency follows *frequency_hz*.

    *tau_s* is the true exponential time constant; the amplitude falls to
    0.1% after ``~6.9 * tau_s`` seconds.
    """
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    tau_s = max(float(tau_s), 1e-9)
    decay_s = tau_s * -np.log(_DECAY_FLOOR)
    envelope = exponential_decay(length, decay_s, fs, floor=_DECAY_FLOOR)
    phase = integrate_phase(frequency_hz, phase0, fs)
    return amplitude * envelope * sine(phase)


def combine(*signals: np.ndarray, length: int) -> np.ndarray:
    """Sum *signals* into one array of exactly *length* samples.

    Shorter inputs are zero-padded; longer ones are truncated, so callers can
    mix contributions of different lengths (e.g. a delayed layer) safely.
    """
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(length, dtype=np.float64)
    for signal in signals:
        if signal is None:
            continue
        available = min(len(signal), length)
        out[:available] += np.asarray(signal, dtype=np.float64)[:available]
    return out

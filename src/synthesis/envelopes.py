"""Envelope generators: attack, exponential decay and pitch contours.

Envelopes are pure functions of length, rates and sample rate. They never
depend on randomness and remain numerically stable for arbitrary durations
and extremely short or long rates.
"""

from __future__ import annotations

import numpy as np


def exponential_decay(
    length: int,
    decay_s: float,
    fs: float = 44100.0,
    floor: float = 1e-3,
) -> np.ndarray:
    """Return an envelope falling from ``1.0`` to *floor* in *decay_s* seconds.

    The decay is ``exp(-t / tau)`` with ``tau = -decay_s / ln(floor)``, so the
    envelope reaches *floor* exactly when ``t == decay_s``.
    """
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    decay_s = max(float(decay_s), 1e-9)
    floor = min(max(float(floor), 1e-9), 1.0)
    tau = -decay_s / np.log(floor)
    time = np.arange(length, dtype=np.float64) / fs
    return np.exp(-time / tau)


def attack_decay(
    length: int,
    attack_s: float,
    decay_s: float,
    fs: float = 44100.0,
    peak: float = 1.0,
) -> np.ndarray:
    """Rise to *peak* over *attack_s* seconds, then decay over *decay_s*.

    The attack is a linear ramp; the decay follows :func:`exponential_decay`.
    Durations longer than the buffer are clamped so the envelope stays finite.
    """
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    attack_s = max(float(attack_s), 1e-9)
    decay_s = max(float(decay_s), 1e-9)
    envelope = np.empty(length, dtype=np.float64)
    attack_n = min(max(1, int(round(attack_s * fs))), length)
    envelope[:attack_n] = peak * np.linspace(0.0, 1.0, attack_n)
    if attack_n < length:
        tail = exponential_decay(
            length - attack_n, decay_s, fs, floor=1e-3
        )
        envelope[attack_n:] = peak * tail
    return envelope


def attack_ramp(
    length: int,
    attack_s: float,
    fs: float = 44100.0,
    peak: float = 1.0,
) -> np.ndarray:
    """Rise linearly to *peak* over *attack_s* seconds, then hold at *peak*.

    Unlike :func:`attack_decay`, this envelope never imposes its own decay.
    It exists for bodies that already decay on their own — e.g. a sum of
    :func:`resonators.resonant_mode` calls, each with its own per-mode time
    constant — where multiplying by a *second*, independently-decaying
    envelope would compound the two exponentials and cut the audible
    sustain far shorter than either was individually meant to. Use this to
    shape only the strike's attack and let the body supply the sustain.
    """
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    attack_s = max(float(attack_s), 1e-9)
    envelope = np.full(length, float(peak), dtype=np.float64)
    attack_n = min(max(1, int(round(attack_s * fs))), length)
    envelope[:attack_n] = peak * np.linspace(0.0, 1.0, attack_n)
    return envelope


def pitch_contour(
    start_hz: float,
    settle_hz: float,
    decay_s: float,
    length: int,
    fs: float = 44100.0,
) -> np.ndarray:
    """Return a per-sample frequency contour that glides and settles.

    The contour starts at *start_hz*, decays exponentially to *settle_hz* and
    is intended to be fed directly to :func:`oscillator.integrate_phase`.
    """
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    decay = exponential_decay(length, decay_s, fs)
    return settle_hz + (start_hz - settle_hz) * decay

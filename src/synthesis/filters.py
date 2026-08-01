"""Per-drum mixing EQ stage (biquad filters, RBJ audio-eq cookbook).

A real tabla recording is sculpted in the mix: the bayan keeps a clean,
weighty low register while the dayan is de-mudded and given presence up top.
Modal synthesis alone has no such separation, so each finished drum stroke
runs through its own small filter chain before the two are mixed together:

    Bayan:  high-pass ~55 Hz  (clear sub-bass rumble)
            peaking  +4 dB ~110 Hz  (body / weight)
            peaking  -3 dB ~320 Hz  (muddiness cut)

    Dayan:  high-pass ~150 Hz  (roll off lows, stay clear of the bayan)
            peaking  +3 dB ~1.8 kHz  (bright "na"/"tin" attack partials)
            high-shelf +0.5 dB ~7 kHz  (subtle air; no hiss boost)

Filters are the standard RBJ biquads (high-pass, peaking EQ, shelf), each
computed per the cookbook and cascaded as second-order sections via
``scipy.signal.sosfilt``. The chain is deterministic and linear, so it never
breaks the engine's determinism guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.signal import sosfilt


class EqFilterKind(Enum):
    """The biquad topologies the EQ stage supports (RBJ cookbook)."""

    HIGH_PASS = "highpass"
    PEAKING = "peaking"
    HIGH_SHELF = "highshelf"
    LOW_SHELF = "lowshelf"


@dataclass(frozen=True)
class EqBand:
    """One parametric EQ band.

    Attributes:
        kind: biquad topology.
        freq_hz: centre frequency.
        gain_db: boost/cut in decibels (unused for high-pass).
        q: resonance; higher = narrower (unused for shelves).
    """

    kind: EqFilterKind
    freq_hz: float
    gain_db: float = 0.0
    q: float = 0.707


#: Bayan (left drum) mix chain: clean, weighty lows, no boxy mids.
BAYAN_EQ: tuple[EqBand, ...] = (
    EqBand(EqFilterKind.HIGH_PASS, 55.0),
    EqBand(EqFilterKind.PEAKING, 110.0, gain_db=4.0, q=0.9),
    EqBand(EqFilterKind.PEAKING, 320.0, gain_db=-3.0, q=1.0),
)

#: Dayan (right drum) mix chain: de-mudded, present, airy top end.
DAYAN_EQ: tuple[EqBand, ...] = (
    EqBand(EqFilterKind.HIGH_PASS, 150.0),
    EqBand(EqFilterKind.PEAKING, 1800.0, gain_db=3.0, q=1.0),
    EqBand(EqFilterKind.HIGH_SHELF, 7000.0, gain_db=0.5),
)


def biquad_coeffs(band: EqBand, fs: float) -> np.ndarray:
    """Return the ``(b0, b1, b2, a0, a1, a2)`` biquad for *band*, a0-normalised."""
    if band.freq_hz <= 0.0 or fs <= 0.0:
        raise ValueError(
            f"freq_hz and fs must be positive, got {band.freq_hz} / {fs}"
        )
    if band.freq_hz >= fs / 2.0:
        raise ValueError(
            f"freq_hz {band.freq_hz} must be below the Nyquist rate {fs / 2.0}"
        )

    gain = float(band.gain_db)
    a_gain = 10.0 ** (gain / 40.0)  # linear amplitude gain
    w0 = 2.0 * np.pi * float(band.freq_hz) / float(fs)
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    q = max(float(band.q), 1e-3)
    kind = band.kind

    if kind is EqFilterKind.HIGH_PASS:
        alpha = sin_w0 / (2.0 * q)
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = b0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif kind is EqFilterKind.PEAKING:
        alpha = sin_w0 / (2.0 * q)
        b0 = 1.0 + alpha * a_gain
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * a_gain
        a0 = 1.0 + alpha / a_gain
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / a_gain
    elif kind in (EqFilterKind.HIGH_SHELF, EqFilterKind.LOW_SHELF):
        # Constant-Q shelf (RBJ slope S = 1).
        alpha = sin_w0 / 2.0 * np.sqrt(2.0)
        if kind is EqFilterKind.LOW_SHELF:
            b0 = a_gain * (
                (a_gain + 1.0)
                - (a_gain - 1.0) * cos_w0
                + 2.0 * np.sqrt(a_gain) * alpha
            )
            b1 = 2.0 * a_gain * ((a_gain - 1.0) - (a_gain + 1.0) * cos_w0)
            b2 = a_gain * (
                (a_gain + 1.0)
                - (a_gain - 1.0) * cos_w0
                - 2.0 * np.sqrt(a_gain) * alpha
            )
            a0 = (a_gain + 1.0) + (a_gain - 1.0) * cos_w0 + 2.0 * np.sqrt(a_gain) * alpha
            a1 = -2.0 * ((a_gain - 1.0) + (a_gain + 1.0) * cos_w0)
            a2 = (a_gain + 1.0) + (a_gain - 1.0) * cos_w0 - 2.0 * np.sqrt(a_gain) * alpha
        else:
            b0 = a_gain * (
                (a_gain + 1.0)
                + (a_gain - 1.0) * cos_w0
                + 2.0 * np.sqrt(a_gain) * alpha
            )
            b1 = -2.0 * a_gain * ((a_gain - 1.0) + (a_gain + 1.0) * cos_w0)
            b2 = a_gain * (
                (a_gain + 1.0)
                + (a_gain - 1.0) * cos_w0
                - 2.0 * np.sqrt(a_gain) * alpha
            )
            a0 = (a_gain + 1.0) - (a_gain - 1.0) * cos_w0 + 2.0 * np.sqrt(a_gain) * alpha
            a1 = 2.0 * ((a_gain - 1.0) - (a_gain + 1.0) * cos_w0)
            a2 = (a_gain + 1.0) - (a_gain - 1.0) * cos_w0 - 2.0 * np.sqrt(a_gain) * alpha
    else:  # pragma: no cover - guarded by the enum
        raise ValueError(f"Unknown EQ filter kind: {band.kind!r}")

    return np.array(
        [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0],
        dtype=np.float64,
    )


def apply_eq(
    x: np.ndarray,
    bands: tuple[EqBand, ...] | list[EqBand],
    fs: float,
) -> np.ndarray:
    """Cascade *bands* over *x* in order and return the equalised signal.

    The result keeps the input's length and is deterministic for a given
    input and filter set. An empty *x* or empty *bands* returns *x* unchanged.
    """
    signal = np.asarray(x, dtype=np.float64)
    if signal.size == 0 or not bands:
        return signal
    sections = np.atleast_2d(
        np.array([biquad_coeffs(band, fs) for band in bands], dtype=np.float64)
    )
    return sosfilt(sections, signal)

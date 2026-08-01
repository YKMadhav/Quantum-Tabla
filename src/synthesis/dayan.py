"""Dayan (right drum) synthesis.

The dayan is a high-pitched modal drum whose tonal centre is the syahi — the
black loaded patch on the membrane. Instead of a full physics simulation, the
syahi is approximated by modal frequency *ratios*, per-mode amplitude weights
and per-mode decay rates that reproduce its effect on the tone:

* the first mode dominates and defines the nominal pitch,
* higher modes are slightly inharmonic and decay progressively faster,
* striking near the centre excites the low modes; striking near the edge
  excites the bright, high modes.

This approximation is documented here and in the README rather than pretending
to solve the membrane wave equation.
"""

from __future__ import annotations

import numpy as np

from src.synthesis.envelopes import attack_ramp, pitch_contour
from src.synthesis.mapping import SynthesisParameters
from src.synthesis.noise import contact_transient
from src.synthesis.resonators import combine, resonant_mode

#: Syahi-approximated modal ratios of the dayan (fundamental first).
DAYAN_MODAL_RATIOS = (1.00, 1.52, 2.05, 2.82, 3.55, 4.35, 5.20)
#: Mode weights when striking dead centre (low modes dominate).
_DAYAN_CENTRE_WEIGHTS = (1.00, 0.44, 0.32, 0.15, 0.08, 0.04, 0.02)
#: Mode weights when striking the rim (bright modes dominate).
_DAYAN_EDGE_WEIGHTS = (0.50, 0.42, 0.58, 0.62, 0.48, 0.30, 0.18)
#: Overall level so the summed modes stay well inside the soft limiter.
_BODY_GAIN = 0.76


def _mode_weights(params: SynthesisParameters) -> np.ndarray:
    """Modal amplitude weights for the current strike position and brightness."""
    centre = np.asarray(_DAYAN_CENTRE_WEIGHTS, dtype=np.float64)
    edge = np.asarray(_DAYAN_EDGE_WEIGHTS, dtype=np.float64)
    pos = float(params.dayan_strike_position)
    weights = (1.0 - pos) * centre + pos * edge
    ratios = np.asarray(DAYAN_MODAL_RATIOS, dtype=np.float64)
    weights *= 1.0 + params.dayan_brightness * (ratios - 1.0) * 0.18
    return weights


def synthesize_dayan(
    params: SynthesisParameters,
    length: int,
    fs: float,
    seed: int,
    muted: bool = False,
) -> np.ndarray:
    """Render a dayan stroke of *length* samples at *fs* Hz.

    *muted* produces the short, bright, sharp-edged stroke used for closed
    strokes: no pitch glide, faster decay and a brighter transient.
    """
    if length <= 0:
        return np.zeros(0, dtype=np.float64)

    decay_s = params.dayan_decay_s * (0.18 if muted else 0.88)
    attack_s = params.dayan_attack_s * (0.35 if muted else 0.75)

    if muted:
        settle = params.dayan_fundamental_hz * 1.04
        contour = np.full(length, settle, dtype=np.float64)
    else:
        settle = params.dayan_fundamental_hz * params.dayan_pitch_settle_ratio
        contour = pitch_contour(
            params.dayan_fundamental_hz * 1.05,
            settle,
            params.dayan_decay_s * 0.5,
            length,
            fs,
        )

    per_mode = _mode_weights(params)
    body = np.zeros(length, dtype=np.float64)
    for ratio, weight in zip(DAYAN_MODAL_RATIOS, per_mode):
        tau = decay_s / (1.0 + 0.72 * (ratio - 1.0))
        body += resonant_mode(contour * ratio, weight, tau, length, fs)

    if muted:
        body += resonant_mode(
            contour * 3.25,
            0.25 + params.dayan_brightness * 0.18,
            0.012,
            length,
            fs,
            phase0=np.pi * 0.5,
        )

    # Shape only the attack here — each resonant_mode() above already decays
    # on its own per-mode time constant, so that (and not a second envelope)
    # is what should determine how long the stroke actually rings.
    attack = attack_ramp(length, attack_s, fs)
    low_hz = 700.0 if muted else 520.0
    high_hz = 6200.0 if muted else 5200.0
    click = contact_transient(
        length,
        fs,
        seed,
        low_hz=low_hz,
        high_hz=high_hz,
        decay_s=0.0045 if muted else 0.0055,
        attack_s=0.00018,
    )
    click_gain = params.dayan_noise_gain * (0.24 if muted else 0.18)

    return combine(body * attack, click * click_gain, length=length) * _BODY_GAIN

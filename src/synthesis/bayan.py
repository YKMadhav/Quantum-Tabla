"""Bayan (left drum) synthesis.

The bayan is modelled as a low, largely inharmonic body: a few partials at
non-harmonic ratios of the fundamental, each decaying at its own rate, over a
pitch contour that bends sharply at the strike and settles. A compact contact
transient, a shell resonance and a low air resonance complete the timbre.

No audio samples are involved; every sample is computed from the modal
parameters in :class:`SynthesisParameters`.
"""

from __future__ import annotations

import numpy as np

from src.synthesis.envelopes import attack_ramp, pitch_contour
from src.synthesis.mapping import SynthesisParameters
from src.synthesis.noise import contact_transient
from src.synthesis.resonators import combine, resonant_mode

#: Inharmonic ratios of the bayan partials (fundamental first).
BAYAN_MODE_RATIOS = (1.00, 1.48, 1.94, 2.86, 3.76, 4.62)
#: Relative amplitude weight of each partial.
BAYAN_MODE_WEIGHTS = (1.00, 0.42, 0.36, 0.24, 0.12, 0.06)
#: Overall level so the summed modes stay well inside the soft limiter.
_BODY_GAIN = 0.72


def synthesize_bayan(
    params: SynthesisParameters,
    length: int,
    fs: float,
    seed: int,
) -> np.ndarray:
    """Render a bayan stroke of *length* samples at *fs* Hz."""
    if length <= 0:
        return np.zeros(0, dtype=np.float64)

    settle = params.bayan_fundamental_hz * params.bayan_pitch_settle_ratio
    strike = params.bayan_fundamental_hz * params.bayan_strike_pitch_ratio
    contour = pitch_contour(
        strike, settle, params.bayan_pitch_env_decay_s, length, fs
    )

    body = np.zeros(length, dtype=np.float64)
    for ratio, weight in zip(BAYAN_MODE_RATIOS, BAYAN_MODE_WEIGHTS):
        tau = params.bayan_decay_s / (1.0 + 0.62 * (ratio - 1.0))
        body += resonant_mode(
            contour * ratio,
            weight * params.bayan_overtone_weight,
            tau,
            length,
            fs,
        )

    body += resonant_mode(
        contour * 1.52,
        params.bayan_shell_gain * 0.62,
        params.bayan_decay_s * 0.85 + params.bayan_resonance_tail_s,
        length,
        fs,
    )
    body += resonant_mode(
        contour,
        params.bayan_air_gain * 0.9,
        params.bayan_decay_s * 1.25 + params.bayan_resonance_tail_s,
        length,
        fs,
    )
    body += resonant_mode(
        contour * 2.18,
        0.18 + params.bayan_noise_gain * 0.55,
        0.030,
        length,
        fs,
        phase0=np.pi * 0.5,
    )

    # Shape only the attack here — each resonant_mode() above already decays
    # on its own per-mode time constant, so that (and not a second envelope)
    # is what should determine how long the stroke actually rings.
    attack = attack_ramp(length, params.bayan_attack_s, fs)
    click = contact_transient(
        length,
        fs,
        seed,
        low_hz=90.0,
        high_hz=1900.0,
        decay_s=0.0065,
        attack_s=0.00025,
    )

    return (
        combine(body * attack, click * params.bayan_noise_gain * 0.20, length=length)
        * _BODY_GAIN
    )

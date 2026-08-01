"""Centralised mapping from the normalised instrument state to synthesis values.

This is the single place where the 16 named parameters produced by the
randomness engine become audible, physical quantities (frequencies, decay
times, gains, timing). The DSP drum builders consume :class:`SynthesisParameters`
and know nothing about randomness or quantum sources.

Every parameter in ``DEFAULT_CATALOG`` is consumed here exactly once; the
mapping is deterministic and purely a function of the instrument state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.instrument.state import InstrumentState
from src.core.randomness.mapper import DEFAULT_CATALOG


@dataclass(frozen=True)
class SynthesisParameters:
    """Physical, acoustic quantities the drum synthesizers actually use."""

    # -- Bayan (left drum) -----------------------------------------------------
    bayan_fundamental_hz: float
    bayan_strike_pitch_ratio: float
    bayan_pitch_settle_ratio: float
    bayan_pitch_env_decay_s: float
    bayan_attack_s: float
    bayan_decay_s: float
    bayan_shell_gain: float
    bayan_air_gain: float
    bayan_resonance_tail_s: float
    bayan_noise_gain: float
    bayan_overtone_weight: float

    # -- Dayan (right drum) ----------------------------------------------------
    dayan_fundamental_hz: float
    dayan_strike_position: float
    dayan_brightness: float
    dayan_attack_s: float
    dayan_decay_s: float
    dayan_noise_gain: float
    dayan_pitch_settle_ratio: float

    # -- Mixing ----------------------------------------------------------------
    stroke_gain: float
    accent_gain: float
    combined_delay_s: float


#: Neutral physical defaults used when every parameter sits at 0.5.
_NEUTRAL = SynthesisParameters(
    bayan_fundamental_hz=100.0,
    bayan_strike_pitch_ratio=1.20,
    bayan_pitch_settle_ratio=1.0,
    bayan_pitch_env_decay_s=0.10,
    bayan_attack_s=0.002,
    bayan_decay_s=0.14,
    bayan_shell_gain=0.35,
    bayan_air_gain=0.30,
    bayan_resonance_tail_s=0.06,
    bayan_noise_gain=0.16,
    bayan_overtone_weight=0.75,
    dayan_fundamental_hz=220.0,
    dayan_strike_position=0.5,
    dayan_brightness=0.5,
    dayan_attack_s=0.0015,
    dayan_decay_s=0.16,
    dayan_noise_gain=0.03,
    dayan_pitch_settle_ratio=1.0,
    stroke_gain=0.75,
    accent_gain=1.0,
    combined_delay_s=0.045,
)

#: Human-readable description of the acoustic role of every parameter.
PARAMETER_EFFECTS: dict[str, str] = {
    "Membrane tension": "Raises the fundamental pitch of both drums and sharpens the strike pitch bend.",
    "Membrane diameter": "Lowers the fundamental pitch of both drums (bigger head = deeper tone).",
    "Shell resonance": "Adds low shell modes and lengthens the resonance tail.",
    "Air resonance": "Reinforces the low air/fundamental ring of the bayan.",
    "Brightness": "Boosts the high-frequency modal energy of the dayan and the bayan's overtones.",
    "Damping": "Shortens every decay time (felt skin dampens vibration).",
    "Noise level": "Sets the amplitude of the broadband strike transient.",
    "Resonance": "Scales the overall sustain of both drums.",
    "Pitch bend": "Controls how far each drum's pitch glides downward after the strike.",
    "Strike position": "Shifts the dayan's modal balance from centre (low modes) to edge (bright modes).",
    "Strike velocity": "Scales the overall strike gain of both drums.",
    "Finger pressure": "Raises the dayan's pitch and brightens it; shortens the bayan's ring.",
    "Attack time": "Scales the attack ramp duration of both drums.",
    "Decay time": "Scales the primary decay duration of both drums.",
    "Accent": "Boosts the final gain of the stroke.",
    "Tempo variation": "Offsets the dayan relative to the bayan in the combined stroke (micro-timing).",
}


def default_synthesis_parameters() -> SynthesisParameters:
    """Neutral synthesis parameters (every instrument value at 0.5)."""
    return _NEUTRAL


def map_parameters(state: InstrumentState) -> SynthesisParameters:
    """Map the current instrument state onto physical synthesis values."""
    v = state.parameters

    tension = v.get("Membrane tension", 0.5)
    diameter = v.get("Membrane diameter", 0.5)
    shell = v.get("Shell resonance", 0.5)
    air = v.get("Air resonance", 0.5)
    brightness = v.get("Brightness", 0.5)
    damping = v.get("Damping", 0.5)
    noise_level = v.get("Noise level", 0.5)
    resonance = v.get("Resonance", 0.5)
    pitch_bend = v.get("Pitch bend", 0.5)
    strike_position = v.get("Strike position", 0.5)
    velocity = v.get("Strike velocity", 0.5)
    pressure = v.get("Finger pressure", 0.5)
    attack = v.get("Attack time", 0.5)
    decay = v.get("Decay time", 0.5)
    accent = v.get("Accent", 0.5)
    tempo = v.get("Tempo variation", 0.5)

    tension_scale = 1.0 + 0.30 * (tension - 0.5)
    diameter_scale = 1.0 / (1.0 + 0.50 * (diameter - 0.5))
    damping_scale = 1.0 / (1.0 + 0.80 * (damping - 0.5))
    sustain_scale = 1.0 + 1.20 * (resonance - 0.5)
    attack_scale = 1.0 + 1.50 * (attack - 0.5)
    decay_scale = 1.0 + 1.50 * (decay - 0.5)
    pressure_scale = 1.0 + 0.15 * (pressure - 0.5)

    return SynthesisParameters(
        bayan_fundamental_hz=_NEUTRAL.bayan_fundamental_hz
        * tension_scale
        * diameter_scale,
        bayan_strike_pitch_ratio=1.20 + 0.10 * (tension - 0.5),
        bayan_pitch_settle_ratio=1.0 - 0.10 * (pitch_bend - 0.5),
        bayan_pitch_env_decay_s=_NEUTRAL.bayan_pitch_env_decay_s * damping_scale,
        bayan_attack_s=_NEUTRAL.bayan_attack_s * attack_scale,
        bayan_decay_s=_NEUTRAL.bayan_decay_s * decay_scale * damping_scale * sustain_scale,
        bayan_shell_gain=0.10 + 0.50 * shell,
        bayan_air_gain=0.06 + 0.48 * air,
        bayan_resonance_tail_s=0.03 + 0.06 * shell * sustain_scale,
        # Floor + a gentler ceiling so the strike click sits well *under* the
        # drum's resonant tone rather than competing with or masking it —
        # the click should read as a touch of attack, never as hiss between
        # consecutive strokes. Floors chosen so the neutral mapping
        # reproduces ``_NEUTRAL`` exactly.
        bayan_noise_gain=0.04 + 0.24 * noise_level,
        bayan_overtone_weight=0.5 + 0.5 * brightness,
        dayan_fundamental_hz=_NEUTRAL.dayan_fundamental_hz
        * tension_scale
        * diameter_scale
        * pressure_scale,
        dayan_strike_position=strike_position,
        dayan_brightness=float(
            np.clip(brightness + 0.30 * (pressure - 0.5), 0.0, 1.0)
        ),
        dayan_attack_s=_NEUTRAL.dayan_attack_s * attack_scale,
        dayan_decay_s=_NEUTRAL.dayan_decay_s * decay_scale * damping_scale * sustain_scale,
        dayan_noise_gain=0.01 + 0.04 * noise_level,
        dayan_pitch_settle_ratio=1.0 - 0.04 * (pitch_bend - 0.5),
        stroke_gain=0.50 + 0.50 * velocity,
        accent_gain=0.70 + 0.60 * accent,
        combined_delay_s=0.02 + 0.05 * tempo,
    )


def catalog_parameter_names() -> tuple[str, ...]:
    """The 16 canonical parameter names the mapping consumes."""
    return tuple(spec.name for spec in DEFAULT_CATALOG.specs)

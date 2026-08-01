"""Unit tests for oscillator primitives and phase integration."""

from __future__ import annotations

import numpy as np
import pytest

from src.synthesis.oscillator import glide, integrate_phase, sine, tone

FS = 44100.0


def test_tone_length_and_range() -> None:
    x = tone(100.0, 0.25, fs=FS)
    assert x.shape == (11025,)
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() <= 1.0 + 1e-12


def test_phase_integration_is_exact_for_constant_frequency() -> None:
    frequency = np.full(1000, 100.0)
    phase = integrate_phase(frequency, phase0=0.0, fs=FS)
    expected = 2.0 * np.pi * 100.0 * np.arange(1000) / FS
    assert np.allclose(phase, expected, atol=1e-9)


def test_phase_increments_match_instantaneous_frequency() -> None:
    frequency = np.linspace(100.0, 800.0, 4000)
    phase = integrate_phase(frequency, phase0=0.0, fs=FS)
    increments = np.diff(phase)
    expected = 2.0 * np.pi * frequency[1:] / FS
    assert np.allclose(increments, expected, atol=1e-12)


def test_phase_is_continuous_across_frequency_jump() -> None:
    frequency = np.concatenate([np.full(2000, 100.0), np.full(2000, 400.0)])
    phase = integrate_phase(frequency, phase0=0.0, fs=FS)
    signal = sine(phase)
    jumps = np.abs(np.diff(signal))
    assert jumps.max() < 0.5


def test_glide_produces_expected_zero_crossings() -> None:
    # A 200 ms constant 200 Hz glide is a 200 Hz tone -> ~80 zero crossings.
    x = glide(200.0, 200.0, 0.2, fs=FS)
    crossings = int(np.sum(np.diff(np.sign(x)) != 0))
    assert 70 <= crossings <= 90


def test_empty_frequency_raises() -> None:
    with pytest.raises(ValueError):
        integrate_phase(np.array([]), fs=FS)

"""Unit tests for envelope generators."""

from __future__ import annotations

import numpy as np
import pytest

from src.synthesis.envelopes import attack_decay, exponential_decay, pitch_contour

FS = 44100.0


def test_exponential_decay_length_and_bounds() -> None:
    env = exponential_decay(22050, 0.5, fs=FS)
    assert len(env) == 22050
    assert np.all(env >= 0.0)
    assert np.all(env <= 1.0 + 1e-12)
    assert env[0] == pytest.approx(1.0)
    assert env[-1] < 0.05


def test_exponential_decay_reaches_floor() -> None:
    floor = 1e-3
    env = exponential_decay(44100, 0.2, fs=FS, floor=floor)
    reached = int(round(0.2 * FS))
    assert env[reached] == pytest.approx(floor, abs=1e-3)


def test_exponential_decay_is_monotonic() -> None:
    env = exponential_decay(88200, 0.4, fs=FS)
    assert np.all(np.diff(env) <= 1e-15)


def test_exponential_decay_empty_and_short() -> None:
    assert exponential_decay(0, 0.5, fs=FS).size == 0
    assert exponential_decay(1, 0.5, fs=FS).size == 1


def test_attack_decay_shape_and_peak() -> None:
    env = attack_decay(44100, 0.01, 0.4, fs=FS, peak=0.8)
    assert len(env) == 44100
    assert np.all(env >= 0.0)
    assert np.all(env <= 0.8 + 1e-12)
    assert env[0] == pytest.approx(0.0)
    assert env[-1] < 0.05
    assert env.max() == pytest.approx(0.8, abs=1e-9)


def test_attack_decay_empty() -> None:
    assert attack_decay(0, 0.01, 0.4, fs=FS).size == 0


def test_attack_decay_shorter_than_attack() -> None:
    env = attack_decay(50, 0.1, 0.4, fs=FS)
    assert len(env) == 50
    assert np.all(env >= 0.0)
    assert np.all(env <= 1.0 + 1e-12)


def test_pitch_contour_starts_and_settles() -> None:
    contour = pitch_contour(110.0, 100.0, 0.15, 4410, fs=FS)
    assert contour[0] == pytest.approx(110.0)
    assert abs(contour[-1] - 100.0) < 0.5
    assert np.all(np.diff(contour) <= 0.0)

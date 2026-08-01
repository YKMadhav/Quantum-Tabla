"""Unit tests for deterministic noise generation and filtering."""

from __future__ import annotations

import numpy as np

from src.synthesis.noise import (
    bandpass,
    highpass,
    lowpass,
    seeded_noise,
    transient,
)

FS = 44100.0


def test_seeded_noise_is_deterministic() -> None:
    a = seeded_noise(4096, seed=42)
    b = seeded_noise(4096, seed=42)
    c = seeded_noise(4096, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_seeded_noise_bounds_and_length() -> None:
    x = seeded_noise(1024, seed=1)
    assert x.shape == (1024,)
    assert np.all(np.isfinite(x))
    assert np.all(x >= -1.0)
    assert np.all(x <= 1.0)


def test_filters_preserve_length_and_finiteness() -> None:
    x = seeded_noise(8192, seed=5)
    for y in (
        lowpass(x, 1000.0, FS),
        highpass(x, 200.0, FS),
        bandpass(x, 300.0, 5000.0, FS),
    ):
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))


def test_highpass_removes_dc() -> None:
    x = np.ones(4096)
    y = highpass(x, 40.0, FS)
    assert abs(y.mean()) < 1e-3


def test_transient_shape_and_finiteness() -> None:
    burst = transient(4096, FS, seed=7)
    assert burst.shape == (4096,)
    assert np.all(np.isfinite(burst))
    assert np.max(np.abs(burst)) < 1.0


def test_transient_is_deterministic() -> None:
    assert np.array_equal(transient(2048, FS, seed=9), transient(2048, FS, seed=9))
    assert not np.array_equal(transient(2048, FS, seed=9), transient(2048, FS, seed=10))


def test_empty_length_is_safe() -> None:
    assert seeded_noise(0, seed=1).size == 0
    assert transient(0, FS, seed=1).size == 0

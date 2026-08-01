"""Unit tests for the per-drum mixing EQ stage (RBJ biquads)."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.instrument.state import InstrumentState
from src.core.randomness.mapper import DEFAULT_CATALOG, ParameterVector
from src.synthesis.filters import (
    BAYAN_EQ,
    DAYAN_EQ,
    EqBand,
    EqFilterKind,
    apply_eq,
    biquad_coeffs,
)
from src.synthesis.tabla import StrokeType, render_stroke

FS = 44100.0


def _sine(freq_hz: float, seconds: float = 1.0) -> np.ndarray:
    samples = int(round(seconds * FS))
    time = np.arange(samples) / FS
    return np.sin(2.0 * np.pi * freq_hz * time)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _state() -> InstrumentState:
    params = {spec.name: 0.5 for spec in DEFAULT_CATALOG.specs}
    return InstrumentState(ParameterVector(params), version=1)


def test_biquad_coeffs_are_normalised_and_finite() -> None:
    for kind in EqFilterKind:
        band = EqBand(kind, 300.0, gain_db=3.0 if kind != EqFilterKind.HIGH_PASS else 0.0)
        b0, b1, b2, a0, a1, a2 = biquad_coeffs(band, FS)
        assert a0 == pytest.approx(1.0)
        assert all(np.isfinite([b0, b1, b2, a1, a2]))


@pytest.mark.parametrize("kind", list(EqFilterKind))
def test_biquad_rejects_invalid_frequencies(kind: EqFilterKind) -> None:
    with pytest.raises(ValueError):
        biquad_coeffs(EqBand(kind, -1.0), FS)
    with pytest.raises(ValueError):
        biquad_coeffs(EqBand(kind, 30000.0), FS)  # above Nyquist


def test_high_pass_rejects_dc() -> None:
    hpf = apply_eq(
        np.ones(int(FS), dtype=np.float64),
        (EqBand(EqFilterKind.HIGH_PASS, 200.0),),
        FS,
    )
    # The steady-state response to a constant must be ~0.
    assert np.max(np.abs(hpf[int(0.5 * FS):])) < 1e-6


def test_high_pass_attenuates_below_cutoff() -> None:
    band = (EqBand(EqFilterKind.HIGH_PASS, 1000.0),)
    low = apply_eq(_sine(120.0), band, FS)
    high = apply_eq(_sine(4000.0), band, FS)
    assert _rms(low) < 0.1 * _rms(high)


def test_peaking_boost_concentrates_energy_at_centre() -> None:
    band = (EqBand(EqFilterKind.PEAKING, 2000.0, gain_db=12.0, q=1.0),)
    at_centre = apply_eq(_sine(2000.0), band, FS)
    away = apply_eq(_sine(200.0), band, FS)
    assert _rms(at_centre) > 3.0 * _rms(away)
    assert _rms(at_centre) > _rms(_sine(2000.0))


def test_peaking_cut_attenuates_centre() -> None:
    band = (EqBand(EqFilterKind.PEAKING, 320.0, gain_db=-12.0, q=1.0),)
    at_centre = apply_eq(_sine(320.0), band, FS)
    away = apply_eq(_sine(3200.0), band, FS)
    assert _rms(at_centre) < 0.5 * _rms(away)


def test_high_shelf_boosts_above_cutoff() -> None:
    band = (EqBand(EqFilterKind.HIGH_SHELF, 4000.0, gain_db=10.0),)
    above = apply_eq(_sine(9000.0), band, FS)
    below = apply_eq(_sine(300.0), band, FS)
    assert _rms(above) > 2.0 * _rms(below)
    assert _rms(above) > _rms(_sine(9000.0))


def test_low_shelf_boosts_below_cutoff() -> None:
    band = (EqBand(EqFilterKind.LOW_SHELF, 500.0, gain_db=10.0),)
    below = apply_eq(_sine(80.0), band, FS)
    above = apply_eq(_sine(5000.0), band, FS)
    assert _rms(below) > 2.0 * _rms(above)


def test_apply_eq_preserves_length_and_boundedness() -> None:
    x = np.random.default_rng(0).uniform(-1.0, 1.0, 4096)
    for bands in (BAYAN_EQ, DAYAN_EQ):
        y = apply_eq(x, bands, FS)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))


def test_apply_eq_is_deterministic() -> None:
    x = np.random.default_rng(1).uniform(-1.0, 1.0, 2048)
    first = apply_eq(x, BAYAN_EQ, FS)
    second = apply_eq(x, BAYAN_EQ, FS)
    np.testing.assert_array_equal(first, second)


def test_apply_eq_identity_for_empty_bands_or_signal() -> None:
    x = np.random.default_rng(2).uniform(-1.0, 1.0, 1024)
    np.testing.assert_array_equal(apply_eq(x, (), FS), x)
    assert apply_eq(np.empty(0, dtype=np.float64), BAYAN_EQ, FS).size == 0


def test_preset_bands_sit_below_nyquist() -> None:
    for band in (*BAYAN_EQ, *DAYAN_EQ):
        assert 0.0 < band.freq_hz < FS / 2.0


def test_presets_match_expected_chain() -> None:
    assert [b.kind for b in BAYAN_EQ] == [
        EqFilterKind.HIGH_PASS,
        EqFilterKind.PEAKING,
        EqFilterKind.PEAKING,
    ]
    assert [b.kind for b in DAYAN_EQ] == [
        EqFilterKind.HIGH_PASS,
        EqFilterKind.PEAKING,
        EqFilterKind.HIGH_SHELF,
    ]


def test_render_with_eq_keeps_output_bounded() -> None:
    state = _state()
    for stroke in StrokeType:
        out = render_stroke(state, stroke, fs=FS)
        assert np.max(np.abs(out)) <= 1.0


def test_eq_bayan_stays_in_low_register() -> None:
    from src.synthesis.tabla import analyze_waveform

    state = _state()
    for seed in range(4):
        out = render_stroke(state, StrokeType.BAYAN_OPEN, fs=FS, seed=seed)
        dominant = analyze_waveform(out, FS)["dominant_hz"]
        assert 60.0 <= dominant <= 180.0


def test_eq_dayan_open_stays_mid_register() -> None:
    from src.synthesis.tabla import analyze_waveform

    state = _state()
    for seed in range(4):
        out = render_stroke(state, StrokeType.DAYAN_OPEN, fs=FS, seed=seed)
        dominant = analyze_waveform(out, FS)["dominant_hz"]
        assert 180.0 <= dominant <= 400.0

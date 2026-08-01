"""End-to-end DSP tests: stroke rendering, mixing, determinism and timbre."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.instrument.state import InstrumentState
from src.core.randomness.mapper import DEFAULT_CATALOG, ParameterVector
from src.synthesis.tabla import (
    StrokeType,
    analyze_waveform,
    render_stroke,
    soft_limit,
    waveform_envelope,
)

FS = 44100.0
DURATION = 0.8
ALL_NAMES = tuple(spec.name for spec in DEFAULT_CATALOG.specs)


def _state(values: dict[str, float] | None = None) -> InstrumentState:
    params = {name: 0.5 for name in ALL_NAMES}
    if values:
        params.update(values)
    return InstrumentState(ParameterVector(params), version=1)


def _stroke(
    stroke: StrokeType,
    seed: int = 7,
    values: dict[str, float] | None = None,
) -> np.ndarray:
    return render_stroke(
        _state(values),
        stroke,
        fs=FS,
        duration_s=DURATION,
        seed=seed,
    )


@pytest.mark.parametrize("stroke", list(StrokeType))
def test_all_strokes_are_finite_bounded_and_correct_length(stroke: StrokeType) -> None:
    x = _stroke(stroke)
    assert x.shape == (int(DURATION * FS),)
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() <= 1.0


def test_render_is_deterministic_for_same_seed() -> None:
    a = _stroke(StrokeType.COMBINED, seed=9)
    b = _stroke(StrokeType.COMBINED, seed=9)
    assert np.array_equal(a, b)


def test_render_changes_with_seed() -> None:
    a = _stroke(StrokeType.BAYAN_OPEN, seed=9)
    b = _stroke(StrokeType.BAYAN_OPEN, seed=10)
    assert not np.array_equal(a, b)


def test_bayan_rings_in_low_register() -> None:
    for seed in (1, 2, 3, 4):
        x = _stroke(StrokeType.BAYAN_OPEN, seed=seed)
        dominant = analyze_waveform(x, FS)["dominant_hz"]
        assert 60.0 <= dominant <= 180.0, f"seed {seed}: {dominant} Hz"


def test_dayan_rings_at_pitched_fundamental() -> None:
    for seed in (1, 2, 3):
        x = _stroke(StrokeType.DAYAN_OPEN, seed=seed)
        dominant = analyze_waveform(x, FS)["dominant_hz"]
        assert 180.0 <= dominant <= 360.0, f"seed {seed}: {dominant} Hz"


def _rms(x: np.ndarray, start: float, end: float) -> float:
    a = int(start * len(x))
    b = int(end * len(x))
    segment = x[a:b]
    return float(np.sqrt(np.mean(segment**2))) if len(segment) else 0.0


def test_muted_dayan_rings_shorter_than_open() -> None:
    open_rms = _rms(_stroke(StrokeType.DAYAN_OPEN), 0.75, 1.0)
    muted_rms = _rms(_stroke(StrokeType.DAYAN_MUTED), 0.75, 1.0)
    assert muted_rms < open_rms * 0.5


def test_combined_contains_both_registers() -> None:
    x = _stroke(StrokeType.COMBINED)
    n = len(x)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(n)))
    low = float(np.sum(spectrum[: int(150.0 / FS * n)]))
    high = float(np.sum(spectrum[int(400.0 / FS * n) : int(3000.0 / FS * n)]))
    assert low > 0.0
    assert high > 0.0


def test_strike_velocity_raises_amplitude() -> None:
    soft = _stroke(StrokeType.BAYAN_OPEN, values={"Strike velocity": 0.0})
    hard = _stroke(StrokeType.BAYAN_OPEN, values={"Strike velocity": 1.0})
    assert _rms(hard, 0.0, 1.0) > _rms(soft, 0.0, 1.0) * 1.2


def test_accent_raises_amplitude() -> None:
    plain = _stroke(StrokeType.DAYAN_OPEN, values={"Accent": 0.0})
    accented = _stroke(StrokeType.DAYAN_OPEN, values={"Accent": 1.0})
    assert _rms(accented, 0.0, 1.0) > _rms(plain, 0.0, 1.0) * 1.15


def test_strike_position_changes_dayan_timbre() -> None:
    centre = _stroke(StrokeType.DAYAN_OPEN, values={"Strike position": 0.0})
    edge = _stroke(StrokeType.DAYAN_OPEN, values={"Strike position": 1.0})
    n = len(centre)
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    mask = (freqs >= 300.0) & (freqs <= 5000.0)
    edge_high = np.sum(np.abs(np.fft.rfft(edge * np.hanning(n)))[mask])
    centre_high = np.sum(np.abs(np.fft.rfft(centre * np.hanning(n)))[mask])
    assert edge_high > centre_high


def test_unknown_stroke_raises() -> None:
    with pytest.raises(ValueError):
        render_stroke(_state(), "Not a stroke", fs=FS, duration_s=DURATION)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        render_stroke(_state(), StrokeType.BAYAN_OPEN, fs=FS, duration_s=0.0)


def test_soft_limit_bounds() -> None:
    x = soft_limit(np.array([-5.0, -1.0, 0.0, 1.0, 5.0]))
    assert np.all(np.abs(x) <= 1.0)
    assert x[np.argmax(np.abs(x))] < 1.0


def test_waveform_envelope_preserves_peak() -> None:
    x = _stroke(StrokeType.COMBINED)
    envelope = waveform_envelope(x, max_points=256)
    assert envelope.shape == (512,)
    assert np.abs(envelope).max() == pytest.approx(np.abs(x).max(), rel=0.05)

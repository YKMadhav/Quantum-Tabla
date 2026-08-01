"""Unit tests for the bounded overlapping-stroke mixer."""

from __future__ import annotations

import numpy as np
import pytest

from src.performance.mixer import StrokeMixer
from src.synthesis.tabla import soft_limit

FS = 44100.0
BLOCK = 256
CEILING = 0.95


def _mixer(**kwargs) -> StrokeMixer:
    kwargs.setdefault("fs", FS)
    kwargs.setdefault("block_size", BLOCK)
    kwargs.setdefault("ring_seconds", 2.5)
    kwargs.setdefault("output_ceiling", CEILING)
    return StrokeMixer(**kwargs)


def test_consume_returns_block_and_advances_playhead() -> None:
    mixer = _mixer()
    block = mixer.consume_block()
    assert block.shape == (BLOCK,)
    assert mixer.playhead == BLOCK


def test_write_then_consume_returns_signal() -> None:
    mixer = _mixer()
    stroke = np.ones(BLOCK, dtype=np.float64) * 0.5
    mixer.write_stroke(time_samples=BLOCK, buffer=stroke)
    first = mixer.consume_block()  # silence before the stroke
    second = mixer.consume_block()
    assert np.max(np.abs(first)) == 0.0
    assert np.allclose(second, soft_limit(0.5, CEILING))


def test_overlapping_strokes_sum() -> None:
    mixer = _mixer()
    half = np.full(BLOCK, 0.2, dtype=np.float64)
    mixer.write_stroke(time_samples=BLOCK, buffer=half)
    mixer.write_stroke(time_samples=BLOCK + BLOCK // 2, buffer=half)
    mixer.consume_block()
    block = mixer.consume_block()
    assert block[0] == pytest.approx(soft_limit(0.2, CEILING), abs=0.02)
    assert block[BLOCK // 2] == pytest.approx(soft_limit(0.2, CEILING) * 2, abs=0.02)


def test_output_stays_within_soft_limit() -> None:
    mixer = _mixer(ring_seconds=1.0, output_ceiling=0.9)
    loud = np.full(BLOCK, 1.0, dtype=np.float64)
    mixer.write_stroke(time_samples=BLOCK, buffer=loud)
    mixer.write_stroke(time_samples=BLOCK + 1, buffer=loud)
    mixer.write_stroke(time_samples=BLOCK + 2, buffer=loud)
    for _ in range(2):
        block = mixer.consume_block()
        assert np.max(np.abs(block)) <= 0.9 + 1e-9


def test_recent_audio_returns_last_samples() -> None:
    mixer = _mixer()
    stroke = np.ones(BLOCK * 2, dtype=np.float64) * 0.3
    mixer.write_stroke(time_samples=BLOCK, buffer=stroke)
    mixer.consume_block()
    mixer.consume_block()
    recent = mixer.recent_audio(BLOCK)
    assert len(recent) == BLOCK
    assert np.allclose(recent, soft_limit(0.3, CEILING))


def test_recent_audio_before_any_data_is_silence() -> None:
    mixer = _mixer()
    recent = mixer.recent_audio(BLOCK)
    assert len(recent) == BLOCK
    assert np.allclose(recent, 0.0)


def test_ring_capacity_is_bounded() -> None:
    mixer = _mixer(ring_seconds=1.0)
    assert mixer._capacity <= int(round(1.0 * FS))
    # Long performance never grows the ring.
    for _ in range(1000):
        mixer.write_stroke(time_samples=mixer.playhead + BLOCK, buffer=np.ones(BLOCK))
        mixer.consume_block()
    assert mixer._capacity <= int(round(1.0 * FS))


def test_prepared_seconds_tracks_ahead() -> None:
    mixer = _mixer()
    assert mixer.prepared_seconds == 0.0
    mixer.write_stroke(time_samples=BLOCK * 4, buffer=np.ones(BLOCK * 4))
    assert mixer.prepared_seconds > 0.0


def test_stale_ring_content_is_replaced_after_a_full_lap() -> None:
    # A stroke written one full ring lap later must replace (not add to) the
    # already-played content in the same ring slots.
    mixer = _mixer(ring_seconds=0.5, output_ceiling=1.0)
    cap = int(0.5 * FS)
    stroke = np.full(BLOCK, 0.4, dtype=np.float64)

    mixer.write_stroke(time_samples=BLOCK, buffer=stroke)
    while mixer.playhead < 2 * BLOCK:
        mixer.consume_block()  # play the first stroke out

    # Same ring slots as the first stroke (BLOCK % cap .. 2*BLOCK % cap).
    t1 = BLOCK + cap
    start_abs = mixer.playhead
    mixer.write_stroke(time_samples=t1, buffer=stroke)

    out = []
    while mixer.playhead < t1 + BLOCK:
        out.append(mixer.consume_block())
    rec = np.concatenate(out)
    second = rec[t1 - start_abs : t1 + BLOCK - start_abs]

    # Only the new stroke is heard — no stale accumulation from the first.
    np.testing.assert_allclose(second, soft_limit(0.4, 1.0), atol=1e-9)


def test_overlapping_strokes_still_sum_within_the_same_lap() -> None:
    # Overlap (unplayed) must keep summing even with stale tracking active.
    mixer = _mixer(output_ceiling=1.0)
    half = np.full(BLOCK, 0.2, dtype=np.float64)
    mixer.write_stroke(time_samples=BLOCK, buffer=half)
    mixer.write_stroke(time_samples=BLOCK + BLOCK // 2, buffer=half)
    mixer.consume_block()
    block = mixer.consume_block()
    assert block[0] == pytest.approx(soft_limit(0.2, 1.0), abs=0.02)
    assert block[BLOCK // 2] == pytest.approx(soft_limit(0.4, 1.0), abs=0.02)


def test_recording_captures_consumed_blocks() -> None:
    mixer = _mixer()
    stroke = np.full(BLOCK * 2, 0.25, dtype=np.float64)
    mixer.write_stroke(time_samples=BLOCK, buffer=stroke)
    mixer.start_recording()
    first = mixer.consume_block()
    second = mixer.consume_block()
    recording = mixer.stop_recording()
    np.testing.assert_array_equal(recording, np.concatenate([first, second]))


def test_recording_resets_between_takes() -> None:
    mixer = _mixer()
    mixer.start_recording()
    mixer.consume_block()
    first = mixer.stop_recording()
    mixer.start_recording()
    mixer.consume_block()
    second = mixer.stop_recording()
    assert len(first) == BLOCK
    assert len(second) == BLOCK

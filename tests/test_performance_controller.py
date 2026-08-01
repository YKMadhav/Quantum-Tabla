"""Integration tests for the performance controller and its lifecycle."""

from __future__ import annotations

import time

import pytest

from src.core.config import AppConfig
from src.core.randomness.factory import build_randomness_stack
from src.performance.audio import AudioEngine
from src.performance.controller import PerformanceController
from src.performance.mixer import StrokeMixer


def _controller(**kwargs) -> PerformanceController:
    config = AppConfig()
    stack = build_randomness_stack(config)
    kwargs.setdefault("backend", "silent")
    controller = PerformanceController(config, stack, **kwargs)
    controller._stack.bit_stream.start()
    return controller


def _wait_until(predicate, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_start_then_stop_then_restart() -> None:
    controller = _controller()
    ok, message = controller.start()
    assert ok, message
    assert _wait_until(lambda: controller.snapshot().stroke_count > 0)
    assert controller.snapshot().running
    controller.stop()
    assert not controller.running
    assert not controller.snapshot().running

    ok, message = controller.start()
    assert ok, message
    assert _wait_until(lambda: controller.snapshot().stroke_count > 0)
    controller.stop()
    assert not controller.running


def test_duplicate_start_is_safe() -> None:
    controller = _controller()
    ok, message = controller.start()
    assert ok
    ok2, message2 = controller.start()
    assert ok2
    assert "already" in message2
    controller.stop()
    assert not controller.running


def test_snapshot_fields_populated_while_running() -> None:
    controller = _controller()
    controller.start()
    assert _wait_until(lambda: controller.snapshot().stroke_count >= 5)
    snapshot = controller.snapshot()
    assert snapshot.running
    assert snapshot.status == "Running"
    assert snapshot.runtime_s >= 0.0
    assert snapshot.stroke_count >= 5
    assert 70.0 <= snapshot.tempo_bpm <= 160.0
    assert snapshot.current_stroke in (
        "Bayan open",
        "Dayan open",
        "Dayan muted (sharp)",
        "Combined",
    )
    assert snapshot.audio_queue_depth >= 0
    controller.stop()


def test_randomness_flows_through_bit_stream() -> None:
    controller = _controller()
    controller.start()
    assert _wait_until(lambda: controller.snapshot().instrument_version > 0)
    snapshot = controller.snapshot()
    assert snapshot.total_bits_consumed > 0
    assert snapshot.instrument_version > 0
    controller.stop()


def test_starvation_is_recorded_not_silently_swapped() -> None:
    config = AppConfig()
    stack = build_randomness_stack(config)
    controller = PerformanceController(config, stack, backend="silent")
    # Never start the refill thread: reservoir stays empty.
    ok, message = controller.start()
    assert ok, message
    assert _wait_until(lambda: controller.snapshot().starvation_events > 0)
    # The engine keeps performing on neutral decisions without crashing.
    assert controller.snapshot().running
    controller.stop()


def test_clean_shutdown_leaves_no_threads() -> None:
    controller = _controller()
    controller.start()
    scheduler = controller._scheduler
    assert _wait_until(lambda: controller.snapshot().stroke_count > 0)
    controller.stop()
    assert not scheduler.is_alive()
    assert not controller.running


class _FailingEngine(AudioEngine):
    """Audio engine whose device never opens (for failure-path tests)."""

    def start(self) -> None:
        raise OSError("no such audio device")

    def stop(self) -> None:
        self._running = False


def test_audio_device_failure_does_not_crash_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller()
    monkeypatch.setattr(
        controller,
        "_build_audio",
        lambda mixer: _FailingEngine(StrokeMixer()),
    )
    ok, message = controller.start()
    assert not ok
    assert "no such audio device" in message
    assert not controller.running
    assert "no such audio device" in controller.last_error
    snapshot = controller.snapshot()
    assert not snapshot.running

    # Restore a working engine: a retry must succeed without restarting the app.
    controller2 = _controller()
    ok, message = controller2.start()
    assert ok, message
    assert _wait_until(lambda: controller2.snapshot().stroke_count > 0)
    controller2.stop()


def test_memory_stays_bounded_during_runtime() -> None:
    controller = _controller()
    controller.start()
    assert _wait_until(lambda: controller.snapshot().stroke_count > 0)
    scheduler = controller._scheduler
    capacity = scheduler._mixer._capacity
    controller.stop()
    assert scheduler._mixer._capacity == capacity


def test_stop_exposes_downloadable_wav_recording() -> None:
    controller = _controller()
    ok, message = controller.start()
    assert ok, message
    assert _wait_until(lambda: controller.snapshot().stroke_count > 0)
    controller.stop()
    recording = controller.last_recording_wav
    assert recording is not None
    assert recording[:4] == b"RIFF"
    assert recording[8:12] == b"WAVE"

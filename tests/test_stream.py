"""Unit tests for the bit stream manager and its refill thread."""

from __future__ import annotations

import time

from src.core.config import RandomnessConfig
from src.core.randomness.classical import ClassicalRandomProvider
from src.core.randomness.stream import BitStreamManager


def _manager(
    target: int = 512,
    batch: int = 128,
    interval: float = 0.005,
) -> BitStreamManager:
    provider = ClassicalRandomProvider(seed=42, buffer_size=256)
    return BitStreamManager(
        provider,
        target_bits=target,
        refill_batch_bits=batch,
        refill_interval_s=interval,
    )


def test_initial_state() -> None:
    manager = _manager()
    assert manager.available() == 0
    assert manager.stats.total_generated == 0
    assert not manager.running


def test_refill_reaches_target() -> None:
    manager = _manager()
    manager.start()
    try:
        deadline = time.monotonic() + 2.0
        while manager.available() < 512 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert manager.available() == 512
        assert manager.stats.total_generated >= 512
    finally:
        manager.stop()


def test_take_returns_requested_bits() -> None:
    manager = _manager()
    manager.start()
    try:
        deadline = time.monotonic() + 2.0
        while manager.available() < 512 and time.monotonic() < deadline:
            time.sleep(0.01)
        bits = manager.take(100)
        assert len(bits) == 100
        assert set(bits) <= {"0", "1"}
        assert manager.stats.total_consumed == 100
    finally:
        manager.stop()


def test_stop_frees_thread() -> None:
    manager = _manager()
    manager.start()
    assert manager.running
    manager.stop()
    assert not manager.running
    time.sleep(0.03)
    assert not manager.running


def test_clear_drains_reservoir() -> None:
    manager = _manager()
    manager.start()
    try:
        deadline = time.monotonic() + 2.0
        while manager.available() < 512 and time.monotonic() < deadline:
            time.sleep(0.01)
        manager.clear()
        assert manager.available() == 0
    finally:
        manager.stop()


def test_config_defaults_are_sane() -> None:
    config = RandomnessConfig()
    assert config.chunk_size_bits == 10
    assert config.slow.refresh_every > config.medium.refresh_every
    assert config.medium.refresh_every > config.fast.refresh_every
    assert config.slow.smoothing < config.medium.smoothing < config.fast.smoothing

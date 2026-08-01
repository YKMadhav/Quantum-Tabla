"""Unit tests for the instrument state manager and update scheduler."""

from __future__ import annotations

import pytest

from src.core.config import AppConfig
from src.core.randomness.factory import build_randomness_stack
from src.core.randomness.mapper import UpdateTier


@pytest.fixture()
def stack():
    stack = build_randomness_stack(AppConfig())
    stack.instrument.reset()
    return stack


def test_first_step_adopts_targets(stack) -> None:
    state = stack.instrument.step(lambda n: "1" * n)
    assert state.version == 1
    assert state.value("Membrane tension") == 1.0
    assert state.value("Strike position") == 1.0


def test_state_placeholder_before_initialisation(stack) -> None:
    state = stack.instrument.state
    assert state.version == 0
    assert state.value("Membrane tension") == 0.5


def test_fast_parameter_interpolates_after_refresh(stack) -> None:
    # Frame 0: all targets 1.0
    stack.instrument.step(lambda n: "1" * n)
    # Frames 1-2: no refresh (fast stride = 3); target still 1.0
    stack.instrument.step(lambda n: "0" * n)
    stack.instrument.step(lambda n: "0" * n)
    assert stack.instrument.state.value("Strike position") == pytest.approx(1.0)
    # Frame 3: fast tier refreshes target to 0.0 -> interpolate one step
    state = stack.instrument.step(lambda n: "0" * n)
    assert state.value("Strike position") == pytest.approx(1.0 * (1 - 0.30))


def test_slow_parameter_refreshes_less_often(stack) -> None:
    stack.instrument.step(lambda n: "1" * n)  # frame 0
    for _ in range(47):
        stack.instrument.step(lambda n: "0" * n)
    # Before frame 48 slow target has not refreshed -> stays 1.0
    assert stack.instrument.state.value("Membrane tension") == pytest.approx(1.0)
    # Frame 48: slow refreshes to 0.0 and interpolates at smoothing 0.03
    state = stack.instrument.step(lambda n: "0" * n)
    assert state.value("Membrane tension") == pytest.approx(1.0 * (1 - 0.03))


def test_values_stay_normalised(stack) -> None:
    for _ in range(100):
        state = stack.instrument.step(lambda n: ("01" * (n // 2 + 1))[:n])
    values = state.as_dict().values()
    assert values
    assert all(0.0 <= v <= 1.0 for v in values)


def test_reset_starts_fresh(stack) -> None:
    stack.instrument.step(lambda n: "1" * n)
    assert stack.instrument.version >= 1
    stack.instrument.reset()
    assert stack.instrument.version == 0
    assert stack.instrument.state.version == 0


def test_scheduler_refresh_hierarchy() -> None:
    config = AppConfig()
    stack = build_randomness_stack(config)
    scheduler = stack.scheduler
    counts = {UpdateTier.SLOW: 0, UpdateTier.MEDIUM: 0, UpdateTier.FAST: 0}
    for _ in range(48):
        for index in scheduler.next_frame():
            counts[stack.mapper.catalog.spec_at(index).tier] += 1
    assert counts[UpdateTier.SLOW] < counts[UpdateTier.MEDIUM]
    assert counts[UpdateTier.MEDIUM] < counts[UpdateTier.FAST]
    # Frame 0 refreshes everything: over 48 frames each slow slot refreshes once.
    n_slow_slots = sum(
        1 for spec in stack.mapper.catalog.specs if spec.tier is UpdateTier.SLOW
    )
    assert counts[UpdateTier.SLOW] == n_slow_slots

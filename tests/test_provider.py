"""Unit tests for the randomness providers and the provider factory."""

from __future__ import annotations

import pytest

from src.core.config import AppConfig
from src.core.randomness import qiskit_provider as qp
from src.core.randomness.classical import ClassicalRandomProvider
from src.core.randomness.factory import build_randomness_stack
from src.utils import bits as bit_utils


def test_provider_returns_valid_bits() -> None:
    provider = ClassicalRandomProvider(seed=7)
    bits = provider.request_bits(1000)
    assert len(bits) == 1000
    assert bit_utils.validate_bitstring(bits)


def test_provider_is_deterministic_given_seed() -> None:
    first = ClassicalRandomProvider(seed=1).request_bits(200)
    second = ClassicalRandomProvider(seed=1).request_bits(200)
    assert first == second


def test_provider_reports_status() -> None:
    provider = ClassicalRandomProvider()
    status = provider.status()
    assert status.online
    assert status.is_quantum is False
    assert status.name == "numpy-classical"


def test_provider_bits_are_balanced() -> None:
    provider = ClassicalRandomProvider(seed=3)
    bits = provider.request_bits(10_000)
    ones = bits.count("1")
    ratio = ones / len(bits)
    assert 0.45 < ratio < 0.55


def test_clear_buffer_resets_availability() -> None:
    provider = ClassicalRandomProvider(seed=5, buffer_size=64)
    provider.request_bits(8)
    provider.clear_buffer()
    assert provider.available_bits() == 0
    bits = provider.request_bits(8)
    assert len(bits) == 8


def test_classical_stats_track_generation() -> None:
    provider = ClassicalRandomProvider(seed=9)
    provider.request_bits(1_000)
    stats = provider.stats()
    assert stats.total_generated >= 1_000
    assert stats.zero_count + stats.one_count == stats.total_generated


def _config(provider_name: str) -> AppConfig:
    from dataclasses import replace

    base = AppConfig()
    return replace(
        base,
        randomness=replace(base.randomness, provider_name=provider_name),
    )


def test_factory_builds_classical_stack() -> None:
    stack = build_randomness_stack(_config("classical"))
    assert stack.provider.name == "numpy-classical"
    assert stack.provider.status().is_quantum is False


@pytest.mark.skipif(
    qp._QISKIT_IMPORT_ERROR is not None,
    reason="Qiskit / qiskit-aer not installed",
)
def test_factory_builds_quantum_stack() -> None:
    stack = build_randomness_stack(_config("quantum"))
    assert stack.provider.name == "qiskit-aer-simulation"
    assert stack.provider.status().is_quantum is True


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown randomness provider"):
        build_randomness_stack(_config("entangled"))

"""Unit tests for the Qiskit Aer quantum-circuit randomness provider."""

from __future__ import annotations

import time

import pytest

from src.core.randomness import qiskit_provider as qp
from src.core.randomness.classical import ClassicalRandomProvider
from src.core.randomness.stream import BitStreamManager
from src.utils import bits as bit_utils

pytestmark = pytest.mark.skipif(
    qp._QISKIT_IMPORT_ERROR is not None,
    reason="Qiskit / qiskit-aer not installed",
)


def test_memory_to_bits_lists_qubit_zero_first() -> None:
    # With the standard wiring measure(i, i), a shot string is indexed by
    # classical bit, so its *rightmost* character is qubit 0. Reversing the
    # shot therefore lists qubits 0..n-1 in order.
    assert qp.memory_to_bits(["10", "01"], 2) == "0110"
    assert qp.memory_to_bits(["000", "111"], 3) == "000111"
    assert qp.memory_to_bits(["1011"], 4) == "1101"


def test_memory_to_bits_rejects_truncated_shots() -> None:
    with pytest.raises(ValueError):
        qp.memory_to_bits(["01"], 3)


def test_provider_reports_quantum_status() -> None:
    provider = qp.QiskitRandomnessProvider(num_qubits=4)
    status = provider.status()
    assert status.online
    assert status.is_quantum is True
    assert status.name == "qiskit-aer-simulation"
    assert "simulator" in status.description.lower()
    assert "Measure" in status.circuit


def test_request_bits_returns_exact_count() -> None:
    provider = qp.QiskitRandomnessProvider(num_qubits=4)
    bits = provider.request_bits(5_000)
    assert len(bits) == 5_000
    assert bit_utils.validate_bitstring(bits)


def test_request_bits_are_binary_only() -> None:
    provider = qp.QiskitRandomnessProvider(num_qubits=2)
    bits = provider.request_bits(1_000)
    assert set(bits) <= {"0", "1"}


def test_requests_are_concatenated() -> None:
    provider = qp.QiskitRandomnessProvider(num_qubits=4)
    first = provider.request_bits(512)
    second = provider.request_bits(512)
    third = provider.request_bits(512)
    assert len(first) == len(second) == len(third) == 512
    combined = first + second + third
    assert bit_utils.validate_bitstring(combined)


def test_batch_generation_tracks_provider_stats() -> None:
    provider = qp.QiskitRandomnessProvider(
        num_qubits=4, batch_shots=2048, low_water_bits=2048
    )
    provider.request_bits(4_000)
    stats = provider.stats()
    assert stats.total_generated >= 4_000
    assert stats.zero_count + stats.one_count == stats.total_generated


def test_clear_buffer_resets_availability() -> None:
    provider = qp.QiskitRandomnessProvider(num_qubits=4, low_water_bits=512)
    provider.request_bits(100)
    assert provider.available_bits() > 0
    provider.clear_buffer()
    assert provider.available_bits() == 0
    bits = provider.request_bits(100)
    assert len(bits) == 100


def test_bits_are_statistically_balanced() -> None:
    provider = qp.QiskitRandomnessProvider(
        num_qubits=4, batch_shots=8192, low_water_bits=8192
    )
    bits = provider.request_bits(20_000)
    assert len(bits) == 20_000
    ones = bits.count("1")
    ratio = ones / len(bits)
    assert 0.45 < ratio < 0.55
    assert bits.count("0") > 0 and ones > 0
    assert "01" in bits or "10" in bits


def test_provider_usable_after_stop_start_like_cycle() -> None:
    provider = qp.QiskitRandomnessProvider(num_qubits=4)
    first = provider.request_bits(1_000)
    provider.clear_buffer()
    second = provider.request_bits(1_000)
    assert len(first) == len(second) == 1_000
    assert provider.status().online


def test_offline_provider_returns_no_bits(monkeypatch) -> None:
    monkeypatch.setattr(qp, "_QISKIT_IMPORT_ERROR", "simulated")
    provider = qp.QiskitRandomnessProvider()
    assert provider.status().online is False
    assert provider.request_bits(100) == ""
    assert provider.available_bits() == 0


def test_job_failure_degrades_gracefully(monkeypatch) -> None:
    provider = qp.QiskitRandomnessProvider(num_qubits=4)
    assert provider.status().online

    def _raise(self) -> str:  # noqa: ANN001
        raise RuntimeError("simulated simulator failure")

    monkeypatch.setattr(qp.QiskitRandomnessProvider, "_run_job", _raise)
    bits = provider.request_bits(1_000)
    assert bits == ""
    assert "Simulator job failed" in provider.status().description


def test_quantum_provider_through_stream_manager() -> None:
    provider = qp.QiskitRandomnessProvider(
        num_qubits=4, batch_shots=2048, low_water_bits=2048
    )
    manager = BitStreamManager(
        provider,
        target_bits=2048,
        refill_batch_bits=1024,
        refill_interval_s=0.005,
    )
    manager.start()
    try:
        deadline = time.monotonic() + 20.0
        while manager.available() < 2048 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert manager.available() == 2048
        bits = manager.take(2048)
        assert len(bits) == 2048
        assert bit_utils.validate_bitstring(bits)
        assert manager.status().is_quantum is True
    finally:
        manager.stop()


def test_classical_and_quantum_are_interchangeable() -> None:
    classical = ClassicalRandomProvider(seed=7)
    quantum = qp.QiskitRandomnessProvider(num_qubits=4)
    for provider in (classical, quantum):
        bits = provider.request_bits(2_000)
        assert len(bits) == 2_000
        assert set(bits) <= {"0", "1"}
        assert provider.status().is_quantum is (provider is quantum)

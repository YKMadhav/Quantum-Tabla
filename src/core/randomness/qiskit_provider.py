"""Qiskit Aer quantum-circuit randomness provider.

Each circuit prepares ``num_qubits`` independent qubits in an equal
superposition with an H (Hadamard) gate and immediately measures every qubit
into its own classical bit::

    QuantumCircuit(n, n)
    h(range(n))
    measure(range(n), range(n))

Running the circuit on the Aer simulator collapses each qubit to ``0`` or
``1`` with probability 1/2, so every shot yields ``num_qubits`` fair, binary
measurement outcomes. Aer is a *classical simulator* of quantum circuits: the
results are produced by a classical random-number engine implementing the
measurement statistics, not by physical hardware. The dashboard and
documentation always describe it that way.

Bit ordering
------------
Aer returns each shot as a string whose characters are indexed by *classical
bit*: with the standard wiring ``measure(i, i)`` the character at index ``k``
is the outcome of qubit ``n - 1 - k``. That means the *leftmost* character is
qubit ``n - 1`` and the *rightmost* character is qubit ``0``. To keep the
stream deterministic and qubit 0 first, :func:`memory_to_bits` reverses each
shot string. Every consumer receives a stream where the first ``num_qubits``
bits are the outcomes of qubits ``0..n-1`` in order.
"""

from __future__ import annotations

import math
import threading

from src.core.randomness.base import (
    ProviderStats,
    ProviderStatus,
    RandomnessProvider,
)
from src.core.randomness.buffer import BitBuffer

try:  # pragma: no cover - depends on the deployed environment
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    _QISKIT_IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 - report, never crash
    QuantumCircuit = None  # type: ignore[assignment,misc]
    transpile = None  # type: ignore[assignment,misc]
    AerSimulator = None  # type: ignore[assignment,misc]
    _QISKIT_IMPORT_ERROR = (
        "Qiskit is not available (install 'qiskit' and 'qiskit-aer' to use "
        f"the quantum provider): {exc}"
    )


def memory_to_bits(memory: list[str], num_qubits: int) -> str:
    """Flatten Aer measurement strings into one bitstream, qubit 0 first.

    Each shot string is ``num_qubits`` characters long; character ``k`` holds
    the outcome of qubit ``n - 1 - k`` under the standard ``measure(i, i)``
    wiring. Reversing the string therefore lists qubits ``0..n-1`` in order,
    and concatenating the reversed shots gives the continuous stream.

    Raises:
        ValueError: if any shot is shorter than ``num_qubits``.
    """
    parts: list[str] = []
    for shot in memory:
        if len(shot) < num_qubits:
            raise ValueError(
                f"Shot {shot!r} is shorter than the circuit's {num_qubits} "
                "classical bits; results may be truncated or misaligned."
            )
        parts.append(shot[::-1][:num_qubits])
    return "".join(parts)


class QiskitRandomnessProvider(RandomnessProvider):
    """A quantum-circuit provider backed by the Qiskit Aer simulator.

    Bits are produced by submitting batch jobs for the reused Hadamard
    measurement circuit. Results land in an internal reservoir that refills
    whenever it drops below a low-water mark, so callers are served from
    memory and the simulator is only consulted in batches.

    If Qiskit is missing or the simulator fails, the provider reports itself
    offline and returns no bits — it never substitutes another entropy source
    and never crashes the application.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        batch_shots: int = 4096,
        low_water_bits: int = 4096,
        max_buffer_bits: int = 1 << 18,
    ) -> None:
        self._num_qubits = max(1, int(num_qubits))
        self._batch_shots = max(1, int(batch_shots))
        self._low_water = max(1, int(low_water_bits))
        self._max_buffer = max(self._low_water, int(max_buffer_bits))
        self._lock = threading.RLock()
        self._buffer = BitBuffer()
        self._available = False
        self._error: str | None = None
        self._generated = 0
        self._zeros = 0
        self._ones = 0
        self._sim = None
        self._circuit = None
        self._setup()

    # -- construction ---------------------------------------------------------

    def _setup(self) -> None:
        """Build the reusable circuit and simulator (one per provider)."""
        if _QISKIT_IMPORT_ERROR is not None:
            self._error = _QISKIT_IMPORT_ERROR
            return
        try:
            n = self._num_qubits
            circuit = QuantumCircuit(n, n)
            circuit.h(range(n))
            circuit.measure(range(n), range(n))
            simulator = AerSimulator()
            self._circuit = transpile(circuit, simulator)
            self._sim = simulator
            self._available = True
        except Exception as exc:  # noqa: BLE001 - report, never crash
            self._error = f"Failed to build the quantum circuit: {exc}"

    # -- interface ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "qiskit-aer-simulation"

    def request_bits(self, count: int) -> str:
        """Return up to *count* bits, generating them in batches as needed."""
        if count <= 0 or not self._available:
            return ""
        with self._lock:
            needed = max(count, self._low_water)
            if self._buffer.available < needed:
                self._refill_until(needed)
            return self._buffer.take(count)

    def available_bits(self) -> int:
        return self._buffer.available

    def clear_buffer(self) -> None:
        with self._lock:
            self._buffer.clear()

    def status(self) -> ProviderStatus:
        description = "Qiskit Aer — classical simulator of quantum circuits"
        if self._error is not None:
            description += f" · {self._error}"
        return ProviderStatus(
            name=self.name,
            online=self._available,
            description=description,
            is_quantum=True,
            circuit=self._circuit_summary(),
        )

    def stats(self) -> ProviderStats:
        with self._lock:
            return ProviderStats(
                total_generated=self._generated,
                zero_count=self._zeros,
                one_count=self._ones,
            )

    # -- internals ------------------------------------------------------------

    def _circuit_summary(self) -> str:
        n = self._num_qubits
        gates = "H" if n == 1 else f"H(0..{n - 1})"
        return f"{gates} → Measure · {n} qubits · {self._batch_shots} shots/job"

    def _refill_until(self, wanted: int) -> None:
        """Run batch jobs until the reservoir holds at least *wanted* bits.

        Stops early if a job fails; the recorded error is surfaced through
        :meth:`status` and the reservoir keeps whatever it already holds.
        """
        per_job = self._batch_shots * self._num_qubits
        remaining = wanted - self._buffer.available
        jobs = max(1, math.ceil(remaining / per_job))
        for _ in range(jobs):
            if self._buffer.available >= wanted:
                break
            try:
                bits = self._run_job()
            except Exception as exc:  # noqa: BLE001 - degrade, never crash
                self._error = f"Simulator job failed: {exc}"
                return
            if not bits:
                self._error = "Simulator returned no bits"
                return
            self._buffer.append(bits)
            self._generated += len(bits)
            self._zeros += bits.count("0")
            self._ones += len(bits) - bits.count("0")
            if self._buffer.available > self._max_buffer:
                return

    def _run_job(self) -> str:
        """Execute one batch of shots on the reused circuit and flatten them."""
        result = self._sim.run(  # type: ignore[union-attr]
            self._circuit, shots=self._batch_shots, memory=True
        ).result()
        memory = result.get_memory(self._circuit)
        if not isinstance(memory, list) or len(memory) == 0:
            return ""
        return memory_to_bits(memory, self._num_qubits)

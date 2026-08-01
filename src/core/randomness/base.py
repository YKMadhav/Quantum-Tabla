"""Abstract randomness provider interface.

A provider is the single point that turns an entropy source into a stream of
binary digits. Consumers of randomness interact only with this interface, so
the entropy source (NumPy, Python ``random``, Qiskit, IBM Quantum, a QRNG API
or a hardware device) can be swapped without touching any other module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderStatus:
    """Read-only snapshot of a randomness provider's health."""

    name: str
    online: bool
    description: str
    is_quantum: bool = False
    #: Short textual description of the entropy circuit ("" when not applicable).
    circuit: str = ""


@dataclass(frozen=True)
class ProviderStats:
    """Lifetime generation counters for a randomness provider.

    Counts refer to bits produced by the *entropy source* (not bits served
    from an internal reservoir), so ``total_generated`` can exceed the
    number of bits actually consumed by the application.
    """

    total_generated: int = 0
    zero_count: int = 0
    one_count: int = 0

    @property
    def ones_ratio(self) -> float:
        """Proportion of generated bits that were ``'1'`` (0.0 if empty)."""
        total = self.total_generated
        if total <= 0:
            return 0.0
        return self.one_count / total


class RandomnessProvider(ABC):
    """Abstract continuous source of binary data.

    Implementations return bits as strings of ``'0'`` and ``'1'``. The
    interface intentionally says nothing about *how* the bits are produced;
    providers may buffer internally and may block inside :meth:`request_bits`
    when their source is slow.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable identifier for the provider."""

    @abstractmethod
    def request_bits(self, count: int) -> str:
        """Return up to *count* random bits.

        A healthy provider returns exactly *count* bits; a degraded or slow
        source may return fewer. Callers must treat the result as a prefix of
        the requested stream and never as a guaranteed length.
        """

    @abstractmethod
    def available_bits(self) -> int:
        """Return how many bits are immediately available without blocking."""

    @abstractmethod
    def clear_buffer(self) -> None:
        """Discard any internally buffered bits."""

    @abstractmethod
    def status(self) -> ProviderStatus:
        """Return a snapshot of the provider's current status."""

    @abstractmethod
    def stats(self) -> ProviderStats:
        """Return lifetime generation counters for this provider."""

"""Immutable instrument state snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.randomness.mapper import ParameterVector


@dataclass(frozen=True)
class InstrumentState:
    """An immutable snapshot of the virtual instrument.

    ``parameters`` holds every normalised synthesis value; ``version`` counts
    update frames and ``created_at_s`` is the monotonic timestamp of creation.
    A version of 0 means the instrument has not been initialised yet.
    """

    parameters: ParameterVector
    version: int = 0
    created_at_s: float = 0.0

    def value(self, name: str, default: float = 0.0) -> float:
        """Return the current normalised value for *name*."""
        return self.parameters.get(name, default)

    def as_dict(self) -> dict[str, float]:
        """Return a plain mutable copy of the parameter values."""
        return self.parameters.as_dict()

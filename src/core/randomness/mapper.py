"""Parameter catalogue and mapper.

Defines the named synthesis parameters, their update tiers and the mapping of
normalised random values onto parameter slots. Physical ranges are applied
later by the DSP stage; here every parameter stays a ``[0.0, 1.0]`` value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from src.utils.bits import clamp01


class UpdateTier(Enum):
    """How often and how quickly a parameter evolves."""

    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


@dataclass(frozen=True)
class ParameterSpec:
    """A single named synthesis parameter."""

    name: str
    tier: UpdateTier


@dataclass(frozen=True)
class ParameterCatalog:
    """Ordered, unique list of parameters; position defines the slot index."""

    specs: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate parameter names: {names}")

    def spec_at(self, index: int) -> ParameterSpec:
        return self.specs[index]


#: Default instrument parameter set (16 parameters across the three tiers).
DEFAULT_CATALOG = ParameterCatalog(
    specs=(
        ParameterSpec("Membrane tension", UpdateTier.SLOW),
        ParameterSpec("Membrane diameter", UpdateTier.SLOW),
        ParameterSpec("Shell resonance", UpdateTier.SLOW),
        ParameterSpec("Air resonance", UpdateTier.SLOW),
        ParameterSpec("Brightness", UpdateTier.MEDIUM),
        ParameterSpec("Damping", UpdateTier.MEDIUM),
        ParameterSpec("Noise level", UpdateTier.MEDIUM),
        ParameterSpec("Resonance", UpdateTier.MEDIUM),
        ParameterSpec("Pitch bend", UpdateTier.MEDIUM),
        ParameterSpec("Strike position", UpdateTier.FAST),
        ParameterSpec("Strike velocity", UpdateTier.FAST),
        ParameterSpec("Finger pressure", UpdateTier.FAST),
        ParameterSpec("Attack time", UpdateTier.FAST),
        ParameterSpec("Decay time", UpdateTier.FAST),
        ParameterSpec("Accent", UpdateTier.FAST),
        ParameterSpec("Tempo variation", UpdateTier.FAST),
    )
)


@dataclass(frozen=True)
class ParameterVector:
    """An immutable snapshot of named, normalised parameter values."""

    values: Mapping[str, float]

    def get(self, name: str, default: float = 0.0) -> float:
        """Return the value for *name*, or *default* if absent."""
        return self.values.get(name, default)

    def as_dict(self) -> dict[str, float]:
        """Return a plain mutable copy of the values."""
        return dict(self.values)

    def __len__(self) -> int:
        return len(self.values)


class ParameterMapper:
    """Assigns normalised random values to catalog parameter slots."""

    def __init__(self, catalog: ParameterCatalog = DEFAULT_CATALOG) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> ParameterCatalog:
        return self._catalog

    @property
    def count(self) -> int:
        """Number of parameters in the catalog."""
        return len(self._catalog.specs)

    def build(
        self, values: Sequence[float], indices: Sequence[int]
    ) -> ParameterVector:
        """Pair *values* positionally with catalog slots given by *indices*.

        Only the min of the two lengths is mapped, so partial decoding (when
        the bit reservoir runs short) degrades gracefully.
        """
        result: dict[str, float] = {}
        for index, value in zip(indices, values):
            spec = self._catalog.spec_at(index)
            result[spec.name] = clamp01(value)
        return ParameterVector(result)

    def defaults(self) -> ParameterVector:
        """A vector with every parameter at the neutral value 0.5."""
        return ParameterVector(
            {spec.name: 0.5 for spec in self._catalog.specs}
        )

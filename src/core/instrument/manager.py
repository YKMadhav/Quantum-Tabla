"""Instrument state manager: smooth, continuous parameter evolution.

Holds the *current* and *target* parameter vectors. Each update frame the
scheduler decides which slots get fresh random targets; the manager then
interpolates every parameter toward its target at a per-tier rate. Parameters
therefore drift continuously instead of jumping between unrelated values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from src.core.config import TierBehaviour
from src.core.instrument.state import InstrumentState
from src.core.randomness.decoder import ParameterDecoder
from src.core.randomness.mapper import (
    ParameterCatalog,
    ParameterMapper,
    ParameterVector,
    UpdateTier,
)
from src.core.randomness.scheduler import UpdateScheduler
from src.core.timing import monotonic
from src.utils.bits import clamp01


class InstrumentStateManager:
    """Produces a fresh, smoothly evolving instrument state every step."""

    def __init__(
        self,
        decoder: ParameterDecoder,
        scheduler: UpdateScheduler,
        mapper: ParameterMapper,
        tier_behaviours: Mapping[UpdateTier, TierBehaviour],
    ) -> None:
        self._decoder = decoder
        self._scheduler = scheduler
        self._mapper = mapper
        self._catalog: ParameterCatalog = mapper.catalog
        self._smoothing = {
            spec.name: tier_behaviours[spec.tier].smoothing
            for spec in self._catalog.specs
        }
        self._current: InstrumentState | None = None
        self._target: dict[str, float] | None = None
        self._version: int = 0

    # -- lifecycle ------------------------------------------------------------

    def reset(self) -> None:
        """Start a fresh performance (used when playback begins)."""
        self._current = None
        self._target = None
        self._version = 0
        self._scheduler.reset()

    # -- accessors ------------------------------------------------------------

    @property
    def state(self) -> InstrumentState:
        """Latest instrument snapshot (neutral placeholder before start)."""
        if self._current is None:
            return InstrumentState(self._mapper.defaults(), version=0)
        return self._current

    @property
    def version(self) -> int:
        return self._version

    # -- evolution ------------------------------------------------------------

    def step(self, bits_source: Callable[[int], str]) -> InstrumentState:
        """Advance the instrument one update frame.

        *bits_source* must accept a bit count and return up to that many bits;
        typically this is ``BitStreamManager.take``.
        """
        indices = self._scheduler.next_frame()
        if indices:
            fresh = self._decoder.decode(indices, bits_source)
            if self._target is None:
                self._target = fresh.as_dict()
            else:
                self._target.update(fresh.as_dict())

        if self._target is None:
            self._target = self._mapper.defaults().as_dict()

        if self._current is None:
            self._version += 1
            self._current = InstrumentState(
                parameters=ParameterVector(dict(self._target)),
                version=self._version,
                created_at_s=monotonic(),
            )
            return self._current

        current = self._current.as_dict()
        evolved: dict[str, float] = {}
        for spec in self._catalog.specs:
            base = current.get(spec.name, 0.5)
            goal = self._target.get(spec.name, 0.5)
            factor = self._smoothing[spec.name]
            evolved[spec.name] = clamp01(base + (goal - base) * factor)

        self._version += 1
        self._current = InstrumentState(
            parameters=ParameterVector(evolved),
            version=self._version,
            created_at_s=monotonic(),
        )
        return self._current

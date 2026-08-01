"""Update scheduler: decides which parameter targets refresh each frame.

Parameters evolve at different speeds. Each tier refreshes its random target
only every N frames (slow least often, fast most often); interpolation then
does the rest. This gives the instrument a slow-changing character with fast
per-stroke variation.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.core.config import TierBehaviour
from src.core.randomness.mapper import ParameterCatalog, UpdateTier


class UpdateScheduler:
    """Cycles the frame counter and reports which slots need fresh targets."""

    def __init__(
        self,
        catalog: ParameterCatalog,
        tier_behaviours: Mapping[UpdateTier, TierBehaviour],
    ) -> None:
        self._catalog = catalog
        self._behaviours: dict[UpdateTier, TierBehaviour] = dict(tier_behaviours)
        self._frame: int = 0

    def reset(self) -> None:
        """Restart the frame cycle (called on every new performance)."""
        self._frame = 0

    def next_frame(self) -> tuple[int, ...]:
        """Advance one frame and return slots to refresh during it.

        Frame 0 refreshes every slot; afterwards each tier refreshes when the
        frame counter is a multiple of its ``refresh_every`` stride.
        """
        frame = self._frame
        self._frame += 1
        indices: list[int] = []
        for index, spec in enumerate(self._catalog.specs):
            stride = self._behaviours[spec.tier].refresh_every
            if frame % stride == 0:
                indices.append(index)
        return tuple(indices)

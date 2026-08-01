"""Performance snapshots and thread-safe snapshot exchange.

The performance engine runs on its own thread while Streamlit only *reads*.
Every mutable view the dashboard needs is captured into one immutable
:class:`PerformanceSnapshot` that the scheduler publishes at a fixed cadence;
assignment of a completed snapshot is atomic under the GIL, so the UI thread
can read the latest snapshot without locks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Immutable, point-in-time view of the running performance.

    ``recent_waveform`` is a bounded min/max envelope of the recent audio
    output, already down-sampled for display; it never holds raw audio.
    """

    running: bool = False
    status: str = "Stopped"
    error: Optional[str] = None
    runtime_s: float = 0.0
    stroke_count: int = 0
    current_stroke: Optional[str] = None
    tempo_bpm: float = 0.0
    current_pitch_hz: float = 0.0
    accent: float = 0.0
    rhythm_density: float = 0.0
    entropy_rate: float = 0.0
    random_buffer_bits: int = 0
    total_bits_consumed: int = 0
    audio_queue_depth: int = 0
    audio_underruns: int = 0
    audio_buffer_fill: float = 0.0
    callback_status: str = "idle"
    starvation_events: int = 0
    instrument_version: int = 0
    recent_waveform: tuple[float, ...] = field(default_factory=tuple)


class SnapshotStore:
    """A tiny lock-free publisher/subscriber for immutable snapshots.

    Writes replace the reference; reads return the latest reference. Both
    operations are atomic under the CPython GIL and never block.
    """

    def __init__(self, initial: PerformanceSnapshot | None = None) -> None:
        self._snapshot: PerformanceSnapshot = initial or PerformanceSnapshot()

    def set(self, snapshot: PerformanceSnapshot) -> None:
        """Publish a new snapshot, replacing the previous one."""
        self._snapshot = snapshot

    def get(self) -> PerformanceSnapshot:
        """Return the most recently published snapshot."""
        return self._snapshot

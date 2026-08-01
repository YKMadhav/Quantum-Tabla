"""Generative scheduler: the real-time performance worker.

A single daemon thread that, while running:

1. reads the current playhead from the mixer,
2. asks the rhythm engine for every event due within the look-ahead window,
3. selects a stroke type, renders the DSP waveform, and mixes it into the ring,
4. advances the instrument state on its own cadence,
5. measures audio health and publishes an immutable :class:`PerformanceSnapshot`.

It never blocks the device callback, never blocks on randomness, and keeps
every internal structure bounded (mix ring, scheduled-event log, waveform
window) so a performance can run indefinitely.
"""

from __future__ import annotations

import threading
from typing import Optional

from src.core.randomness.factory import RandomnessStack
from src.core.timing import monotonic
from src.performance.audio import AudioEngine
from src.performance.mixer import StrokeMixer
from src.performance.randomness import PerformanceRandomness
from src.performance.rhythm import RhythmEngine
from src.performance.selector import StrokeSelector
from src.performance.state import PerformanceSnapshot, SnapshotStore
from src.performance.synthesis import render_performance_stroke, stroke_waveform
from src.synthesis.tabla import StrokeType

#: Frames of look-ahead margin below which the mix is considered at risk.
_UNDERRUN_MARGIN_BLOCKS = 4
#: Instrument source entropy is measured from the mapper catalog size.
_CATALOG_ENTROPY = 4.0


class GenerativeScheduler(threading.Thread):
    """Owns the rhythm engine, stroke selector and audio output."""

    def __init__(
        self,
        *,
        stack: RandomnessStack,
        performance_randomness: PerformanceRandomness,
        mixer: StrokeMixer,
        audio: AudioEngine,
        config,
        waveform_max_points: int = 512,
    ) -> None:
        super().__init__(name="qtabla-performance", daemon=True)
        self._stack = stack
        self._performance = performance_randomness
        self._mixer = mixer
        self._audio = audio
        self._cfg = config
        self._fs = float(config.audio.sample_rate)

        self._rhythm = RhythmEngine(
            fs=self._fs,
            chunk=performance_randomness.chunk,
            tempo_start_bpm=config.performance.tempo_start_bpm,
            tempo_min_bpm=config.performance.tempo_min_bpm,
            tempo_max_bpm=config.performance.tempo_max_bpm,
            tempo_step_per_measure=config.performance.tempo_step_per_measure,
            measures_per_phrase=config.performance.measures_per_phrase,
            subdivision_weights=config.performance.subdivision_weights,
            density_low=config.performance.density_low,
            density_high=config.performance.density_high,
            density_drift_per_measure=config.performance.density_drift_per_measure,
            micro_timing_max_s=config.performance.micro_timing_max_s,
            ghost_probability=config.performance.ghost_probability,
            max_consecutive_rests=config.performance.max_consecutive_rests,
            start_sample=config.performance.min_lead_s * self._fs,
        )
        self._selector = StrokeSelector(performance_randomness.chunk)
        self._waveform_points = max(16, int(waveform_max_points))

        self._stop_event = threading.Event()
        self._snapshots = SnapshotStore()
        self._started_at = 0.0
        self._stroke_count = 0
        self._current_stroke: Optional[StrokeType] = None
        self._last_pitch_hz = 0.0
        self._last_waveform_at = 0
        self._last_snapshot_at = 0.0
        self._recent_events: list[str] = []
        self._recent_waveform: tuple[float, ...] = ()
        self._recording = None
        self._running_engine = False
        self._fatal_error: Optional[str] = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Begin the performance: audio first, then the generation loop."""
        self._performance.reset()
        self._selector.reset()
        self._stack.instrument.reset()
        self._stack.scheduler.reset()
        self._started_at = monotonic()
        self._stroke_count = 0
        self._current_stroke = None
        self._recent_events = []
        self._recent_waveform = ()
        self._last_waveform_at = 0
        self._last_snapshot_at = 0.0

        self._mixer.start_recording()
        try:
            self._audio.start()
            self._running_engine = True
            self._stop_event.clear()
            super().start()
            self._publish_snapshot()
        except Exception:
            self._recording = self._mixer.stop_recording()
            raise

    def stop(self) -> None:
        """Stop generation and playback, waiting for the loop to finish."""
        self._stop_event.set()
        if self._running_engine:
            self.join(timeout=3.0)
        self._audio.stop()
        self._recording = self._mixer.stop_recording()
        self._publish_snapshot()

    def terminate_immediately(self) -> None:
        """Stop without waiting (used when the audio device failed to open)."""
        self._stop_event.set()
        self._running_engine = False
        try:
            self._audio.stop()
        except Exception:
            pass
        self._recording = self._mixer.stop_recording()
        self._publish_snapshot()

    # -- reporting -------------------------------------------------------------

    @property
    def snapshots(self) -> SnapshotStore:
        """Snapshot store the UI polls without locking."""
        return self._snapshots

    @property
    def running(self) -> bool:
        return self._running_engine

    def snapshot(self) -> PerformanceSnapshot:
        """Latest published snapshot."""
        return self._snapshots.get()

    @property
    def recording(self):
        """Recorded mono audio from the most recently stopped run, if any."""
        return self._recording

    # -- main loop -------------------------------------------------------------

    def run(self) -> None:
        interval = max(0.005, float(self._cfg.performance.scheduler_sleep_s))
        last_instrument_at = monotonic()
        try:
            while not self._stop_event.wait(interval):
                now = monotonic()
                playhead = self._mixer.playhead
                self._schedule_ahead(playhead)

                if now - last_instrument_at >= self._cfg.performance.instrument_step_interval_s:
                    last_instrument_at = now
                    self._stack.instrument.step(
                        lambda n: self._stack.bit_stream.take(n)
                    )

                self._capture_waveform()
                if now - self._last_snapshot_at >= interval:
                    self._publish_snapshot()
        except Exception as exc:  # noqa: BLE001 - surface as telemetry, keep UI alive
            self._fatal_error = str(exc)
        finally:
            self._running_engine = False
            self._publish_snapshot(final=True)

    def _schedule_ahead(self, playhead: int) -> None:
        """Generate, render and mix every event due within the look-ahead."""
        horizon = playhead + int(round(self._cfg.performance.lookahead_s * self._fs))
        produced = 0
        max_events = 64
        state = self._stack.instrument.state
        while (
            self._rhythm.peek_time() <= horizon
            and produced < max_events
            and not self._stop_event.is_set()
        ):
            decision = self._rhythm.take_next()
            produced += 1
            if decision.is_rest:
                self._record_event("rest")
                continue

            on_beat = decision.time_samples % (
                self._rhythm.samples_per_beat
            ) < self._mixer.block_size
            stroke_type = self._selector.select(
                strength=decision.strength,
                accent=decision.accent,
                on_beat=on_beat,
                phrase_position=self._rhythm.phrase_position,
                phrase_start=self._rhythm.phrase_position == 0.0,
                phrase_end=self._rhythm.phrase_position
                >= 1.0 - 1.0 / self._cfg.performance.measures_per_phrase,
            )
            waveform = render_performance_stroke(
                state,
                stroke_type,
                decision,
                fs=self._fs,
                duration_s=self._cfg.performance.stroke_duration_s,
                seed=self._stroke_count,
            )
            self._mixer.write_stroke(decision.time_samples, waveform)
            self._stroke_count += 1
            self._current_stroke = stroke_type
            self._record_event(stroke_type.name)
            self._last_pitch_hz = self._dominant_pitch(waveform)

    def _dominant_pitch(self, waveform) -> float:
        import numpy as np

        signal = np.asarray(waveform, dtype=np.float64)[:4096]
        if signal.size < 64:
            return 0.0
        windowed = signal * np.hanning(signal.size)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(signal.size, 1.0 / self._fs)
        peak = int(np.argmax(spectrum))
        return float(freqs[peak]) if peak > 0 else 0.0

    def _record_event(self, label: str) -> None:
        self._recent_events.append(label)
        cap = self._cfg.performance.max_scheduled_events
        if len(self._recent_events) > cap:
            del self._recent_events[: len(self._recent_events) - cap]

    def _capture_waveform(self) -> None:
        """Sample the recent mixed output, bounded by the ring window."""
        window_s = 0.4
        count = int(round(window_s * self._fs))
        margin = self._mixer.playhead - self._last_waveform_at
        if margin < self._mixer.block_size:
            return
        self._last_waveform_at = self._mixer.playhead
        recent = self._mixer.recent_audio(count)
        self._recent_waveform = stroke_waveform(recent, self._waveform_points)

    # -- snapshots -------------------------------------------------------------

    def _publish_snapshot(self, final: bool = False) -> None:
        elapsed = monotonic() - self._started_at if self._started_at else 0.0
        stats = self._stack.bit_stream.stats
        entropy = _CATALOG_ENTROPY
        total_bits = getattr(stats, "total_consumed", 0)
        prepared = self._mixer.prepared_seconds
        margin_blocks = prepared * self._fs / max(1, self._mixer.block_size)
        if margin_blocks < _UNDERRUN_MARGIN_BLOCKS:
            underruns = self._audio.underruns + 1
        else:
            underruns = self._audio.underruns

        pitch = self._last_pitch_hz
        snapshot = PerformanceSnapshot(
            running=self._running_engine,
            status="Running" if self._running_engine else "Stopped",
            error=self._fatal_error,
            runtime_s=elapsed,
            stroke_count=self._stroke_count,
            current_stroke=self._current_stroke.value if self._current_stroke else None,
            tempo_bpm=round(self._rhythm.tempo_bpm, 1),
            current_pitch_hz=round(pitch, 1),
            accent=round(self._rhythm.last_accent, 2),
            rhythm_density=round(self._rhythm.density, 2),
            entropy_rate=entropy,
            random_buffer_bits=self._stack.bit_stream.available(),
            total_bits_consumed=total_bits,
            audio_queue_depth=min(
                len(self._recent_events), self._cfg.performance.max_scheduled_events
            ),
            audio_underruns=underruns,
            audio_buffer_fill=round(min(1.0, prepared / self._cfg.performance.lookahead_s), 3),
            callback_status=self._audio_status_text(),
            starvation_events=self._performance.starvation_events,
            instrument_version=self._stack.instrument.version,
            recent_waveform=self._recent_waveform,
        )
        self._snapshots.set(snapshot)

    def _audio_status_text(self) -> str:
        message = getattr(self._audio, "status_message", "")
        if message:
            return message
        if not self._audio.running:
            return "no device"
        return "ok"

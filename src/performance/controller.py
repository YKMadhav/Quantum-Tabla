"""Performance controller: the single entry point Streamlit talks to.

Owns the audio engine, the generative scheduler and the snapshot store. It
guards against duplicate starts, cleans up cleanly on stop, and never lets an
audio-device failure crash the app: the engine is torn down, the error is
surfaced in telemetry, and a later Start is allowed to retry.
"""

from __future__ import annotations

import io
import threading
import wave

import numpy as np

from src.core.config import AppConfig
from src.core.randomness.factory import RandomnessStack
from src.performance.audio import AudioEngine, SilentEngine, SoundDeviceEngine
from src.performance.mixer import StrokeMixer
from src.performance.randomness import PerformanceRandomness
from src.performance.scheduler import GenerativeScheduler
from src.performance.state import PerformanceSnapshot


class PerformanceController:
    """Lifecycle manager for the real-time performance engine."""

    def __init__(
        self,
        config: AppConfig,
        stack: RandomnessStack,
        *,
        backend: str = "sounddevice",
        device: str | None = None,
    ) -> None:
        self._config = config
        self._stack = stack
        self._backend = backend
        self._device = device if backend == "sounddevice" else None
        self._lock = threading.Lock()
        self._scheduler: GenerativeScheduler | None = None
        self._starting = False
        self._last_error: str | None = None
        self._last_recording_wav: bytes | None = None

    # -- lifecycle -------------------------------------------------------------

    @property
    def running(self) -> bool:
        scheduler = self._scheduler
        return scheduler is not None and scheduler.running

    @property
    def last_error(self) -> str | None:
        """Most recent failure message (None once a performance succeeds)."""
        return self._last_error

    @property
    def last_recording_wav(self) -> bytes | None:
        """WAV bytes captured from the most recently stopped performance."""
        return self._last_recording_wav

    def snapshot(self) -> PerformanceSnapshot:
        """Latest telemetry snapshot (safe to call from any thread)."""
        scheduler = self._scheduler
        if scheduler is None:
            snapshot = PerformanceSnapshot(status="Stopped", error=self._last_error)
            return snapshot
        current = scheduler.snapshot()
        if self._last_error and current.error is None:
            from dataclasses import replace

            return replace(current, error=self._last_error)
        return current

    def start(self) -> tuple[bool, str]:
        """Begin an indefinite performance. Returns ``(ok, message)``.

        Safe to call repeatedly: a second Start while running is a no-op that
        reports success. If the audio device cannot be opened, the engine is
        torn down immediately, the error is recorded, and ``ok`` is False.
        """
        with self._lock:
            if self._starting:
                return True, "Performance already starting"
            scheduler = self._scheduler
            if scheduler is not None and scheduler.running:
                return True, "Performance already running"

            self._starting = True
            try:
                scheduler = self._build_scheduler()
                self._scheduler = scheduler
                self._last_error = None
                self._last_recording_wav = None
                try:
                    scheduler.start()
                except Exception as exc:  # noqa: BLE001 - surface to UI, not crash
                    scheduler.terminate_immediately()
                    self._last_error = str(exc)
                    return False, f"Audio engine error: {exc}"
                return True, "Performance running"
            finally:
                self._starting = False

    def stop(self) -> None:
        """Stop the performance cleanly. Safe to call when not running."""
        with self._lock:
            scheduler = self._scheduler
            self._scheduler = None
        if scheduler is not None:
            scheduler.stop()
            self._last_recording_wav = self._encode_wav(scheduler.recording)

    def retry(self) -> tuple[bool, str]:
        """Clear the stored error, then start (used after a device failure)."""
        with self._lock:
            self._last_error = None
        return self.start()

    # -- internals -------------------------------------------------------------

    def _build_scheduler(self) -> GenerativeScheduler:
        cfg = self._config
        mixer = StrokeMixer(
            fs=cfg.audio.sample_rate,
            block_size=cfg.performance.block_size,
            ring_seconds=cfg.performance.ring_seconds,
            output_ceiling=cfg.performance.output_ceiling,
        )
        audio = self._build_audio(mixer)
        performance = PerformanceRandomness(
            self._stack.bit_stream, chunk_size_bits=cfg.randomness.chunk_size_bits
        )
        return GenerativeScheduler(
            stack=self._stack,
            performance_randomness=performance,
            mixer=mixer,
            audio=audio,
            config=cfg,
            waveform_max_points=cfg.performance.waveform_points,
        )

    def _build_audio(self, mixer: StrokeMixer) -> AudioEngine:
        if self._backend == "sounddevice":
            return SoundDeviceEngine(
                mixer,
                device=self._device,
                sample_rate=self._config.audio.sample_rate,
            )
        return SilentEngine(mixer)

    def _encode_wav(self, audio: np.ndarray | None) -> bytes | None:
        """Convert a mono float waveform into 16-bit PCM WAV bytes."""
        if audio is None or audio.size == 0:
            return None
        signal = np.asarray(audio, dtype=np.float64)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        signal = np.clip(signal, -1.0, 1.0)
        pcm = (signal * 32767.0).astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(self._config.audio.sample_rate))
            wav.writeframes(pcm.tobytes())
        return buffer.getvalue()

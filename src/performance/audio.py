"""Audio output backends.

The :class:`AudioEngine` abstraction lets the scheduler drive either real
hardware (via sounddevice) or a silent, wall-clock clock for tests and
headless environments. The callback never synthesises; it only reads the next
pre-mixed block from the :class:`StrokeMixer`.

In the sounddevice backend the callback additionally clears its status flag
implicitly (sounddevice passes ``status`` each block). The silent backend
simulates the same cadence with a timer thread so the playhead advances at
real speed.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.performance.mixer import StrokeMixer


class AudioEngine(ABC):
    """Plays blocks from a mixer to a (possibly virtual) audio device."""

    def __init__(self, mixer: StrokeMixer, device: str | None = None) -> None:
        self._mixer = mixer
        self._device = device
        self._running = False
        self._underruns = 0

    @property
    def mixer(self) -> StrokeMixer:
        return self._mixer

    @property
    def running(self) -> bool:
        return self._running

    @property
    def underruns(self) -> int:
        """Cumulative count of device underrun/overflow conditions."""
        return self._underruns

    @abstractmethod
    def start(self) -> None:  # pragma: no cover - interface
        """Open the device and begin playback."""

    @abstractmethod
    def stop(self) -> None:  # pragma: no cover - interface
        """Stop playback and release the device."""

    def _mix(self, frames: int) -> None:
        """Read one block from the mixer (keeps the playhead moving)."""
        self._mixer.consume_block()


class SoundDeviceEngine(AudioEngine):
    """Real-time output through the ``sounddevice`` backend."""

    def __init__(
        self,
        mixer: StrokeMixer,
        *,
        device: str | None = None,
        sample_rate: float = 44100.0,
    ) -> None:
        super().__init__(mixer, device=device)
        self._sample_rate = float(sample_rate)
        self._stream: object | None = None
        self._status_lock = threading.Lock()
        self._status_message = ""

    @property
    def status_message(self) -> str:
        """Latest non-empty sounddevice callback status message."""
        with self._status_lock:
            return self._status_message

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    def start(self) -> None:
        import sounddevice as sd

        block = self._mixer._block
        device = self._resolve_device(sd)
        self._stream = sd.OutputStream(
            samplerate=int(self._sample_rate),
            blocksize=block,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._callback,
        )
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()
        self._running = False

    def _resolve_device(self, sd: object) -> int | None:
        """Translate a device name into an index (default output if unnamed)."""
        if not self._device:
            default = sd.default.device
            if isinstance(default, (tuple, list)) and len(default) > 1:
                return int(default[1])
            return None
        try:
            return int(self._device)
        except (TypeError, ValueError):
            pass
        try:
            for index, info in enumerate(sd.query_devices()):
                if self._device.lower() in str(info.get("name", "")).lower():
                    return int(index)
        except Exception:
            pass
        return None

    def _callback(self, outdata, frames, time_info, status):  # noqa: ANN001
        block = self._mixer.consume_block()
        mono = block.astype(np.float32)
        if outdata.ndim == 1:
            outdata[:] = mono
        else:
            outdata[:] = mono[:, None]
        if status:
            self._underruns += 1
            message = str(status)
            with self._status_lock:
                self._status_message = message[:200]


class SilentEngine(AudioEngine):
    """Advances the mixer on a wall-clock cadence without any device.

    Used by tests and headless environments. Underruns are measured the same
    way the scheduler measures them for the real backend (via the prepared
    margin), so behaviour stays comparable.
    """

    def __init__(self, mixer: StrokeMixer) -> None:
        super().__init__(mixer)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._tick_loop,
                name="qtabla-silent-audio",
                daemon=True,
            )
            self._thread.start()
        self._running = True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._running = False

    def _tick_loop(self) -> None:
        interval = self._mixer._block / self._mixer._fs
        while not self._stop_event.wait(interval):
            self._mixer.consume_block()

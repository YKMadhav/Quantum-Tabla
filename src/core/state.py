"""Session-scoped application state.

Wraps Streamlit's ``session_state`` behind a small, stable API so the rest of
the application never touches raw session keys. Initialisation is idempotent,
which means repeated reruns never reset or duplicate the state.

The state owns the whole randomness stack (provider, bit stream, decoder,
scheduler, instrument manager) for the session and manages its lifecycle
together with the playback loop. The entropy provider (classical or quantum)
is chosen per session; switching it while stopped rebuilds the stack cleanly.
"""

from __future__ import annotations

import os
from dataclasses import replace

import streamlit as st

from src.core.config import AppConfig
from src.core.instrument.state import InstrumentState
from src.core.randomness.base import ProviderStats, ProviderStatus
from src.core.randomness.factory import (
    PROVIDER_REGISTRY,
    RandomnessStack,
    build_randomness_stack,
)
from src.core.randomness.stream import BufferStats
from src.core.timing import RuntimeMetrics
from src.performance.controller import PerformanceController
from src.performance.state import PerformanceSnapshot

_RUNNING_KEY = "qtabla.running"
_METRICS_KEY = "qtabla.metrics"
_STACK_KEY = "qtabla.randomness_stack"
_PERFORMANCE_KEY = "qtabla.performance"
_CONFIG_KEY = "qtabla.config"
_BACKEND_KEY = "qtabla.audio_backend"
_PROVIDER_KEY = "qtabla.provider_name"

#: Set QTABALA_AUDIO_BACKEND=silent to run the engine without a real device
#: (used by headless tests and CI; default is the real sounddevice backend).
_AUDIO_BACKEND_ENV = "QTABALA_AUDIO_BACKEND"


class ApplicationState:
    """Facade over the persistent, per-session application state.

    Create instances exclusively through :meth:`get`, which guarantees the
    session keys are initialised exactly once.
    """

    def __init__(self) -> None:
        self._validate_initialised()

    # -- construction ---------------------------------------------------------

    @classmethod
    def initialize(cls, config: AppConfig) -> None:
        """Create the session keys once. Idempotent and safe to call often."""
        if _CONFIG_KEY not in st.session_state:
            st.session_state[_CONFIG_KEY] = config
        if _BACKEND_KEY not in st.session_state:
            st.session_state[_BACKEND_KEY] = os.environ.get(
                _AUDIO_BACKEND_ENV, "sounddevice"
            )
        if _PROVIDER_KEY not in st.session_state:
            st.session_state[_PROVIDER_KEY] = config.randomness.provider_name
        if _RUNNING_KEY not in st.session_state:
            st.session_state[_RUNNING_KEY] = False
        if _METRICS_KEY not in st.session_state:
            st.session_state[_METRICS_KEY] = RuntimeMetrics()
        if _STACK_KEY not in st.session_state:
            st.session_state[_STACK_KEY] = cls._build_stack(st.session_state[_CONFIG_KEY])
        if _PERFORMANCE_KEY not in st.session_state:
            st.session_state[_PERFORMANCE_KEY] = cls._build_controller(
                st.session_state[_CONFIG_KEY], st.session_state[_STACK_KEY]
            )

    @staticmethod
    def _build_stack(config: AppConfig) -> RandomnessStack:
        return build_randomness_stack(config)

    @classmethod
    def _build_controller(
        cls, config: AppConfig, stack: RandomnessStack
    ) -> PerformanceController:
        return PerformanceController(
            config, stack, backend=st.session_state[_BACKEND_KEY]
        )

    @classmethod
    def get(cls, config: AppConfig) -> ApplicationState:
        """Return a new handle over the initialised session state."""
        cls.initialize(config)
        return cls()

    @staticmethod
    def _validate_initialised() -> None:
        if _RUNNING_KEY not in st.session_state:
            raise RuntimeError(
                "ApplicationState.initialize() must run before creating an "
                "ApplicationState instance."
            )

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> tuple[bool, str]:
        """Start the real-time performance engine.

        Returns ``(ok, message)``. The engine runs on its own thread and keeps
        playing until :meth:`stop` is called. A failure to open the audio
        device returns ``(False, message)`` without crashing the app.
        """
        stack = self.stack
        stack.bit_stream.start()
        stack.instrument.reset()
        ok, message = self.performance.start()
        if ok:
            self.metrics.start()
            st.session_state[_RUNNING_KEY] = True
        else:
            stack.bit_stream.stop()
            st.session_state[_RUNNING_KEY] = False
        return ok, message

    def stop(self) -> None:
        """Stop the performance engine and the entropy stream."""
        self.performance.stop()
        self.stack.bit_stream.stop()
        st.session_state[_RUNNING_KEY] = False

    def set_provider(self, name: str) -> tuple[bool, str]:
        """Switch the session's entropy provider (only while stopped).

        Rebuilds the randomness stack and the performance controller around
        the chosen provider. Refuses to switch mid-performance so an ongoing
        stream is never torn down under the audio engine.

        Returns ``(ok, message)``.
        """
        if self.running:
            return False, "Stop playback before switching randomness mode."
        if name not in PROVIDER_REGISTRY:
            return False, f"Unknown randomness provider: {name!r}"
        if name == st.session_state.get(_PROVIDER_KEY):
            return True, f"Already using the {name!r} provider."

        config = replace(
            st.session_state[_CONFIG_KEY],
            randomness=replace(
                st.session_state[_CONFIG_KEY].randomness, provider_name=name
            ),
        )
        st.session_state[_CONFIG_KEY] = config
        stack = self._build_stack(config)
        st.session_state[_STACK_KEY] = stack
        st.session_state[_PERFORMANCE_KEY] = self._build_controller(config, stack)
        st.session_state[_PROVIDER_KEY] = name
        return True, f"Switched to the {name!r} provider."

    def tick(self) -> None:
        """Record one completed live update frame."""
        self.metrics.tick()

    # -- read-only accessors --------------------------------------------------

    @property
    def running(self) -> bool:
        """True while the live performance engine is active."""
        return self.performance.running

    @property
    def performance(self) -> PerformanceController:
        """The real-time performance engine for this session."""
        return st.session_state[_PERFORMANCE_KEY]

    @property
    def performance_snapshot(self) -> PerformanceSnapshot:
        """Latest telemetry snapshot from the performance engine."""
        return self.performance.snapshot()

    @property
    def status(self) -> str:
        """Human-readable playback status."""
        return "Running" if self.running else "Stopped"

    @property
    def metrics(self) -> RuntimeMetrics:
        """The live timing metrics for the current session."""
        return st.session_state[_METRICS_KEY]

    @property
    def stack(self) -> RandomnessStack:
        """The fully wired randomness subsystem for this session."""
        return st.session_state[_STACK_KEY]

    @property
    def instrument_state(self) -> InstrumentState:
        """Latest instrument snapshot (advanced each frame while running)."""
        return self.stack.instrument.state

    @property
    def provider_status(self) -> ProviderStatus:
        """Status of the active randomness provider."""
        return self.stack.bit_stream.status()

    @property
    def provider_name(self) -> str:
        """Name of the provider the session is currently using."""
        return st.session_state[_PROVIDER_KEY]

    @property
    def provider_stats(self) -> ProviderStats:
        """Lifetime generation counters of the active provider."""
        return self.stack.provider.stats()

    @property
    def buffer_available(self) -> int:
        """Random bits currently available in the reservoir."""
        return self.stack.bit_stream.available()

    @property
    def buffer_stats(self) -> BufferStats:
        """Lifetime stats of the bit stream manager."""
        return self.stack.bit_stream.stats

    @property
    def elapsed_seconds(self) -> float:
        """Runtime elapsed since the last Start."""
        return self.metrics.elapsed_seconds

    @property
    def total_ticks(self) -> int:
        """Total live update frames completed since the last Start."""
        return self.metrics.total_ticks

    @property
    def rolling_fps(self) -> float:
        """Instantaneous update rate over the rolling window."""
        return self.metrics.rolling_fps

    @property
    def average_fps(self) -> float:
        """Average update rate over the whole runtime."""
        return self.metrics.average_fps

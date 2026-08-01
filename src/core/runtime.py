"""Live update loop that keeps the dashboard refreshing while running.

The update loop follows Streamlit's rerun-based model: each script execution
is one update frame. While the app is running this module records the tick,
paces the run with a short sleep and schedules the next rerun. As soon as the
state is stopped it does nothing, so idle CPU usage drops to zero.
"""

from __future__ import annotations

import time

import streamlit as st

from src.core.config import AppConfig
from src.core.state import ApplicationState


class LiveUpdateLoop:
    """Paces a live dashboard and requests continuous reruns.

    Call :meth:`step` at the very end of every script execution. The loop is
    self-terminating: it only schedules further work while the application is
    in the running state.
    """

    def __init__(self, config: AppConfig, state: ApplicationState) -> None:
        self._config: AppConfig = config
        self._state: ApplicationState = state

    def step(self) -> None:
        """Advance one update frame, or do nothing when the app is stopped."""
        if not self._state.running:
            return
        self._state.tick()
        time.sleep(self._config.update_interval_s)
        st.rerun()

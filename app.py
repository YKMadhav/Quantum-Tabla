"""Quantum Tabla — application entry point.

Orchestrates the Streamlit dashboard: applies page configuration, initialises
session state, advances the instrument one frame while running, renders the
UI and drives the live update loop. All details are delegated to dedicated
modules so this file stays thin.

Run with:

    streamlit run app.py
"""

from __future__ import annotations

from src.core.config import AppConfig
from src.core.runtime import LiveUpdateLoop
from src.core.state import ApplicationState
from src.dashboard import layout, styles


def main() -> None:
    """Configure, render and drive the Quantum Tabla dashboard."""
    config = AppConfig()
    styles.configure_page(config)
    styles.apply()

    state = ApplicationState.get(config)

    layout.render_header(config)
    layout.render_status_cards(state, config)
    layout.render_live_metrics(state, config)
    layout.render_performance_status(state, config)
    layout.render_waveform(state, config)
    layout.render_randomness_source(state, config)
    layout.render_parameter_status(state, config)
    layout.render_debug(state, config)
    layout.render_controls(state)

    LiveUpdateLoop(config, state).step()


if __name__ == "__main__":
    main()

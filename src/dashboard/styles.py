"""Page configuration and visual styling for the dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.core.config import AppConfig

_STYLESHEET_PATH = Path(__file__).resolve().parents[2] / "assets" / "style.css"


def configure_page(config: AppConfig) -> None:
    """Set the Streamlit page options (must be the first Streamlit call)."""
    st.set_page_config(
        page_title=config.app_title,
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def apply() -> None:
    """Inject the dashboard stylesheet into the page."""
    css = _STYLESHEET_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

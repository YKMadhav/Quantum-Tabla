"""End-to-end dashboard tests via Streamlit's AppTest framework."""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QTABALA_AUDIO_BACKEND", "silent")

streamlit = pytest.importorskip("streamlit")

try:
    from streamlit.testing.v1 import AppTest

    _HAS_APPTEST = True
except Exception:  # pragma: no cover - depends on Streamlit version
    _HAS_APPTEST = False

pytestmark = pytest.mark.skipif(
    not _HAS_APPTEST, reason="Streamlit AppTest framework not available"
)


@pytest.fixture()
def app():
    """A fresh, runnable AppTest of the dashboard with a stubbed rerun."""
    if hasattr(streamlit, "rerun"):
        streamlit.rerun = lambda *args, **kwargs: None  # type: ignore[assignment]
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    return at


def test_dashboard_renders_initial_state(app) -> None:
    assert not app.exception
    assert len(app.button) >= 2
    assert app.metric[0].value == "Stopped"


def test_start_then_stop_lifecycle(app) -> None:
    start_button = app.button[0]
    stop_button = app.button[1]
    assert start_button.disabled is False
    assert stop_button.disabled is True

    start_button.click().run()
    assert not app.exception
    # The performance engine starts on its own thread; give it time to play.
    deadline = time.monotonic() + 10.0
    running = False
    while time.monotonic() < deadline:
        app.run()
        if app.metric[0].value == "Running":
            running = True
            break
        time.sleep(0.2)
    assert running, "dashboard never entered Running state"

    app.button[1].click().run()
    assert app.metric[0].value == "Stopped"
    assert not app.exception


def test_duplicate_start_is_guarded(app) -> None:
    app.button[0].click().run()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        app.run()
        if app.metric[0].value == "Running":
            break
        time.sleep(0.2)
    # Start is disabled while running: clicking it must be a no-op.
    start_button = app.button[0]
    assert start_button.disabled is True
    app.button[1].click().run()
    assert app.metric[0].value == "Stopped"


def test_switch_randomness_mode_to_quantum(app) -> None:
    # Classical is the default: the circuit description is not rendered.
    assert len(app.radio) == 1
    assert not any("Measure" in code.value for code in app.code)

    app.radio[0].set_value("quantum").run()
    assert not app.exception

    # The quantum provider is now live: its circuit is displayed and the
    # dashboard still allows a Start/Stop lifecycle on the rebuilt stack.
    assert any("Measure" in code.value for code in app.code)
    app.button[0].click().run()
    deadline = time.monotonic() + 10.0
    running = False
    while time.monotonic() < deadline:
        app.run()
        if app.metric[0].value == "Running":
            running = True
            break
        time.sleep(0.2)
    assert running, "dashboard never entered Running state with quantum mode"
    app.button[1].click().run()
    assert app.metric[0].value == "Stopped"

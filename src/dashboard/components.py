"""Reusable dashboard UI primitives.

Small, framework-specific building blocks (metric cards, placeholder panels,
parameter bars, debug panel, controls) that ``layout`` composes into the
page. Keeping them here makes them independently testable and reusable.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.core.config import AppConfig
from src.core.instrument.state import InstrumentState
from src.core.randomness.mapper import ParameterCatalog
from src.core.state import ApplicationState
from src.synthesis.tabla import (
    StrokeType,
    analyze_waveform,
    render_stroke,
    waveform_envelope,
)
from src.utils import bits as bit_utils

#: Session key backing the randomness-mode radio widget.
_RANDOMNESS_MODE_KEY = "qtabla.provider_choice"
_RANDOMNESS_ERROR_KEY = "qtabla.randomness_error"


def metric_card(
    label: str, value: str, help_text: str | None = None
) -> None:
    """Render a single labelled metric value as a native Streamlit metric."""
    st.metric(label=label, value=value, help=help_text)


def randomness_metric_offset() -> None:
    """Align randomness metrics with the radio selector card."""
    st.markdown(
        '<div class="qtabla-randomness-card-offset"></div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, note: str | None = None) -> None:
    """Render a small uppercase section title with an optional muted note."""
    html = f'<div class="qtabla-section-title">{title}</div>'
    if note:
        html += f'<div class="qtabla-section-note">{note}</div>'
    st.markdown(html, unsafe_allow_html=True)


def placeholder_panel(title: str, note: str) -> None:
    """Render a dashed placeholder panel indicating a pending subsystem."""
    st.markdown(
        '<div class="qtabla-placeholder">'
        f'<div class="qtabla-placeholder-title">{title}</div>'
        f'<div class="qtabla-placeholder-note">{note}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def waveform_panel(
    waveform: np.ndarray,
    fs: float,
    max_points: int,
    caption: str | None = None,
) -> None:
    """Render a synthesised stroke as a lightweight envelope line chart."""
    envelope = waveform_envelope(waveform, max_points)
    analysis = analyze_waveform(waveform, fs)
    st.line_chart(envelope, height=160)
    st.caption(
        caption
        or (
            f"Peak {analysis['peak']:.3f} · "
            f"dominant {analysis['dominant_hz']:.0f} Hz · "
            f"{len(waveform) / fs:.2f}s"
        )
    )


def envelope_panel(
    envelope: tuple[float, ...],
    caption: str | None = None,
) -> None:
    """Render an already-computed min/max envelope as a line chart.

    Used for the rolling visualisation of the *actual* recent audio output,
    whose envelope is produced by the performance engine (never re-computed on
    the UI thread).
    """
    st.line_chart(np.asarray(envelope, dtype=np.float64), height=160)
    st.caption(caption or "Rolling audio output")


def parameter_bars(
    state: InstrumentState,
    catalog: ParameterCatalog,
    columns: int = 2,
) -> None:
    """Render live progress bars, one per instrument parameter."""
    cols = st.columns(columns)
    initialised = state.version > 0
    for index, spec in enumerate(catalog.specs):
        value = state.value(spec.name) if initialised else 0.0
        label = f"{spec.name}  {value:.2f}"
        with cols[index % columns]:
            st.progress(value, text=label)


def _apply_randomness_mode(state: ApplicationState) -> None:
    """Callback: rebuild the session stack around the newly chosen provider."""
    ok, message = state.set_provider(st.session_state[_RANDOMNESS_MODE_KEY])
    if ok:
        st.session_state.pop(_RANDOMNESS_ERROR_KEY, None)
    else:
        st.session_state[_RANDOMNESS_ERROR_KEY] = message


def randomness_mode_selector(
    state: ApplicationState, config: AppConfig
) -> None:
    """Render the Classical/Quantum entropy-source selector.

    The selector is disabled while a performance is running so a live stream
    is never torn down under the audio engine; switch by Stop → select →
    Start.
    """
    labels = {
        "classical": config.mode_classical_label,
        "quantum": config.mode_quantum_label,
    }
    options = list(labels)
    st.radio(
        config.randomness_mode_label,
        options,
        index=options.index(state.provider_name),
        key=_RANDOMNESS_MODE_KEY,
        horizontal=True,
        disabled=state.running,
        help=config.randomness_mode_help,
        format_func=lambda name: labels[name],
        on_change=lambda: _apply_randomness_mode(state),
    )
    error = st.session_state.pop(_RANDOMNESS_ERROR_KEY, None)
    if error:
        st.warning(error)


def zero_one_stats(state: ApplicationState, config: AppConfig) -> None:
    """Render lifetime 0/1 counts and percentages from the entropy source."""
    stats = state.provider_stats
    total = stats.total_generated
    if total <= 0:
        st.caption("0/1 distribution — no bits generated yet.")
        return
    zero_pct = 100.0 * stats.zero_count / total
    one_pct = 100.0 * stats.one_count / total
    col_zero, col_one = st.columns(2)
    with col_zero:
        st.metric(
            "0 bits",
            f"{stats.zero_count:,}  ({zero_pct:.1f}%)",
            help=config.zero_ones_help,
        )
    with col_one:
        st.metric(
            "1 bits",
            f"{stats.one_count:,}  ({one_pct:.1f}%)",
            help=config.zero_ones_help,
        )


def audition_testbench(state: ApplicationState, config: AppConfig) -> None:
    """Render the collapsed one-shot DSP audition bench (development tool).

    Renders any of the four stroke types with an in-browser player and
    waveform. It is never part of live playback, so it stays tucked away in
    the collapsed debug section.
    """
    with st.expander("DSP test bench — audition strokes", expanded=False):
        audio = config.audio
        stroke_label = st.selectbox(
            "Stroke",
            [stroke.value for stroke in StrokeType],
            index=0,
        )
        if state.instrument_state.version == 0:
            st.info("Start playback to fill the instrument with parameters.")
            return
        stroke = StrokeType(stroke_label)
        waveform = render_stroke(
            state.instrument_state,
            stroke,
            fs=float(audio.sample_rate),
            duration_s=audio.audition_duration_s,
            seed=7,
        )
        st.audio(waveform, sample_rate=audio.sample_rate)
        waveform_panel(
            waveform,
            fs=float(audio.sample_rate),
            max_points=audio.waveform_max_points,
            caption=f"{stroke.value} · rendered from state v{state.instrument_state.version}",
        )


def debug_panel(state: ApplicationState, catalog: ParameterCatalog) -> None:
    """Render the collapsed debugging section for the randomness internals."""
    chunk_size = state.stack.decoder.chunk_size_bits
    max_value = (1 << chunk_size) - 1
    stats = state.buffer_stats
    preview = state.stack.bit_stream.preview(64)

    with st.expander("Debug — randomness internals", expanded=False):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Provider**")
            st.code(
                f"name      : {state.provider_status.name}\n"
                f"online    : {state.provider_status.online}\n"
                f"quantum   : {state.provider_status.is_quantum}\n"
                f"elapsed   : {stats.elapsed_seconds:.1f}s\n"
                f"generated : {stats.total_generated} bit\n"
                f"consumed  : {stats.total_consumed} bit\n"
                f"refills   : {stats.refill_count}"
            )

        with col_right:
            st.markdown("**Bit buffer (first 64)**")
            st.code(preview if preview else "(empty)")
            st.markdown("**Last consumed chunk**")
            st.code(stats.last_take if stats.last_take else "(none)")
            st.markdown("**Extraction example**")
            example_bits = preview[:chunk_size]
            if len(example_bits) == chunk_size and bit_utils.validate_bitstring(
                example_bits
            ):
                value = bit_utils.bits_to_int(example_bits)
                normalised = bit_utils.normalize(value, max_value)
                st.code(f"{example_bits} -> {value} -> {normalised:.3f}")
            else:
                st.code("(insufficient bits)")

        with st.container(border=True):
            names = ", ".join(spec.name for spec in catalog.specs)
            st.markdown(f"**Parameter slots ({len(catalog.specs)}):** {names}")


def control_buttons(state: ApplicationState) -> None:
    """Render the Start/Stop controls and apply their lifecycle effects.

    Start is disabled while running and Stop is disabled while stopped, so
    only one state transition is possible per click. State transitions happen
    in ``on_click`` callbacks, which Streamlit runs *before* the next rerun,
    so the dashboard always reflects the new state on the very next render.

    A failed audio-device open does not crash the app: the error is shown and
    the Start button becomes available again for a retry.
    """
    start_col, stop_col = st.columns(2)
    with start_col:
        st.button(
            "Start",
            type="primary",
            disabled=state.running,
            on_click=state.start,
        )
    with stop_col:
        st.button(
            "Stop",
            disabled=not state.running,
            on_click=state.stop,
        )
    if state.running:
        st.caption("Performing — audio runs until Stop is pressed.")
    st.caption(
        "Hearing tiny pops or noise in live playback? Stop and download the "
        "last performance for the cleanest captured random tabla strokes."
    )
    recording = state.performance.last_recording_wav
    if recording and not state.running:
        st.download_button(
            "Download last performance",
            data=recording,
            file_name="qtabla-performance.wav",
            mime="audio/wav",
            use_container_width=True,
        )
    error = state.performance.last_error
    if error:
        st.error(f"Audio engine error: {error}. Press Start to retry.")

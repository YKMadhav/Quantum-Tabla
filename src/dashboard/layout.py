"""Assembles the single-page dashboard layout.

This module owns the high-level arrangement of the page (header, status rows,
live metrics, waveform, parameters, debug and controls). Individual visual
primitives live in ``components`` so they can be reused independently.
"""

from __future__ import annotations

import streamlit as st

from src.core.config import AppConfig
from src.core.state import ApplicationState
from src.dashboard import components
from src.utils import formatting


def render_header(config: AppConfig) -> None:
    """Render the project title and subtitle."""
    st.markdown(
        f'<div class="qtabla-title">{config.app_title}</div>'
        f'<div class="qtabla-subtitle">{config.app_subtitle}</div>',
        unsafe_allow_html=True,
    )


def render_status_cards(state: ApplicationState, config: AppConfig) -> None:
    """Render the four high-level status cards in a single row."""
    snapshot = state.performance_snapshot
    instrument_status = (
        config.instrument_status_active
        if snapshot.running
        else config.instrument_status_idle
    )
    provider = state.provider_status
    quantum_status = (
        config.randomness_source_connected
        if provider.online
        else config.randomness_source_disconnected
    )
    quantum_help = (
        f"{provider.name}: {provider.description}" if provider.online else None
    )

    col_playback, col_system, col_instrument, col_quantum = st.columns(4)
    with col_playback:
        components.metric_card(
            "Playback status", snapshot.status, config.playback_status_help
        )
    with col_system:
        components.metric_card(
            "System status", config.system_status, config.system_status_help
        )
    with col_instrument:
        components.metric_card(
            "Instrument status",
            instrument_status,
            f"{config.instrument_status_help} State v{state.instrument_state.version}.",
        )
    with col_quantum:
        components.metric_card(
            "Randomness source", quantum_status, quantum_help
        )


def render_live_metrics(state: ApplicationState, config: AppConfig) -> None:
    """Render the performance row: runtime, strokes played and tempo.

    Entropy readouts (rate, reservoir, generated/consumed) live in the
    randomness-source panel so each metric appears exactly once.
    """
    snapshot = state.performance_snapshot
    col_runtime, col_strokes, col_tempo = st.columns(3)
    with col_runtime:
        components.metric_card(
            "Performance runtime",
            formatting.format_duration(snapshot.runtime_s),
            config.runtime_help,
        )
    with col_strokes:
        components.metric_card(
            "Strokes played", f"{snapshot.stroke_count:,}", config.update_count_help
        )
    with col_tempo:
        components.metric_card(
            "Tempo",
            f"{snapshot.tempo_bpm:.1f} BPM" if snapshot.tempo_bpm else "—",
            config.performance_tempo_help,
        )


def render_performance_status(state: ApplicationState, config: AppConfig) -> None:
    """Render the live performance monitor (current stroke + audio health)."""
    snapshot = state.performance_snapshot
    (
        col_stroke,
        col_pitch,
        col_accent,
        col_density,
        col_health,
        col_starvation,
    ) = st.columns(6)
    with col_stroke:
        components.metric_card(
            config.current_stroke_label,
            snapshot.current_stroke or config.no_stroke_yet,
            config.current_stroke_help,
        )
    with col_pitch:
        components.metric_card(
            "Pitch",
            f"{snapshot.current_pitch_hz:.0f} Hz" if snapshot.current_pitch_hz else "—",
            config.current_pitch_help,
        )
    with col_accent:
        components.metric_card(
            "Accent", f"{snapshot.accent:.2f}", config.accent_help
        )
    with col_density:
        components.metric_card(
            "Rhythm density", f"{snapshot.rhythm_density:.2f}", config.density_help
        )
    with col_health:
        components.metric_card(
            config.audio_health_label,
            snapshot.callback_status,
            f"{config.audio_health_help} Underruns: {snapshot.audio_underruns} "
            f"· buffer fill {snapshot.audio_buffer_fill:.0%}.",
        )
    with col_starvation:
        components.metric_card(
            "Starvation", f"{snapshot.starvation_events}", config.starvation_help
        )


def render_randomness_source(state: ApplicationState, config: AppConfig) -> None:
    """Render the entropy-source panel: mode selector, provider stats,
    circuit description and 0/1 distribution."""
    components.section_header("Randomness source", None)
    status = state.provider_status
    provider_stats = state.provider_stats
    buffer_stats = state.buffer_stats

    (
        col_mode,
        col_status,
        col_rate,
        col_buffer,
        col_generated,
        col_consumed,
    ) = st.columns(6)
    with col_mode:
        components.randomness_mode_selector(state, config)
    with col_status:
        components.randomness_metric_offset()
        components.metric_card(
            "Source status",
            config.randomness_source_connected
            if status.online
            else config.randomness_source_disconnected,
            config.quantum_source_status_help,
        )
    with col_rate:
        components.randomness_metric_offset()
        components.metric_card(
            "Entropy rate",
            formatting.format_bits_per_second(buffer_stats.bits_per_second),
            config.entropy_rate_help,
        )
    with col_buffer:
        components.randomness_metric_offset()
        components.metric_card(
            "Random buffer",
            formatting.format_bits(state.buffer_available),
            config.buffer_size_help,
        )
    with col_generated:
        components.randomness_metric_offset()
        components.metric_card(
            config.generated_label,
            formatting.format_bits(provider_stats.total_generated),
            "Bits produced by the entropy source since it was selected.",
        )
    with col_consumed:
        components.randomness_metric_offset()
        components.metric_card(
            "Bits consumed",
            formatting.format_bits(buffer_stats.total_consumed),
            config.bits_consumed_help,
        )

    if status.circuit:
        st.code(status.circuit)
    components.zero_one_stats(state, config)


def render_waveform_placeholder(config: AppConfig) -> None:
    """Render the placeholder panel shown before playback produces audio."""
    components.placeholder_panel(
        config.waveform_label,
        "Start playback to see the live audio output.",
    )


def render_waveform(state: ApplicationState, config: AppConfig) -> None:
    """Render a rolling visualisation of the actual recent audio output."""
    components.section_header(
        config.waveform_label,
        config.waveform_note,
    )
    snapshot = state.performance_snapshot
    if not snapshot.running or not snapshot.recent_waveform:
        render_waveform_placeholder(config)
        return
    components.envelope_panel(
        snapshot.recent_waveform,
        caption=f"Live output · stroke {snapshot.stroke_count:,} "
        f"· tempo {snapshot.tempo_bpm:.1f} BPM",
    )


def render_audition(state: ApplicationState, config: AppConfig) -> None:
    """Render the one-shot DSP audition test bench (not the live engine)."""
    components.audition_testbench(state, config)


def render_parameter_status(state: ApplicationState, config: AppConfig) -> None:
    """Render live instrument parameter bars for the current state."""
    catalog = state.stack.mapper.catalog
    note = (
        config.parameter_status_note
        + f"  ·  state v{state.instrument_state.version}"
    )
    components.section_header("Synthesis parameters", note)
    components.parameter_bars(state.instrument_state, catalog)


def render_debug(state: ApplicationState, config: AppConfig) -> None:
    """Render the collapsed debugging sections (internals + DSP test bench)."""
    catalog = state.stack.mapper.catalog
    components.debug_panel(state, catalog)
    components.audition_testbench(state, config)


def render_controls(state: ApplicationState) -> None:
    """Render the Start/Stop controls."""
    components.control_buttons(state)

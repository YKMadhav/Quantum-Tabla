"""Static application configuration.

Every tunable constant the application needs lives here so that layout,
timing, randomness and update-tier settings are defined in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TierBehaviour:
    """How one update tier refreshes and tracks its parameter targets.

    Attributes:
        refresh_every: refresh a new random target every N update frames.
        smoothing: per-frame interpolation factor toward the current target
            (0 = frozen, 1 = instant snap).
    """

    refresh_every: int
    smoothing: float


@dataclass(frozen=True)
class AudioConfig:
    """Tuning for the procedural DSP synthesis engine."""

    #: Samples per second for every rendered waveform.
    sample_rate: int = 44100
    #: Length of one rendered stroke for the live waveform preview.
    stroke_duration_s: float = 0.8
    #: Length of one rendered stroke in the audition test bench.
    audition_duration_s: float = 0.8
    #: Maximum points used when down-sampling a waveform for display.
    waveform_max_points: int = 512
    #: Ceiling applied by the soft limiter; output stays within ``+/-ceiling``.
    soft_ceiling: float = 0.95


@dataclass(frozen=True)
class PerformanceConfig:
    """Tuning for the real-time performance engine.

    The performance engine runs on its own thread and delivers audio through a
    device callback; Streamlit only observes it. All of these values are
    timing/buffer knobs, not musical content.
    """

    # --- Audio engine --------------------------------------------------------
    #: Samples per callback block delivered to the audio device.
    block_size: int = 256
    #: Size of the overlapping-stroke mix ring, in seconds of audio.
    ring_seconds: float = 2.5
    #: Ceiling applied when soft-limiting the mixed output block.
    output_ceiling: float = 0.95
    #: Named audio output device (None = system default).
    audio_device: str | None = None

    # --- Scheduling ----------------------------------------------------------
    #: How far ahead of the playhead the scheduler keeps events, in seconds.
    lookahead_s: float = 0.25
    #: Minimum lead time before an event may play (keeps the callback safe).
    min_lead_s: float = 0.05
    #: Sleep granularity of the scheduler loop, in seconds.
    scheduler_sleep_s: float = 0.02
    #: Cadence of instrument parameter evolution, in seconds.
    instrument_step_interval_s: float = 0.1
    #: Duration of each rendered stroke waveform, in seconds.
    stroke_duration_s: float = 0.8
    #: Cap on telemetry entries held for recently scheduled events.
    max_scheduled_events: int = 32
    #: Points kept in the rolling waveform visualisation buffer.
    waveform_points: int = 512

    # --- Rhythm engine -------------------------------------------------------
    #: Starting tempo, in beats per minute.
    tempo_start_bpm: float = 86.0
    #: Tempo bounds, in beats per minute.
    tempo_min_bpm: float = 70.0
    tempo_max_bpm: float = 160.0
    #: Maximum relative tempo change applied per measure.
    tempo_step_per_measure: float = 0.04
    #: Beats per phrase (in 4/4, a measure is four beats).
    measures_per_phrase: int = 4
    #: Relative weights for quarter / eighth / sixteenth subdivision choices.
    subdivision_weights: tuple[float, float, float] = (0.32, 0.42, 0.26)
    #: Lower and upper bounds of the rhythmic density random walk.
    density_low: float = 0.36
    density_high: float = 0.82
    #: Maximum density drift applied per measure.
    density_drift_per_measure: float = 0.08
    #: Maximum micro-timing jitter applied to a hit, in seconds.
    micro_timing_max_s: float = 0.015
    #: Base probability of a ghost (light, quiet) stroke at weak positions.
    ghost_probability: float = 0.08
    #: Maximum consecutive rests before a hit is forced.
    max_consecutive_rests: int = 2


@dataclass(frozen=True)
class QuantumConfig:
    """Tuning for the Qiskit Aer quantum-circuit provider.

    The quantum provider prepares *num_qubits* independent qubits in an equal
    superposition, measures each one, and reads one bit per measurement. Aer
    is a classical *simulator* of those circuits; it never claims hardware
    results. Bits are produced in batches of *batch_shots* circuit shots to
    amortise the simulator's per-job overhead.
    """

    #: Number of independent qubits measured per circuit (one bit each).
    num_qubits: int = 4
    #: Circuit shots per simulator job (each shot yields ``num_qubits`` bits).
    batch_shots: int = 4096
    #: Reserve at which the provider runs a fresh job before running dry.
    low_water_bits: int = 4096
    #: Upper bound on bits kept in the provider's internal reservoir.
    max_buffer_bits: int = 1 << 18


@dataclass(frozen=True)
class RandomnessConfig:
    """Tuning for the randomness provider and parameter decoder."""

    #: Identifies the randomness provider to build ("classical" or "quantum").
    provider_name: str = "classical"
    #: Optional seed for the classical provider (None = non-deterministic).
    provider_seed: int | None = None
    #: Bits kept in the provider's internal buffer.
    provider_buffer_bits: int = 4096
    #: Settings for the Qiskit Aer quantum-circuit provider.
    quantum: QuantumConfig = QuantumConfig()

    #: Bits grouped into one chunk and normalised to a single parameter value.
    chunk_size_bits: int = 10
    #: Reservoir size the bit stream manager keeps refilled to.
    buffer_target_bits: int = 8192
    #: Maximum bits fetched from the provider in a single refill.
    refill_batch_bits: int = 2048
    #: Seconds between refill loop checks.
    refill_interval_s: float = 0.02

    #: Update tier behaviours (slow / medium / fast).
    slow: TierBehaviour = TierBehaviour(refresh_every=48, smoothing=0.03)
    medium: TierBehaviour = TierBehaviour(refresh_every=12, smoothing=0.10)
    fast: TierBehaviour = TierBehaviour(refresh_every=3, smoothing=0.30)


@dataclass(frozen=True)
class AppConfig:
    """Immutable, centralised configuration for the Quantum Tabla dashboard."""

    # --- Identity ------------------------------------------------------------
    app_title: str = "Quantum Tabla"
    app_subtitle: str = (
        "A Real-Time Procedural Tabla Synthesizer Driven by Quantum Randomness"
    )

    # --- Live update loop ----------------------------------------------------
    #: Target delay between update frames while the app is running, in seconds.
    update_interval_s: float = 0.1
    #: Number of recent update frames used to compute the rolling frame rate.
    metric_window_size: int = 20

    # --- Randomness subsystem ------------------------------------------------
    randomness: RandomnessConfig = RandomnessConfig()

    # --- DSP synthesis engine ------------------------------------------------
    audio: AudioConfig = AudioConfig()

    # --- Real-time performance engine ----------------------------------------
    performance: PerformanceConfig = PerformanceConfig()

    # --- Status card copy ----------------------------------------------------
    system_status: str = "Ready"
    instrument_status_active: str = "Active"
    instrument_status_idle: str = "Idle"
    randomness_source_connected: str = "Connected"
    randomness_source_disconnected: str = "Disconnected"

    # --- Randomness mode copy ------------------------------------------------
    randomness_mode_label: str = "Randomness mode"
    randomness_mode_help: str = (
        "Choose the entropy source. Playback must be stopped to switch."
    )
    mode_classical_label: str = "Classical PRNG"
    mode_quantum_label: str = "Quantum — Qiskit Aer"
    circuit_label: str = "Entropy circuit"
    generated_label: str = "Bits generated"
    zero_ones_help: str = (
        "Lifetime 0/1 counts from the entropy source; a balanced "
        "measurement stream clusters around 50% but is not exactly 50%."
    )

    # --- Performance copy ----------------------------------------------------
    current_stroke_label: str = "Current stroke"
    audio_health_label: str = "Audio health"
    no_stroke_yet: str = "—"
    audio_error_prefix: str = "Audio engine error"

    # --- Placeholder copy ----------------------------------------------------
    waveform_label: str = "Waveform"
    waveform_note: str = "Rolling visualisation of the live audio output"
    parameter_status_note: str = "Normalised 0–1 · driven live by the entropy stream"

    # --- Metric help text ----------------------------------------------------
    playback_status_help: str = "Whether the live performance engine is active."
    system_status_help: str = "Health of the dashboard runtime."
    instrument_status_help: str = "Virtual tabla state built by the DSP engine."
    quantum_source_status_help: str = (
        "Status of the random bit source feeding the synthesizer."
    )
    runtime_help: str = "Time elapsed since Start was pressed."
    update_count_help: str = "Number of strokes played by the engine."
    frame_rate_help: str = "Rolling update rate across recent frames."
    entropy_rate_help: str = "Random bits consumed by the decoder per second."
    buffer_size_help: str = "Random bits currently available in the reservoir."
    bits_consumed_help: str = "Total random bits consumed since Start."
    performance_tempo_help: str = "Live tempo of the generative rhythm engine."
    current_stroke_help: str = "Most recently played stroke type."
    current_pitch_help: str = "Dominant frequency of the most recent stroke."
    accent_help: str = "Accent applied to the most recent stroke."
    density_help: str = "Rhythmic density of the current phrase."
    audio_health_help: str = "Status reported by the audio device callback."
    starvation_help: str = "Times the randomness reservoir could not serve a decision."

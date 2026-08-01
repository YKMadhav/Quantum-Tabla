"""Factory: assemble the full randomness stack from configuration.

Keeps construction wiring out of ``app.py``. To swap the entropy source later,
add a new provider in ``providers`` and register it here — no other module
changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.config import AppConfig, TierBehaviour
from src.core.instrument.manager import InstrumentStateManager
from src.core.randomness.base import RandomnessProvider
from src.core.randomness.classical import ClassicalRandomProvider
from src.core.randomness.decoder import ParameterDecoder
from src.core.randomness.mapper import (
    DEFAULT_CATALOG,
    ParameterCatalog,
    ParameterMapper,
    UpdateTier,
)
from src.core.randomness.qiskit_provider import QiskitRandomnessProvider
from src.core.randomness.scheduler import UpdateScheduler
from src.core.randomness.stream import BitStreamManager

#: Provider names accepted by :func:`_build_provider`.
PROVIDER_REGISTRY: tuple[str, ...] = ("classical", "quantum")


@dataclass(frozen=True)
class RandomnessStack:
    """The fully wired randomness subsystem for one session."""

    provider: RandomnessProvider
    bit_stream: BitStreamManager
    mapper: ParameterMapper
    decoder: ParameterDecoder
    scheduler: UpdateScheduler
    instrument: InstrumentStateManager


def _build_provider(config: AppConfig) -> RandomnessProvider:
    """Create the provider named by the configuration.

    Classical and quantum providers are registered here; consumers only ever
    see the ``RandomnessProvider`` interface, so swapping the entropy source
    touches nothing else.
    """
    randomness = config.randomness
    if randomness.provider_name == "classical":
        return ClassicalRandomProvider(
            seed=randomness.provider_seed,
            buffer_size=randomness.provider_buffer_bits,
        )
    if randomness.provider_name == "quantum":
        quantum = randomness.quantum
        return QiskitRandomnessProvider(
            num_qubits=quantum.num_qubits,
            batch_shots=quantum.batch_shots,
            low_water_bits=quantum.low_water_bits,
            max_buffer_bits=quantum.max_buffer_bits,
        )
    raise ValueError(
        f"Unknown randomness provider: {randomness.provider_name!r} "
        f"(choose from {', '.join(PROVIDER_REGISTRY)})"
    )


def _tier_behaviours(
    config: AppConfig,
) -> dict[UpdateTier, TierBehaviour]:
    randomness = config.randomness
    return {
        UpdateTier.SLOW: randomness.slow,
        UpdateTier.MEDIUM: randomness.medium,
        UpdateTier.FAST: randomness.fast,
    }


def build_randomness_stack(
    config: AppConfig,
    catalog: ParameterCatalog = DEFAULT_CATALOG,
) -> RandomnessStack:
    """Assemble every randomness component wired from *config*."""
    randomness = config.randomness
    tier_behaviours = _tier_behaviours(config)

    provider = _build_provider(config)
    bit_stream = BitStreamManager(
        provider,
        target_bits=randomness.buffer_target_bits,
        refill_batch_bits=randomness.refill_batch_bits,
        refill_interval_s=randomness.refill_interval_s,
    )
    mapper = ParameterMapper(catalog)
    scheduler = UpdateScheduler(catalog, tier_behaviours)
    decoder = ParameterDecoder(
        mapper, chunk_size_bits=randomness.chunk_size_bits
    )
    instrument = InstrumentStateManager(
        decoder, scheduler, mapper, tier_behaviours
    )

    return RandomnessStack(
        provider=provider,
        bit_stream=bit_stream,
        mapper=mapper,
        decoder=decoder,
        scheduler=scheduler,
        instrument=instrument,
    )

"""Binary-string utilities for the randomness subsystem.

All randomness processing treats bits as strings of ``'0'``/``'1'`` characters.
These pure functions perform the core conversions used by the chunk extractor
and are kept framework-free so they can be unit-tested in isolation.
"""

from __future__ import annotations


def validate_bitstring(bits: object) -> bool:
    """Return True if *bits* is a non-empty string of ``'0'``/``'1'`` only."""
    return isinstance(bits, str) and bool(bits) and set(bits) <= {"0", "1"}


def bits_to_int(bits: str) -> int:
    """Convert a binary string to its integer value.

    The most significant bit is the first character, e.g. ``"101"`` -> ``5``.

    Raises:
        ValueError: if *bits* is empty or contains anything but ``'0'``/``'1'``.
    """
    if not validate_bitstring(bits):
        raise ValueError(f"Invalid binary string: {bits!r}")
    return int(bits, 2)


def normalize(value: int, max_value: int) -> float:
    """Map an integer chunk onto ``[0.0, 1.0]`` by dividing by *max_value*."""
    if max_value <= 0:
        raise ValueError(f"max_value must be positive, got {max_value}")
    return max(0.0, min(1.0, value / max_value))


def clamp01(value: float) -> float:
    """Clamp a value into the closed interval ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, value))

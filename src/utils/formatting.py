"""Human-friendly formatting helpers used across the dashboard."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``HH:MM:SS`` (``MM:SS`` when short).

    Negative input is clamped to zero; fractional seconds are truncated.
    """
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_bits(count: int) -> str:
    """Format a number of bits with a ``bit``/``kbit`` unit suffix."""
    if count >= 1000:
        return f"{count / 1000.0:.1f} kbit"
    return f"{count} bit"


def format_bits_per_second(rate: float) -> str:
    """Format a bit rate with a ``bit/s``/``kbit/s`` unit suffix."""
    if rate >= 1000.0:
        return f"{rate / 1000.0:.1f} kbit/s"
    return f"{rate:.0f} bit/s"

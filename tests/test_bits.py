"""Unit tests for binary-string utilities."""

from __future__ import annotations

import pytest

from src.utils import bits as bit_utils

CHUNK = "1010011010"  # the canonical example: 666 -> 0.651


def test_validate_bitstring() -> None:
    assert bit_utils.validate_bitstring("1010")
    assert bit_utils.validate_bitstring("0")
    assert not bit_utils.validate_bitstring("")
    assert not bit_utils.validate_bitstring("102")
    assert not bit_utils.validate_bitstring(123)
    assert not bit_utils.validate_bitstring(None)


def test_bits_to_int() -> None:
    assert bit_utils.bits_to_int(CHUNK) == 666
    assert bit_utils.bits_to_int("0000000000") == 0
    assert bit_utils.bits_to_int("1111111111") == 1023
    assert bit_utils.bits_to_int("101") == 5


def test_bits_to_int_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        bit_utils.bits_to_int("")
    with pytest.raises(ValueError):
        bit_utils.bits_to_int("10a2")


def test_normalize() -> None:
    assert bit_utils.normalize(0, 1023) == 0.0
    assert bit_utils.normalize(1023, 1023) == 1.0
    assert bit_utils.normalize(666, 1023) == pytest.approx(0.651, abs=0.001)
    assert bit_utils.normalize(-5, 1023) == 0.0
    assert bit_utils.normalize(5000, 1023) == 1.0


def test_normalize_rejects_bad_max() -> None:
    with pytest.raises(ValueError):
        bit_utils.normalize(1, 0)


def test_clamp01() -> None:
    assert bit_utils.clamp01(-1.0) == 0.0
    assert bit_utils.clamp01(0.5) == 0.5
    assert bit_utils.clamp01(1.5) == 1.0

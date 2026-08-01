"""Unit tests for the thread-safe bit buffer."""

from __future__ import annotations

import pytest

from src.core.randomness.buffer import BitBuffer


def test_append_take_round_trip() -> None:
    buffer = BitBuffer()
    buffer.append("101001")
    assert buffer.available == 6
    assert buffer.take(6) == "101001"
    assert buffer.available == 0


def test_append_is_fifo() -> None:
    buffer = BitBuffer()
    buffer.append("1010")
    buffer.append("0101")
    assert buffer.take(4) == "1010"
    assert buffer.take(4) == "0101"


def test_take_partial_chunk() -> None:
    buffer = BitBuffer()
    buffer.append("1010011010")
    assert buffer.take(4) == "1010"
    assert buffer.take(4) == "0110"
    assert buffer.available == 2


def test_take_more_than_available() -> None:
    buffer = BitBuffer()
    buffer.append("1010")
    assert buffer.take(100) == "1010"
    assert buffer.take(100) == ""
    assert buffer.available == 0


def test_preview_does_not_consume() -> None:
    buffer = BitBuffer()
    buffer.append("110011")
    assert buffer.preview(4) == "1100"
    assert buffer.available == 6
    assert buffer.take(6) == "110011"


def test_clear() -> None:
    buffer = BitBuffer()
    buffer.append("1010")
    buffer.clear()
    assert buffer.available == 0
    assert buffer.take(2) == ""


def test_rejects_invalid_bits() -> None:
    buffer = BitBuffer()
    with pytest.raises(ValueError):
        buffer.append("102")
    with pytest.raises(ValueError):
        buffer.append("")


def test_concurrent_append_and_take() -> None:
    import threading

    buffer = BitBuffer()
    errors: list[Exception] = []

    def producer() -> None:
        try:
            for _ in range(200):
                buffer.append("1" * 50)
        except Exception as exc:  # pragma: no cover - safety net
            errors.append(exc)

    thread = threading.Thread(target=producer)
    thread.start()
    consumed = 0
    for _ in range(200):
        consumed += len(buffer.take(40))
    thread.join()
    assert not errors
    assert buffer.available == 200 * 50 - consumed

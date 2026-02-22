#!/usr/bin/env python3
import bisect
from typing import Iterable, List


MAX_CODE = 0xFFFFFFFF
HALF = 0x80000000
QUARTER = 0x40000000
THREE_QUARTER = 0xC0000000


class BitWriter:
    def __init__(self) -> None:
        self._buf = bytearray()
        self._acc = 0
        self._nbits = 0

    def write_bit(self, bit: int) -> None:
        self._acc = (self._acc << 1) | (bit & 1)
        self._nbits += 1
        if self._nbits == 8:
            self._buf.append(self._acc & 0xFF)
            self._acc = 0
            self._nbits = 0

    def finish(self) -> bytes:
        if self._nbits:
            self._acc <<= (8 - self._nbits)
            self._buf.append(self._acc & 0xFF)
            self._acc = 0
            self._nbits = 0
        return bytes(self._buf)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._idx = 0
        self._acc = 0
        self._nbits = 0

    def read_bit(self) -> int:
        if self._nbits == 0:
            if self._idx >= len(self._data):
                return 0
            self._acc = self._data[self._idx]
            self._idx += 1
            self._nbits = 8
        bit = (self._acc >> (self._nbits - 1)) & 1
        self._nbits -= 1
        return bit


def build_cumulative(freqs: List[int]) -> List[int]:
    total = 0
    cum = [0]
    for f in freqs:
        if f < 0:
            raise ValueError("Negative frequency not allowed.")
        total += f
        cum.append(total)
    if total <= 0:
        raise ValueError("At least one symbol must have positive frequency.")
    return cum


def arithmetic_encode(symbols: Iterable[int], freqs: List[int]) -> bytes:
    cum = build_cumulative(freqs)
    total = cum[-1]
    low = 0
    high = MAX_CODE
    pending = 0
    writer = BitWriter()

    def output_with_pending(bit: int) -> None:
        nonlocal pending
        writer.write_bit(bit)
        inv = 1 - bit
        while pending > 0:
            writer.write_bit(inv)
            pending -= 1

    for sym in symbols:
        if sym < 0 or sym >= len(freqs):
            raise ValueError(f"Symbol out of range: {sym}")
        sym_low = cum[sym]
        sym_high = cum[sym + 1]
        if sym_low == sym_high:
            raise ValueError(f"Symbol {sym} has zero frequency in codebook.")

        current_range = high - low + 1
        high = low + (current_range * sym_high // total) - 1
        low = low + (current_range * sym_low // total)

        while True:
            if high < HALF:
                output_with_pending(0)
            elif low >= HALF:
                output_with_pending(1)
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTER:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low = (low << 1) & MAX_CODE
            high = ((high << 1) & MAX_CODE) | 1

    pending += 1
    if low < QUARTER:
        output_with_pending(0)
    else:
        output_with_pending(1)
    return writer.finish()


def arithmetic_decode(data: bytes, freqs: List[int], count: int) -> List[int]:
    cum = build_cumulative(freqs)
    total = cum[-1]
    low = 0
    high = MAX_CODE
    reader = BitReader(data)
    value = 0
    for _ in range(32):
        value = ((value << 1) | reader.read_bit()) & MAX_CODE

    out: List[int] = []
    for _ in range(count):
        current_range = high - low + 1
        scaled = ((value - low + 1) * total - 1) // current_range
        sym = bisect.bisect_right(cum, scaled) - 1
        if sym < 0 or sym >= len(freqs):
            raise ValueError("Decode failed: symbol out of range.")
        if cum[sym] == cum[sym + 1]:
            raise ValueError("Decode failed: resolved zero-frequency symbol.")
        out.append(sym)

        sym_low = cum[sym]
        sym_high = cum[sym + 1]
        high = low + (current_range * sym_high // total) - 1
        low = low + (current_range * sym_low // total)

        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= QUARTER and high < THREE_QUARTER:
                low -= QUARTER
                high -= QUARTER
                value -= QUARTER
            else:
                break
            low = (low << 1) & MAX_CODE
            high = ((high << 1) & MAX_CODE) | 1
            value = ((value << 1) & MAX_CODE) | reader.read_bit()

    return out

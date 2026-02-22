#!/usr/bin/env python3
from typing import List, Tuple


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


class AdaptiveBinaryModel:
    def __init__(self, contexts: int, init0: int = 1, init1: int = 1, rescale_threshold: int = 1 << 15) -> None:
        self.count0 = [init0] * contexts
        self.count1 = [init1] * contexts
        self.rescale_threshold = rescale_threshold

    @classmethod
    def from_counts(cls, count0: List[int], count1: List[int], rescale_threshold: int) -> "AdaptiveBinaryModel":
        if len(count0) != len(count1):
            raise ValueError("count0/count1 length mismatch.")
        obj = cls(1, 1, 1, rescale_threshold)
        obj.count0 = list(count0)
        obj.count1 = list(count1)
        return obj

    def get(self, ctx: int) -> Tuple[int, int]:
        return self.count0[ctx], self.count1[ctx]

    def update(self, ctx: int, bit: int) -> None:
        if bit == 0:
            self.count0[ctx] += 1
        else:
            self.count1[ctx] += 1
        total = self.count0[ctx] + self.count1[ctx]
        if total >= self.rescale_threshold:
            self.count0[ctx] = max(1, self.count0[ctx] >> 1)
            self.count1[ctx] = max(1, self.count1[ctx] >> 1)


class BinaryArithmeticEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = MAX_CODE
        self.pending = 0
        self.writer = BitWriter()

    def _output_with_pending(self, bit: int) -> None:
        self.writer.write_bit(bit)
        inv = 1 - bit
        while self.pending > 0:
            self.writer.write_bit(inv)
            self.pending -= 1

    def encode_bit(self, bit: int, count0: int, count1: int) -> None:
        total = count0 + count1
        rng = self.high - self.low + 1
        split = self.low + (rng * count0 // total) - 1
        if bit == 0:
            self.high = split
        else:
            self.low = split + 1

        while True:
            if self.high < HALF:
                self._output_with_pending(0)
            elif self.low >= HALF:
                self._output_with_pending(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MAX_CODE
            self.high = ((self.high << 1) & MAX_CODE) | 1

    def finish(self) -> bytes:
        self.pending += 1
        if self.low < QUARTER:
            self._output_with_pending(0)
        else:
            self._output_with_pending(1)
        return self.writer.finish()


class BinaryArithmeticDecoder:
    def __init__(self, data: bytes) -> None:
        self.low = 0
        self.high = MAX_CODE
        self.reader = BitReader(data)
        self.value = 0
        for _ in range(32):
            self.value = ((self.value << 1) | self.reader.read_bit()) & MAX_CODE

    def decode_bit(self, count0: int, count1: int) -> int:
        total = count0 + count1
        rng = self.high - self.low + 1
        split = self.low + (rng * count0 // total) - 1
        if self.value <= split:
            bit = 0
            self.high = split
        else:
            bit = 1
            self.low = split + 1

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MAX_CODE
            self.high = ((self.high << 1) & MAX_CODE) | 1
            self.value = ((self.value << 1) & MAX_CODE) | self.reader.read_bit()

        return bit

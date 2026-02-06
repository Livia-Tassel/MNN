#!/usr/bin/env python3
import heapq
from typing import Dict, Iterable, List, Tuple


def build_huffman_code_lengths(freqs: List[int]) -> List[int]:
    n = len(freqs)
    heap = []
    for sym, f in enumerate(freqs):
        if f > 0:
            heap.append((f, sym, None, None))
    heapq.heapify(heap)

    if not heap:
        return [0] * n
    if len(heap) == 1:
        lengths = [0] * n
        lengths[heap[0][1]] = 1
        return lengths

    while len(heap) > 1:
        f1, s1, l1, r1 = heapq.heappop(heap)
        f2, s2, l2, r2 = heapq.heappop(heap)
        node = (f1 + f2, -1, (f1, s1, l1, r1), (f2, s2, l2, r2))
        heapq.heappush(heap, node)

    _, _, left, right = heap[0]
    lengths = [0] * n

    def walk(node, depth):
        f, sym, l, r = node
        if sym >= 0:
            lengths[sym] = max(1, depth)
            return
        walk(l, depth + 1)
        walk(r, depth + 1)

    walk(left, 0)
    walk(right, 0)
    return lengths


def build_canonical_codes(lengths: List[int]) -> Dict[int, Tuple[int, int]]:
    items = [(l, s) for s, l in enumerate(lengths) if l > 0]
    items.sort()
    codes: Dict[int, Tuple[int, int]] = {}
    code = 0
    prev_len = 0
    for length, sym in items:
        code <<= (length - prev_len)
        codes[sym] = (code, length)
        code += 1
        prev_len = length
    return codes


def build_decode_table(lengths: List[int]) -> Dict[int, Dict[int, int]]:
    codes = build_canonical_codes(lengths)
    table: Dict[int, Dict[int, int]] = {}
    for sym, (code, length) in codes.items():
        table.setdefault(length, {})[code] = sym
    return table


class BitWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.bits = 0

    def write(self, code: int, length: int) -> None:
        for i in range(length - 1, -1, -1):
            bit = (code >> i) & 1
            self.acc = (self.acc << 1) | bit
            self.bits += 1
            if self.bits == 8:
                self.buf.append(self.acc & 0xFF)
                self.acc = 0
                self.bits = 0

    def finish(self) -> bytes:
        if self.bits > 0:
            self.acc <<= (8 - self.bits)
            self.buf.append(self.acc & 0xFF)
            self.acc = 0
            self.bits = 0
        return bytes(self.buf)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte_idx = 0
        self.acc = 0
        self.bits = 0

    def read_bit(self) -> int:
        if self.bits == 0:
            if self.byte_idx >= len(self.data):
                return -1
            self.acc = self.data[self.byte_idx]
            self.byte_idx += 1
            self.bits = 8
        bit = (self.acc >> (self.bits - 1)) & 1
        self.bits -= 1
        return bit


def encode_symbols(symbols: Iterable[int], lengths: List[int]) -> bytes:
    codes = build_canonical_codes(lengths)
    writer = BitWriter()
    for sym in symbols:
        code, length = codes[sym]
        writer.write(code, length)
    return writer.finish()


def decode_symbols(data: bytes, lengths: List[int], count: int) -> List[int]:
    table = build_decode_table(lengths)
    max_len = max((l for l in lengths if l > 0), default=0)
    reader = BitReader(data)
    out = []
    code = 0
    length = 0
    while len(out) < count:
        bit = reader.read_bit()
        if bit < 0:
            break
        code = (code << 1) | bit
        length += 1
        if length in table and code in table[length]:
            out.append(table[length][code])
            code = 0
            length = 0
        elif length > max_len:
            raise ValueError("Invalid bitstream: no matching Huffman code.")
    if len(out) != count:
        raise ValueError(f"Decoded {len(out)} symbols, expected {count}.")
    return out

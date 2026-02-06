#!/usr/bin/env python3
from typing import List, Tuple


TOT_BITS = 12
TOT = 1 << TOT_BITS  # 4096


def normalize_freqs(freqs: List[int], total: int = TOT) -> Tuple[List[int], List[int]]:
    if len(freqs) == 0:
        raise ValueError("Empty frequency list.")
    s = sum(freqs)
    if s <= 0:
        raise ValueError("All zero frequencies.")
    scaled = [0] * len(freqs)
    nonzero = [i for i, f in enumerate(freqs) if f > 0]
    for i in nonzero:
        scaled[i] = max(1, int(freqs[i] * total / s))
    scaled_sum = sum(scaled)
    # Fix sum to total by adjusting largest symbols
    if scaled_sum != total:
        dif = total - scaled_sum
        order = sorted(nonzero, key=lambda i: freqs[i], reverse=True)
        if not order:
            raise ValueError("No nonzero symbols to normalize.")
        idx = 0
        while dif != 0:
            i = order[idx]
            if dif > 0:
                scaled[i] += 1
                dif -= 1
            else:
                if scaled[i] > 1:
                    scaled[i] -= 1
                    dif += 1
            idx = (idx + 1) % len(order)
    # Build cumulative
    cum = [0]
    for f in scaled:
        cum.append(cum[-1] + f)
    if cum[-1] != total:
        raise ValueError("Normalization failed to reach total.")
    return scaled, cum


def build_symbol_table(freqs: List[int]) -> Tuple[List[int], List[int], List[int]]:
    norm_freqs, cum = normalize_freqs(freqs)
    # For decode: table of size TOT mapping slot -> symbol
    sym_table = [0] * TOT
    for sym, f in enumerate(norm_freqs):
        start = cum[sym]
        end = cum[sym + 1]
        for i in range(start, end):
            sym_table[i] = sym
    return norm_freqs, cum, sym_table


def rans_encode(symbols: List[int], norm_freqs: List[int], cum: List[int]) -> bytes:
    # rANS encoder (byte-wise)
    x = 1 << 23
    out = bytearray()
    for sym in reversed(symbols):
        f = norm_freqs[sym]
        c = cum[sym]
        while x >= (f << 16):
            out.append(x & 0xFF)
            x >>= 8
        x = (x // f) * TOT + (x % f) + c
    # flush state
    for _ in range(4):
        out.append(x & 0xFF)
        x >>= 8
    return bytes(out)


def rans_decode(data: bytes, count: int, norm_freqs: List[int], cum: List[int], sym_table: List[int]) -> List[int]:
    if len(data) < 4:
        raise ValueError("Data too short for rANS decode.")
    idx = len(data) - 1
    x = 0
    for shift in range(0, 32, 8):
        x |= data[idx] << shift
        idx -= 1

    out = []
    for _ in range(count):
        slot = x & (TOT - 1)
        sym = sym_table[slot]
        out.append(sym)
        f = norm_freqs[sym]
        c = cum[sym]
        x = f * (x >> TOT_BITS) + (slot - c)
        while x < (1 << 23):
            if idx < 0:
                raise ValueError("rANS underflow: not enough bytes.")
            x = (x << 8) | data[idx]
            idx -= 1
    return out


def rans_encode_ctx(symbols: List[int], ctx_ids: List[int], norm_freqs_list: List[List[int]], cum_list: List[List[int]]) -> bytes:
    if len(symbols) != len(ctx_ids):
        raise ValueError("symbols and ctx_ids length mismatch.")
    x = 1 << 23
    out = bytearray()
    for sym, ctx in reversed(list(zip(symbols, ctx_ids))):
        norm_freqs = norm_freqs_list[ctx]
        cum = cum_list[ctx]
        f = norm_freqs[sym]
        c = cum[sym]
        while x >= (f << 16):
            out.append(x & 0xFF)
            x >>= 8
        x = (x // f) * TOT + (x % f) + c
    for _ in range(4):
        out.append(x & 0xFF)
        x >>= 8
    return bytes(out)


def rans_decode_ctx(
    data: bytes,
    count: int,
    ctx_ids: List[int],
    norm_freqs_list: List[List[int]],
    cum_list: List[List[int]],
    sym_tables: List[List[int]],
) -> List[int]:
    if len(ctx_ids) != count:
        raise ValueError("ctx_ids length mismatch.")
    if len(data) < 4:
        raise ValueError("Data too short for rANS decode.")
    idx = len(data) - 1
    x = 0
    for shift in range(0, 32, 8):
        x |= data[idx] << shift
        idx -= 1
    out = [0] * count
    for i in range(count):
        ctx = ctx_ids[i]
        sym_table = sym_tables[ctx]
        norm_freqs = norm_freqs_list[ctx]
        cum = cum_list[ctx]
        slot = x & (TOT - 1)
        sym = sym_table[slot]
        out[i] = sym
        f = norm_freqs[sym]
        c = cum[sym]
        x = f * (x >> TOT_BITS) + (slot - c)
        while x < (1 << 23):
            if idx < 0:
                raise ValueError("rANS underflow: not enough bytes.")
            x = (x << 8) | data[idx]
            idx -= 1
    return out


class RansDecoder:
    def __init__(self, data: bytes) -> None:
        if len(data) < 4:
            raise ValueError("Data too short for rANS decode.")
        self._data = data
        self._idx = len(data) - 1
        self._x = 0
        for shift in range(0, 32, 8):
            self._x |= data[self._idx] << shift
            self._idx -= 1

    def decode_symbol(self, sym_table: List[int], norm_freqs: List[int], cum: List[int]) -> int:
        slot = self._x & (TOT - 1)
        sym = sym_table[slot]
        f = norm_freqs[sym]
        c = cum[sym]
        self._x = f * (self._x >> TOT_BITS) + (slot - c)
        while self._x < (1 << 23):
            if self._idx < 0:
                raise ValueError("rANS underflow: not enough bytes.")
            self._x = (self._x << 8) | self._data[self._idx]
            self._idx -= 1
        return sym

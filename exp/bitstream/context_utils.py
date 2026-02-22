#!/usr/bin/env python3
from typing import Tuple


def ctx_id(hi6: int, prev_hi6: int, prev_lo10: int, buckets: int, a: int, b: int) -> int:
    return (hi6 * a + prev_hi6 * b + prev_lo10) % buckets


def update_prev(prev_hi6: int, prev_lo10: int, hi6: int, lo10: int) -> Tuple[int, int]:
    return hi6, lo10

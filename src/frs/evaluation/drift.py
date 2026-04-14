from __future__ import annotations

import math
from typing import Sequence


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-9) -> float:
    if len(p) != len(q):
        raise ValueError('p and q must have the same length')
    total = 0.0
    for p_i, q_i in zip(p, q):
        p_safe = max(float(p_i), eps)
        q_safe = max(float(q_i), eps)
        total += p_safe * math.log(p_safe / q_safe)
    return total


def mean_kl_divergence(ps: Sequence[Sequence[float]], qs: Sequence[Sequence[float]]) -> float:
    if len(ps) != len(qs):
        raise ValueError('ps and qs must have the same number of rows')
    if not ps:
        return 0.0
    return sum(kl_divergence(p, q) for p, q in zip(ps, qs)) / len(ps)

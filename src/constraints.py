from __future__ import annotations
from typing import Dict
import math


def derive_priorities_from_ranks(
    rank_map: Dict[str, int],
    alpha: float = 0.35,
) -> Dict[str, float]:
    """
    Monotonic weight mapping from integer ranks (1=highest). Ties allowed.
    Uses exponential decay: w_raw = exp(-alpha * (rank-1)).
    Returns unnormalized weights (0,1].
    """
    if not rank_map:
        raise ValueError("Empty ranks not allowed")
    weights: Dict[str, float] = {}
    for k, r in rank_map.items():
        r = max(1, int(r))
        w = math.exp(-alpha * (r - 1))
        weights[k] = float(w)
    return weights


def normalize_weights(priorities: Dict[str, float]) -> Dict[str, float]:
    if not priorities:
        return {}
    mx = max(priorities.values())
    if mx <= 0:
        return {k: 0.0 for k in priorities}
    return {k: v / mx for k, v in priorities.items()}


def normalize_to_range(priorities: Dict[str, float], lo: float = 0.2, hi: float = 1.0) -> Dict[str, float]:
    """
    Affine-normalize arbitrary non-negative weights into [lo, hi].
    If all zeros, return equal weights at lo.
    """
    if not priorities:
        return {}
    vals = list(priorities.values())
    mn, mx = min(vals), max(vals)
    if mx <= 0:
        # all zeros → set to lo
        return {k: lo for k in priorities}
    if abs(mx - mn) < 1e-12:
        # constant weights → map all to hi
        return {k: hi for k in priorities}
    out: Dict[str, float] = {}
    for k, v in priorities.items():
        z = (v - mn) / (mx - mn)
        out[k] = lo + z * (hi - lo)
    return out

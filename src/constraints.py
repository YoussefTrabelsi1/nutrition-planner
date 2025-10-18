from __future__ import annotations
from typing import Dict, Iterable


def derive_priorities_from_ranks(
    rank_map: Dict[str, int],
    default: float = 1.0,
) -> Dict[str, float]:
    """Monotonic weight mapping from integer ranks (1=highest). Ties allowed.
    We map weight = 1 - (rank-1)/(max_rank) to get (0,1].
    """
    if not rank_map:
        raise ValueError("Empty ranks not allowed")
    max_rank = max(rank_map.values())
    weights: Dict[str, float] = {}
    for k, r in rank_map.items():
        w = 1.0 - (r - 1) / max_rank
        weights[k] = round(w, 6)
    return weights


def normalize_weights(priorities: Dict[str, float]) -> Dict[str, float]:
    if not priorities:
        return {}
    mx = max(priorities.values())
    if mx <= 0:
        return {k: 0.0 for k in priorities}
    return {k: v / mx for k, v in priorities.items()}

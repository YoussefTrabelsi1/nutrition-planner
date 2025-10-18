from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .models import Problem, Food


def _omit_near_zero(plan: Dict[str, float], eps: float = 1e-6) -> Dict[str, float]:
    return {k: v for k, v in plan.items() if v > eps}


def compute_totals(problem: Problem, plan: Dict[str, float]) -> Dict[str, float]:
    totals: Dict[str, float] = {"kcal": 0.0}
    for f in problem.foods:
        g = plan.get(f.name, 0.0)
        if g <= 0:
            continue
        totals["kcal"] += g * (f.kcal_per_100g / 100.0)
        for k, v in f.per100.items():
            totals[k] = totals.get(k, 0.0) + g * (v / 100.0)
    return totals


def solve_lp_only(problem: Problem) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """Generic LP builder/solver.
    Returns (plan, totals) if status is Optimal; returns None if PuLP missing
    or the model is infeasible/non-optimal.
    """
    try:
        import pulp  # type: ignore
    except Exception:
        return None

    foods = problem.foods
    targets = problem.targets

    # Decision variables: grams per food, 0..max_serving_g
    x = {f.name: pulp.LpVariable(f"g_{f.name}", lowBound=0, upBound=f.max_serving_g) for f in foods}
    model = pulp.LpProblem("nutrition_min_kcal", pulp.LpMinimize)

    # Objective: minimize calories
    model += pulp.lpSum([x[f.name] * (f.kcal_per_100g / 100.0) for f in foods])

    # Calorie budget
    model += pulp.lpSum([x[f.name] * (f.kcal_per_100g / 100.0) for f in foods]) <= targets.kcal_remaining

    # Min constraints (any nutrient present in mins)
    for key, need in targets.mins.items():
        model += pulp.lpSum([x[f.name] * (f.per100.get(key, 0.0) / 100.0) for f in foods]) >= need

    # Max constraints (any compound present in maxes)
    for key, cap in targets.maxes.items():
        model += pulp.lpSum([x[f.name] * (f.per100.get(key, 0.0) / 100.0) for f in foods]) <= cap

    # Solve silently
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[model.status] != "Optimal":
        return None

    plan = {f.name: float(x[f.name].value() or 0.0) for f in foods}
    plan = _omit_near_zero(plan)
    totals = compute_totals(problem, plan)
    return plan, totals


def solve_greedy(problem: Problem) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Deterministic, priority-aware greedy fallback.

    Scoring per kcal (using per-100g data):
        score(food) = sum_n (w[n]*per100[n]) / kcal_per_100g
                       - toxin_penalty * (sum_t per100[toxin]) / (kcal_per_100g * 1000)

    Algorithm:
      - Fixed chunk size per add (CHUNK_G).
      - While mins unmet and kcal remaining, pick highest-score food (tie: name asc).
      - Add up to min(chunk, max_serving_g-left, kcal headroom, toxin headrooms). If zero, mark exhausted.
      - Stop when no progress possible.
    """
    foods = problem.foods
    mins = problem.targets.mins
    maxes = problem.targets.maxes
    kcal_cap = problem.targets.kcal_remaining
    weights = problem.priorities or {k: 1.0 for k in mins}
    toxin_penalty = problem.toxin_penalty

    # Deterministic chunk size
    CHUNK_G = 25.0

    # State
    plan: Dict[str, float] = {f.name: 0.0 for f in foods}
    totals: Dict[str, float] = {"kcal": 0.0}

    # Track exhaustion deterministically
    exhausted: Dict[str, bool] = {f.name: False for f in foods}

    def unmet_exist() -> bool:
        for k, need in mins.items():
            if totals.get(k, 0.0) + 1e-9 < need:
                return True
        return False

    def kcal_remaining() -> float:
        return max(0.0, kcal_cap - totals.get("kcal", 0.0))

    def max_add_by_caps(f: Food) -> float:
        """Compute max grams we can add for this food without violating kcal or toxin caps."""
        # serving limit
        rem_serv = max(0.0, f.max_serving_g - plan[f.name])
        if rem_serv <= 0:
            return 0.0
        # kcal limit
        kpg = max(f.kcal_per_100g / 100.0, 1e-9)
        rem_kcal_g = kcal_remaining() / kpg if kpg > 0 else rem_serv
        g_cap = min(rem_serv, rem_kcal_g)
        # toxin caps
        for t, cap in maxes.items():
            per_g = f.per100.get(t, 0.0) / 100.0
            if per_g <= 0:
                continue
            head = max(0.0, cap - totals.get(t, 0.0))
            if head <= 1e-12:
                return 0.0
            g_cap = min(g_cap, head / per_g)
        return max(0.0, g_cap)

    def score_food(f: Food) -> float:
        kcal100 = max(f.kcal_per_100g, 1e-6)
        # Only count weights for nutrients that are still unmet
        benefit = 0.0
        for n, w in weights.items():
            if mins.get(n, 0.0) > totals.get(n, 0.0) + 1e-9:
                benefit += w * f.per100.get(n, 0.0)
        toxin_sum = 0.0
        for t in maxes.keys():
            toxin_sum += f.per100.get(t, 0.0)
        return (benefit / kcal100) - toxin_penalty * (toxin_sum / (kcal100 * 1000.0))

    # Main loop (bounded iterations for determinism)
    for _ in range(10000):
        if not unmet_exist():
            break
        if kcal_remaining() <= 1e-9:
            break

        # Score candidates
        candidates: List[Tuple[float, Food]] = []
        for f in foods:
            if exhausted[f.name]:
                continue
            # Skip foods that cannot contribute to any unmet nutrient
            contributes = any(
                f.per100.get(n, 0.0) > 0 and mins.get(n, 0.0) > totals.get(n, 0.0) + 1e-9
                for n in mins
            )
            if not contributes:
                continue
            s = score_food(f)
            candidates.append((s, f))
        if not candidates:
            break
        # Deterministic ordering: highest score, tie by name
        candidates.sort(key=lambda x: (-x[0], x[1].name))

        progressed = False
        for _, f in candidates:
            g_cap = max_add_by_caps(f)
            if g_cap <= 1e-9:
                exhausted[f.name] = True
                continue
            add = min(CHUNK_G, g_cap)
            if add <= 1e-9:
                exhausted[f.name] = True
                continue
            # apply
            plan[f.name] += add
            totals["kcal"] = totals.get("kcal", 0.0) + add * (f.kcal_per_100g / 100.0)
            for k, v in f.per100.items():
                totals[k] = totals.get(k, 0.0) + add * (v / 100.0)
            progressed = True
            break  # re-evaluate deficits each iteration

        if not progressed:
            break

    plan = _omit_near_zero(plan)
    return plan, totals


def solve_lp(problem: Problem):
    """Orchestrator kept for backward-compat: try LP-only; if None, fall back to greedy."""
    res = solve_lp_only(problem)
    if res is not None:
        return res
    return solve_greedy(problem)

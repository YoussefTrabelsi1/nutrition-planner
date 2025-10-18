from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from models import Problem, Food


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
    """Deterministic greedy: fill top-priority deficits per kcal, penalize toxins.
    Strategy:
      - Repeatedly find the most deficient nutrient by weighted deficit.
      - Choose the food with best (benefit - toxin_penalty * toxin_risk) / kcal.
      - Allocate in small steps until either the nutrient is met or food hits its cap.
    """
    foods = problem.foods
    mins = problem.targets.mins
    maxes = problem.targets.maxes
    kcal_cap = problem.targets.kcal_remaining
    prio = problem.priorities or {k: 1.0 for k in mins}
    toxin_penalty = problem.toxin_penalty

    # Current plan
    plan: Dict[str, float] = {f.name: 0.0 for f in foods}
    totals: Dict[str, float] = {"kcal": 0.0}

    # Helper to compute deficits and toxin headroom
    def deficit(key: str) -> float:
        have = totals.get(key, 0.0)
        need = mins[key]
        return max(0.0, need - have)

    def worst_deficit_key() -> str | None:
        candidates = [(prio.get(k, 0.0) * deficit(k), k) for k in mins]
        candidates.sort(reverse=True)
        top = candidates[0]
        return top[1] if top[0] > 1e-9 else None

    step = 5.0  # grams per iteration for determinism

    # Precompute toxin keys
    toxin_keys = list(maxes.keys())

    # Loop until all mins met or kcal exhausted
    for _ in range(10000):  # deterministic bound
        key = worst_deficit_key()
        if key is None:
            break
        # Evaluate foods for this key
        scored: List[Tuple[float, Food]] = []
        for f in foods:
            if plan[f.name] >= f.max_serving_g:
                continue
            inc = f.per100.get(key, 0.0)
            if inc <= 0:
                continue
            kcal_per_g = f.kcal_per_100g / 100.0
            benefit = inc / 100.0  # per gram
            # Toxin risk proxy: sum of (per gram / remaining headroom)
            toxin_risk = 0.0
            for t in toxin_keys:
                per_g = f.per100.get(t, 0.0) / 100.0
                cap = maxes[t]
                have = totals.get(t, 0.0)
                headroom = max(0.0, cap - have)
                if headroom <= 0 and per_g > 0:
                    toxin_risk += 1e6  # prohibit
                elif per_g > 0:
                    toxin_risk += per_g / (headroom + 1e-6)
            score = (benefit - toxin_penalty * toxin_risk) / (kcal_per_g + 1e-9)
            scored.append((score, f))
        if not scored:
            # No food can help this nutrient further
            break
        scored.sort(key=lambda x: (-x[0], x[1].name))
        best = scored[0][1]

        # Allocate step grams respecting all caps
        add = min(step, best.max_serving_g - plan[best.name])
        # Respect kcal cap
        if totals["kcal"] + add * (best.kcal_per_100g / 100.0) > kcal_cap + 1e-9:
            # shrink to fit
            rem = max(0.0, kcal_cap - totals["kcal"]) / (best.kcal_per_100g / 100.0)
            add = min(add, rem)
        if add <= 1e-9:
            break
        plan[best.name] += add
        # update totals
        totals["kcal"] = totals.get("kcal", 0.0) + add * (best.kcal_per_100g / 100.0)
        for k, v in best.per100.items():
            totals[k] = totals.get(k, 0.0) + add * (v / 100.0)
        # Enforce toxin caps strictly
        violated = False
        for t, cap in maxes.items():
            if totals.get(t, 0.0) > cap + 1e-9:
                violated = True
                break
        if violated:
            # rollback
            plan[best.name] -= add
            totals["kcal"] -= add * (best.kcal_per_100g / 100.0)
            for k, v in best.per100.items():
                totals[k] -= add * (v / 100.0)
            # Mark this food as exhausted to avoid infinite loop
            plan[best.name] = best.max_serving_g
            continue

    plan = _omit_near_zero(plan)
    return plan, totals


def solve_lp(problem: Problem):
    """Orchestrator kept for backward-compat: try LP-only; if None, fall back to greedy."""
    res = solve_lp_only(problem)
    if res is not None:
        return res
    return solve_greedy(problem)

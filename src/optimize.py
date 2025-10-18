from __future__ import annotations
from typing import Dict, List, Tuple, Optional
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


def _add_category_caps_lp(model, x_vars, foods: List[Food], caps: Dict[str, float]) -> None:
    import pulp  # type: ignore
    if not caps:
        return
    by_cat: Dict[str, List[Food]] = {}
    for f in foods:
        c = (f.category or "").strip().lower()
        by_cat.setdefault(c, []).append(f)
    for cat, cap in caps.items():
        flist = by_cat.get(cat.lower(), [])
        if not flist:
            continue
        model += pulp.lpSum([x_vars[f.name] for f in flist]) <= cap


def _soft_lp(problem: Problem, weights: Dict[str, float]) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """Soft-constraint LP: minimize weighted shortfall + eps*calories.
    Returns None if PuLP missing or not optimal.
    """
    try:
        import pulp  # type: ignore
    except Exception:
        return None

    foods = problem.foods
    targets = problem.targets

    x = {f.name: pulp.LpVariable(f"g_{f.name}", lowBound=0, upBound=f.max_serving_g) for f in foods}
    # Slack for each min
    s = {k: pulp.LpVariable(f"slack_{k}", lowBound=0) for k in targets.mins.keys()}

    model = pulp.LpProblem("nutrition_soft_min_shortfall", pulp.LpMinimize)

    # Objective: minimize Σ w[n] * s[n] + ε * calories
    eps = 1e-6
    calories_expr = pulp.lpSum([x[f.name] * (f.kcal_per_100g / 100.0) for f in foods])
    shortfall_expr = pulp.lpSum([weights.get(k, 1.0) * s[k] for k in targets.mins.keys()])
    model += shortfall_expr + eps * calories_expr

    # Calories cap
    model += calories_expr <= targets.kcal_remaining

    # Min constraints with slack: provided + slack ≥ need
    for key, need in targets.mins.items():
        model += pulp.lpSum([x[f.name] * (f.per100.get(key, 0.0) / 100.0) for f in foods]) + s[key] >= need

    # Max constraints (toxins)
    for key, cap in targets.maxes.items():
        model += pulp.lpSum([x[f.name] * (f.per100.get(key, 0.0) / 100.0) for f in foods]) <= cap

    # Category caps (optional)
    if problem.category_caps_g:
        _add_category_caps_lp(model, x, foods, problem.category_caps_g)

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    if getattr(pulp, "LpStatus", {}).get(model.status, None) != "Optimal" and pulp.LpStatus[model.status] != "Optimal":
        return None

    plan = {f.name: float(x[f.name].value() or 0.0) for f in foods}
    plan = _omit_near_zero(plan)
    totals = compute_totals(problem, plan)
    return plan, totals


def solve_lp_only(problem: Problem, allow_soft: bool = False) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """Generic LP builder/solver.
    Returns (plan, totals) if status is Optimal;
    If infeasible and allow_soft=True, returns best-effort soft solution;
    Returns None on failure/unavailability.
    """
    try:
        import pulp  # type: ignore
    except Exception:
        from .optimize import _soft_lp  # type: ignore  # circular in this snippet context
        return _soft_lp(problem, problem.priorities) if allow_soft else None

    foods = problem.foods
    targets = problem.targets

    x = {f.name: pulp.LpVariable(f"g_{f.name}", lowBound=0, upBound=f.max_serving_g) for f in foods}
    model = pulp.LpProblem("nutrition_min_kcal", pulp.LpMinimize)

    # Objective: minimize calories
    calories_expr = pulp.lpSum([x[f.name] * (f.kcal_per_100g / 100.0) for f in foods])
    model += calories_expr

    # Calorie budget
    model += calories_expr <= targets.kcal_remaining

    # Min constraints
    for key, need in targets.mins.items():
        model += pulp.lpSum([x[f.name] * (f.per100.get(key, 0.0) / 100.0) for f in foods]) >= need

    # Max constraints (toxins)
    for key, cap in targets.maxes.items():
        model += pulp.lpSum([x[f.name] * (f.per100.get(key, 0.0) / 100.0) for f in foods]) <= cap

    # Category caps (Ticket 8)
    if problem.category_caps_g:
        _add_category_caps_lp(model, x, foods, problem.category_caps_g)

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    if getattr(pulp, "LpStatus", {}).get(model.status, None) != "Optimal" and pulp.LpStatus[model.status] != "Optimal":
        # Try soft solution if enabled
        return _soft_lp(problem, problem.priorities) if allow_soft else None

    plan = {f.name: float(x[f.name].value() or 0.0) for f in foods}
    plan = _omit_near_zero(plan)
    totals = compute_totals(problem, plan)
    return plan, totals


def solve_greedy(problem: Problem) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Deterministic, priority-aware greedy fallback with:
      - Category caps, toxin penalty.
      - NEW: Strict *avoidance* of overshooting any already-met min when alternatives exist.
             We prefer foods where a chunk does NOT push any min above its target.
             Only if *all* candidates would overshoot do we allow it (then we still cap
             the added grams to keep overshoot minimal).

    Scoring per kcal (using per-100g data):
        score(food) = sum_n (w[n]*per100[n]) / kcal_per_100g
                       - toxin_penalty * (sum_t per100[toxin]) / (kcal_per_100g * 1000)
    """
    foods = problem.foods
    mins = problem.targets.mins
    maxes = problem.targets.maxes
    kcal_cap = problem.targets.kcal_remaining
    weights = problem.priorities or {k: 1.0 for k in mins}
    toxin_penalty = problem.toxin_penalty
    cat_caps = problem.category_caps_g or {}

    CHUNK_G = 25.0

    plan: Dict[str, float] = {f.name: 0.0 for f in foods}
    totals: Dict[str, float] = {"kcal": 0.0}
    exhausted: Dict[str, bool] = {f.name: False for f in foods}

    def unmet_exist() -> bool:
        return any(totals.get(k, 0.0) + 1e-9 < need for k, need in mins.items())

    def kcal_remaining() -> float:
        return max(0.0, kcal_cap - totals.get("kcal", 0.0))

    def category_used_g(cat: str) -> float:
        c = (cat or "").strip().lower()
        return sum(plan[f.name] for f in foods if (f.category or "").strip().lower() == c)

    def max_add_by_caps(f: Food) -> float:
        """Hard caps from servings, kcal, toxins, category."""
        rem_serv = max(0.0, f.max_serving_g - plan[f.name])
        if rem_serv <= 0:
            return 0.0
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
        # category caps
        if cat_caps:
            cat = (f.category or "").strip().lower()
            if cat in cat_caps:
                remaining_cat = max(0.0, cat_caps[cat] - category_used_g(cat))
                g_cap = min(g_cap, remaining_cat)
        return max(0.0, g_cap)

    def non_overshoot_cap(f: Food) -> float:
        """
        Max grams we can add without pushing *any* min beyond its need.
        If the food adds to a nutrient that's already >= need, this returns 0.
        """
        cap = float("inf")
        for k, need in mins.items():
            per_g = f.per100.get(k, 0.0) / 100.0
            if per_g <= 0:
                continue
            head = need - totals.get(k, 0.0)
            if head <= 1e-12:
                # Already at/over need; any positive addition would overshoot this nutrient
                return 0.0
            cap = min(cap, head / per_g)
        return 0.0 if cap == float("inf") else max(0.0, cap)

    def score_food(f: Food) -> float:
        kcal100 = max(f.kcal_per_100g, 1e-6)
        # Benefit only from unmet mins
        benefit = 0.0
        for n, w in weights.items():
            if mins.get(n, 0.0) > totals.get(n, 0.0) + 1e-9:
                benefit += w * f.per100.get(n, 0.0)
        # Toxin cost (independent of distance to caps)
        toxin_sum = sum(f.per100.get(t, 0.0) for t in maxes.keys())
        toxin_cost = toxin_penalty * (toxin_sum / (kcal100 * 1000.0))
        return (benefit / kcal100) - toxin_cost

    for _ in range(10000):
        if not unmet_exist() or kcal_remaining() <= 1e-9:
            break

        # Build candidates and classify as CLEAN (no overshoot possible for a chunk) vs DIRTY
        clean: List[Tuple[float, float, Food]] = []  # (score, add_cap, food)
        dirty: List[Tuple[float, float, Food]] = []

        for f in foods:
            if exhausted[f.name]:
                continue
            # Must help at least one unmet nutrient
            if not any(f.per100.get(n, 0.0) > 0 and mins.get(n, 0.0) > totals.get(n, 0.0) + 1e-9 for n in mins):
                continue

            hard_cap = max_add_by_caps(f)
            if hard_cap <= 1e-9:
                exhausted[f.name] = True
                continue

            no_cap = non_overshoot_cap(f)  # cap to avoid exceeding any need
            # Proposed addition without overshoot
            add_clean = min(CHUNK_G, hard_cap, no_cap) if no_cap > 0 else 0.0
            s = score_food(f)

            if add_clean > 1e-9:
                clean.append((s, add_clean, f))
            else:
                # If we can't add even a small amount without overshoot, classify as dirty
                # We'll only consider these if there are no clean candidates.
                # Also cap "dirty" add to the minimal overshoot to progress.
                # Compute minimal grams to progress at least one unmet nutrient but keep overshoot tiny.
                add_dirty = min(CHUNK_G, hard_cap)
                if add_dirty > 1e-9:
                    dirty.append((s, add_dirty, f))

        picked_list = clean if clean else dirty
        if not picked_list:
            break

        # Deterministic ordering: highest score, tie by name
        picked_list.sort(key=lambda x: (-x[0], x[2].name))

        progressed = False
        for s, add_cap, f in picked_list:
            add = add_cap
            if add <= 1e-9:
                exhausted[f.name] = True
                continue
            # Apply addition
            plan[f.name] += add
            totals["kcal"] = totals.get("kcal", 0.0) + add * (f.kcal_per_100g / 100.0)
            for k, v in f.per100.items():
                totals[k] = totals.get(k, 0.0) + add * (v / 100.0)
            progressed = True
            break

        if not progressed:
            break

    plan = _omit_near_zero(plan)
    return plan, totals

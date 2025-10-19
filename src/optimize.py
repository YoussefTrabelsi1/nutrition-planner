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
    Deterministic, priority-aware greedy that **does not overshoot targets**,
    unless *every* available food would overshoot (then allows tiny 2% wiggle).

    Core tweaks vs previous version:
      1) **Deficit-weighted scoring per kcal**:
         benefit_per_g = Σ_n w[n] * min(deficit[n], per_g[n])
         score = benefit_per_g / kcal_per_g - toxin_penalty * toxin_per_g
      2) **Tight add cap** to avoid overshooting unmet nutrients:
         g_def_cap = min_n ( deficit[n] / per_g[n] ) over n with deficit>0 and per_g[n]>0
      3) **Small overshoot tolerance** (2%) for already-met nutrients to avoid dead-ends.
      4) Respects kcal, serving, toxin, and optional category caps.
    """
    foods = problem.foods
    mins = problem.targets.mins
    maxes = problem.targets.maxes
    kcal_cap = problem.targets.kcal_remaining
    weights = problem.priorities or {k: 1.0 for k in mins}
    toxin_penalty = problem.toxin_penalty
    cat_caps = problem.category_caps_g or {}

    CHUNK_G = 25.0
    MET_OVERSHOOT_TOL = 0.02  # allow up to +2% on nutrients already met only if no clean option exists

    plan: Dict[str, float] = {f.name: 0.0 for f in foods}
    totals: Dict[str, float] = {"kcal": 0.0}
    exhausted: Dict[str, bool] = {f.name: False for f in foods}

    def kcal_remaining() -> float:
        return max(0.0, kcal_cap - totals.get("kcal", 0.0))

    def deficits() -> Dict[str, float]:
        return {k: max(0.0, need - totals.get(k, 0.0)) for k, need in mins.items()}

    def unmet_exist() -> bool:
        return any(v > 1e-9 for v in deficits().values())

    def category_used_g(cat: str) -> float:
        c = (cat or "").strip().lower()
        return sum(plan[f.name] for f in foods if (f.category or "").strip().lower() == c)

    def hard_caps_for_food(f: Food) -> float:
        """Max grams permitted by serving, kcal, toxin caps, and category caps."""
        rem_serv = max(0.0, f.max_serving_g - plan[f.name])
        if rem_serv <= 0:
            return 0.0
        kpg = max(f.kcal_per_100g / 100.0, 1e-9)
        rem_kcal_g = kcal_remaining() / kpg if kpg > 0 else rem_serv
        g_cap = min(rem_serv, rem_kcal_g)
        # toxins
        for t, cap in maxes.items():
            per_g = f.per100.get(t, 0.0) / 100.0
            if per_g <= 0:
                continue
            head = max(0.0, cap - totals.get(t, 0.0))
            if head <= 1e-12:
                return 0.0
            g_cap = min(g_cap, head / per_g)
        # category
        if cat_caps:
            cat = (f.category or "").strip().lower()
            if cat in cat_caps:
                remaining_cat = max(0.0, cat_caps[cat] - category_used_g(cat))
                g_cap = min(g_cap, remaining_cat)
        return max(0.0, g_cap)

    def g_cap_unmet(f: Food, d: Dict[str, float]) -> float:
        """Cap to avoid overshooting any **unmet** nutrient."""
        cap = float("inf")
        has_binding = False
        for k, deficit in d.items():
            if deficit <= 1e-12:
                continue
            per_g = f.per100.get(k, 0.0) / 100.0
            if per_g <= 0:
                continue
            cap = min(cap, deficit / per_g)
            has_binding = True
        if not has_binding:
            return 0.0  # this food doesn't help any unmet nutrient
        return max(0.0, cap)

    def g_cap_met(f: Food, tol: float) -> float:
        """Cap to avoid pushing any **already-met** nutrient beyond (1+tol)*need."""
        cap = float("inf")
        touched = False
        for k, need in mins.items():
            per_g = f.per100.get(k, 0.0) / 100.0
            if per_g <= 0:
                continue
            have = totals.get(k, 0.0)
            if have + 1e-12 >= need:  # met or above
                limit = (need * (1.0 + tol) - have) / per_g
                cap = min(cap, max(0.0, limit))
                touched = True
        if not touched:
            return float("inf")
        return cap

    def score_food(f: Food, d: Dict[str, float]) -> float:
        """Deficit-weighted marginal value per kcal minus toxin per kcal."""
        kcal_per_g = max(f.kcal_per_100g / 100.0, 1e-9)
        # Weighted benefit per gram with saturation at deficit
        benefit_per_g = 0.0
        for n, w in weights.items():
            if d.get(n, 0.0) <= 1e-12:
                continue
            per_g = f.per100.get(n, 0.0) / 100.0
            if per_g <= 0:
                continue
            benefit_per_g += w * min(d[n], per_g)
        toxin_per_g = 0.0
        for t in maxes.keys():
            toxin_per_g += f.per100.get(t, 0.0) / 100.0
        return (benefit_per_g / kcal_per_g) - toxin_penalty * (toxin_per_g / max(kcal_per_g, 1e-9))

    for _ in range(10000):
        if not unmet_exist() or kcal_remaining() <= 1e-9:
            break
        d = deficits()

        # Build candidate (score, clean_add_cap, dirty_add_cap, food)
        candidates: List[Tuple[float, float, float, Food]] = []
        for f in foods:
            if exhausted[f.name]:
                continue
            hard_cap = hard_caps_for_food(f)
            if hard_cap <= 1e-9:
                exhausted[f.name] = True
                continue
            cap_unmet = g_cap_unmet(f, d)
            if cap_unmet <= 1e-9:
                # doesn't help any unmet nutrient
                continue
            # "clean" cap: don't exceed any **met** nutrient (tol=0)
            cap_met_clean = g_cap_met(f, tol=0.0)
            clean_cap = min(CHUNK_G, hard_cap, cap_unmet, cap_met_clean)
            # "dirty" (tiny wiggle): allow +2% on met nutrients if needed
            cap_met_dirty = g_cap_met(f, tol=MET_OVERSHOOT_TOL)
            dirty_cap = min(CHUNK_G, hard_cap, cap_unmet, cap_met_dirty)
            sc = score_food(f, d)
            candidates.append((sc, max(0.0, clean_cap), max(0.0, dirty_cap), f))

        if not candidates:
            break

        # Prefer CLEAN additions first; if no clean can progress, allow DIRTY
        # Sort deterministically by score desc, then name asc
        candidates.sort(key=lambda x: (-x[0], x[3].name))

        progressed = False
        # Try clean
        for sc, clean_cap, dirty_cap, f in candidates:
            add = clean_cap
            if add > 1e-9:
                plan[f.name] += add
                totals["kcal"] = totals.get("kcal", 0.0) + add * (f.kcal_per_100g / 100.0)
                for k, v in f.per100.items():
                    totals[k] = totals.get(k, 0.0) + add * (v / 100.0)
                progressed = True
                break

        # If no clean move, try tiny dirty (≤2% overshoot on already met mins)
        if not progressed:
            for sc, clean_cap, dirty_cap, f in candidates:
                add = dirty_cap
                if add > 1e-9:
                    plan[f.name] += add
                    totals["kcal"] = totals.get("kcal", 0.0) + add * (f.kcal_per_100g / 100.0)
                    for k, v in f.per100.items():
                        totals[k] = totals.get(k, 0.0) + add * (v / 100.0)
                    progressed = True
                    break

        if not progressed:
            # mark top candidate as exhausted to avoid infinite loop
            exhausted[candidates[0][3].name] = True
            continue

    plan = _omit_near_zero(plan)
    return plan, totals
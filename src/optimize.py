from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from models import Problem, Food

def _omit_near_zero(plan: Dict[str, float], eps: float = 1e-6) -> Dict[str, float]:
    return {k: v for k, v in plan.items() if v > eps}

def compute_totals(problem: Problem, plan: Dict[str, float]) -> Dict[str, float]:
    totals: Dict[str, float] = {"kcal": 0.0, "cost": 0.0}
    for f in problem.foods:
        g = plan.get(f.name, 0.0)
        if g <= 0:
            continue
        totals["kcal"] += g * (f.kcal_per_100g / 100.0)
        if f.price_per_100g is not None:
            totals["cost"] += g * (f.price_per_100g / 100.0)
        for k, v in f.per100.items():
            totals[k] = totals.get(k, 0.0) + g * (v / 100.0)
    return totals

# ---------- LP: add secondary (cheapest) stage when feasible ----------
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

def _solve_lp_core(problem: Problem):
    import pulp  # type: ignore
    foods = problem.foods
    targets = problem.targets
    x = {f.name: pulp.LpVariable(f"g_{f.name}", lowBound=0,
                                 upBound=min(f.max_serving_g, f.daily_max_g if f.daily_max_g is not None else f.max_serving_g))
         for f in foods}
    calories_expr = pulp.lpSum([x[f.name] * (f.kcal_per_100g / 100.0) for f in foods])
    model = pulp.LpProblem("nutrition_min_kcal", pulp.LpMinimize)
    model += calories_expr
    model += calories_expr <= targets.kcal_remaining
    for key, need in targets.mins.items():
        model += pulp.lpSum([x[f.name] * (f.per100.get(key, 0.0) / 100.0) for f in foods]) >= need
    for key, cap in targets.maxes.items():
        model += pulp.lpSum([x[f.name] * (f.per100.get(key, 0.0) / 100.0) for f in foods]) <= cap
    if problem.category_caps_g:
        _add_category_caps_lp(model, x, foods, problem.category_caps_g)
    return model, x, calories_expr

def _extract_plan(foods: List[Food], x) -> Dict[str, float]:
    return {f.name: float(x[f.name].value() or 0.0) for f in foods}

def solve_lp_only(problem: Problem, allow_soft: bool = False) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    try:
        import pulp  # type: ignore
    except Exception:
        # (soft unchanged here)
        return None

    # Stage 1: minimize calories
    model, x, calories_expr = _solve_lp_core(problem)
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[model.status] != "Optimal":
        return None

    best_cal = float(pulp.value(calories_expr))
    plan = _extract_plan(problem.foods, x)

    # Stage 2: minimize cost subject to calories <= best_cal (+tiny epsilon)
    # If no prices are provided at all, skip stage 2.
    if any(f.price_per_100g is not None for f in problem.foods):
        cost_model, x2, calories_expr2 = _solve_lp_core(problem)
        epsilon = 1e-6
        cost_model += calories_expr2 <= best_cal + epsilon
        cost_expr = pulp.lpSum([
            x2[f.name] * ((f.price_per_100g or 0.0) / 100.0) for f in problem.foods
        ])
        cost_model.sense = pulp.LpMinimize
        cost_model.setObjective(cost_expr)
        cost_model.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[cost_model.status] == "Optimal":
            plan = _extract_plan(problem.foods, x2)

    plan = _omit_near_zero(plan)
    totals = compute_totals(problem, plan)
    return plan, totals

# ---------- Greedy: tie-breaker by cheaper price/kcal ----------
def solve_greedy(problem: Problem) -> Tuple[Dict[str, float], Dict[str, float]]:
    foods = problem.foods
    mins = problem.targets.mins
    maxes = problem.targets.maxes
    kcal_cap = problem.targets.kcal_remaining
    base_w = problem.priorities or {k: 1.0 for k in mins}
    min_w = min(base_w.values()) if base_w else 0.2
    toxin_penalty = problem.toxin_penalty
    cat_caps = problem.category_caps_g or {}

    CHUNK_G = 25.0

    plan: Dict[str, float] = {f.name: 0.0 for f in foods}
    totals: Dict[str, float] = {"kcal": 0.0, "cost": 0.0}
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

    def hard_cap_for_food(f: Food) -> float:
        rem_serv = max(0.0, f.max_serving_g - plan[f.name])
        if rem_serv <= 0:
            return 0.0
        if f.daily_max_g is not None:
            rem_serv = min(rem_serv, max(0.0, f.daily_max_g - plan[f.name]))
            if rem_serv <= 0:
                return 0.0
        kpg = max(f.kcal_per_100g / 100.0, 1e-9)
        g_cap = min(rem_serv, kcal_remaining() / kpg)
        for t, cap in maxes.items():
            per_g = f.per100.get(t, 0.0) / 100.0
            if per_g <= 0:
                continue
            head = max(0.0, cap - totals.get(t, 0.0))
            if head <= 1e-12:
                return 0.0
            g_cap = min(g_cap, head / per_g)
        if cat_caps:
            cat = (f.category or "").strip().lower()
            if cat in cat_caps:
                remaining_cat = max(0.0, cat_caps[cat] - category_used_g(cat))
                g_cap = min(g_cap, remaining_cat)
        return max(0.0, g_cap)

    def dynamic_weights(d: Dict[str, float]) -> Dict[str, float]:
        w: Dict[str, float] = {}
        for n in mins.keys():
            w[n] = base_w.get(n, min_w) if d.get(n, 0.0) > 1e-12 else min_w
        return w

    def score_food(f: Food, d: Dict[str, float], w: Dict[str, float]) -> float:
        kcal_per_g = max(f.kcal_per_100g / 100.0, 1e-9)
        benefit_per_g = 0.0
        for n, wn in w.items():
            per_g = f.per100.get(n, 0.0) / 100.0
            if per_g <= 0:
                continue
            need_left = d.get(n, 0.0)
            if need_left <= 0:
                continue
            benefit_per_g += wn * min(need_left, per_g)
        toxin_per_g = sum(f.per100.get(t, 0.0) / 100.0 for t in maxes.keys())
        value = (benefit_per_g / kcal_per_g) - toxin_penalty * (toxin_per_g / kcal_per_g)
        return value

    def price_per_kcal(f: Food) -> float:
        if f.price_per_100g is None or f.kcal_per_100g <= 0:
            return float("inf") if f.kcal_per_100g <= 0 else 0.0
        # (price/100g) / (kcal/100g) = price per kcal
        return (f.price_per_100g / 100.0) / (f.kcal_per_100g / 100.0)

    for _ in range(10000):
        if not unmet_exist() or kcal_remaining() <= 1e-9:
            break
        d = deficits()
        w = dynamic_weights(d)

        candidates: List[Tuple[float, float, float, Food]] = []
        for f in foods:
            if exhausted[f.name]:
                continue
            if not any((f.per100.get(n, 0.0) > 0 and d.get(n, 0.0) > 1e-12) for n in mins.keys()):
                continue
            g_cap = min(CHUNK_G, hard_cap_for_food(f))
            if g_cap <= 1e-9:
                exhausted[f.name] = True
                continue
            s = score_food(f, d, w)
            ppk = price_per_kcal(f)
            candidates.append((s, ppk, g_cap, f))

        if not candidates:
            break

        # Sort by: score desc, then cheaper price/kcal asc, then name asc
        candidates.sort(key=lambda x: (-x[0], x[1], x[3].name))

        progressed = False
        for s, ppk, g_cap, f in candidates:
            add = g_cap
            if add <= 1e-9:
                exhausted[f.name] = True
                continue
            plan[f.name] += add
            totals["kcal"] += add * (f.kcal_per_100g / 100.0)
            if f.price_per_100g is not None:
                totals["cost"] += add * (f.price_per_100g / 100.0)
            for k, v in f.per100.items():
                totals[k] = totals.get(k, 0.0) + add * (v / 100.0)
            progressed = True
            break

        if not progressed:
            break

    plan = _omit_near_zero(plan)
    totals = compute_totals(problem, plan)
    return plan, totals

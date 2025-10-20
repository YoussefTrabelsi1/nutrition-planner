from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from models import Problem, Food
import random
import math


def _omit_near_zero(plan: Dict[str, float], eps: float = 1e-6) -> Dict[str, float]:
    return {k: v for k, v in plan.items() if v > eps}


def compute_totals(problem: Problem, plan: Dict[str, float]) -> Dict[str, float]:
    totals: Dict[str, float] = {"kcal": 0.0, "cost": 0.0}
    for f in problem.foods:
        g = plan.get(f.name, 0.0)
        if g <= 0:
            continue
        totals["kcal"] += g * (f.kcal_per_100g / 100.0)
        if getattr(f, "price_per_100g", None) is not None:
            totals["cost"] += g * (float(f.price_per_100g) / 100.0)
        for k, v in f.per100.items():
            totals[k] = totals.get(k, 0.0) + g * (v / 100.0)
    return totals


# ---------- LP (now supports per-food increments & all-or-nothing packs) ----------

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


def _solve_lp_core(problem: Problem,
                   rng: random.Random | None,
                   special_cats: set[str] | None):
    import pulp  # type: ignore
    foods = problem.foods
    targets = problem.targets

    # Decision variables:
    # - binary_pack foods:   y_f ∈ {0,1}, grams = unit_size_g * y_f
    # - incremented foods:   n_f ∈ Integers≥0, grams = increment_g * n_f
    # - others:               x_f ∈ [0, upBound] grams
    y = {}
    n = {}
    x = {}

    def up_bound_grams(f: Food) -> float:
        ub = f.max_serving_g
        if getattr(f, "daily_max_g", None) is not None:
            ub = min(ub, float(f.daily_max_g))
        return ub

    # Create variables
    for f in foods:
        ub = up_bound_grams(f)
        if f.binary_pack and f.unit_size_g and f.unit_size_g > 0:
            # Single pack: allow 0 or 1 only. If pack exceeds ub, it can never be chosen.
            y[f.name] = pulp.LpVariable(f"y_{f.name}", lowBound=0, upBound=1, cat="Binary")
        elif f.increment_g and f.increment_g > 0:
            # Integer multiples
            max_units = int(math.floor(ub / float(f.increment_g))) if ub < float("inf") else None
            if max_units is not None and max_units < 0:
                max_units = 0
            if max_units is None:
                n[f.name] = pulp.LpVariable(f"n_{f.name}", lowBound=0, cat="Integer")
            else:
                n[f.name] = pulp.LpVariable(f"n_{f.name}", lowBound=0, upBound=max_units, cat="Integer")
        else:
            x[f.name] = pulp.LpVariable(f"g_{f.name}", lowBound=0, upBound=ub)

    model = pulp.LpProblem("nutrition_min_kcal", pulp.LpMinimize)

    # Helper: grams expression for each food
    def grams_var(f: Food):
        if f.name in y:
            return y[f.name] * float(f.unit_size_g or 0.0)
        if f.name in n:
            return n[f.name] * float(f.increment_g or 0.0)
        return x[f.name]

    # Objective with tiny jitter for variety
    def cal_per_g(f: Food) -> float:
        base = f.kcal_per_100g / 100.0
        if rng is None:
            return base
        jitter = 1.0 + (rng.random() - 0.5) * 0.01
        return base * jitter

    calories_expr = pulp.lpSum([grams_var(f) * cal_per_g(f) for f in foods])
    model += calories_expr

    # Hard kcal cap uses exact calories, no jitter
    model += pulp.lpSum([grams_var(f) * (f.kcal_per_100g / 100.0) for f in foods]) <= targets.kcal_remaining

    # Min/Max constraints
    for key, need in targets.mins.items():
        model += pulp.lpSum([grams_var(f) * (f.per100.get(key, 0.0) / 100.0) for f in foods]) >= need
    for key, cap in targets.maxes.items():
        model += pulp.lpSum([grams_var(f) * (f.per100.get(key, 0.0) / 100.0) for f in foods]) <= cap

    # Category caps (optional)
    if problem.category_caps_g:
        # Build a proxy x_vars map using grams expressions
        x_proxy = {f.name: grams_var(f) for f in foods}
        _add_category_caps_lp(model, x_proxy, foods, problem.category_caps_g)

    # Special-category (meat/fish/fish_canned) 1..3 items via binaries:
    if special_cats:
        y_pick = []
        for f in foods:
            if (f.category or "").strip().lower() in special_cats:
                # If not binary_pack already, create a selection binary to count distinct items
                sel = pulp.LpVariable(f"zsel_{f.name}", lowBound=0, upBound=1, cat="Binary")
                y_pick.append(sel)
                # link selection to grams: grams_var(f) <= ub * sel
                ub = up_bound_grams(f)
                model += grams_var(f) <= ub * sel
        if y_pick:
            model += pulp.lpSum(y_pick) >= 1
            model += pulp.lpSum(y_pick) <= 3

    return model, grams_var, calories_expr


def _extract_plan_from_solution(foods: List[Food], grams_expr) -> Dict[str, float]:
    plan: Dict[str, float] = {}
    for f in foods:
        g_val = float(grams_expr(f).value() or 0.0)
        if g_val > 0:
            plan[f.name] = g_val
    return plan


def solve_lp_only(problem: Problem,
                  allow_soft: bool = False,
                  seed: Optional[int] = None,
                  special_categories: Optional[List[str]] = None
                  ) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    try:
        import pulp  # type: ignore
    except Exception:
        return None

    rng = random.Random(seed) if seed is not None else random.Random()
    specials = set((special_categories or ["meat", "fish", "fish_canned"]))
    specials = {s.strip().lower() for s in specials}

    model, grams_var, calories_expr = _solve_lp_core(problem, rng, specials)
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[model.status] != "Optimal":
        return None

    best_cal = float(pulp.value(calories_expr))
    plan = _extract_plan_from_solution(problem.foods, grams_var)

    # Second stage: minimize cost at fixed calories (if prices present)
    if any(getattr(f, "price_per_100g", None) is not None for f in problem.foods):
        model2, grams_var2, calories_expr2 = _solve_lp_core(problem, rng, specials)
        epsilon = 1e-6
        model2 += calories_expr2 <= best_cal + epsilon
        cost_terms = []
        for f in problem.foods:
            price = float(getattr(f, "price_per_100g", 0.0) or 0.0) / 100.0
            jitter = 1.0 + (rng.random() - 0.5) * 0.01
            cost_terms.append(grams_var2(f) * (price * jitter))
        model2.sense = pulp.LpMinimize
        model2.setObjective(sum(cost_terms))
        model2.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[model2.status] == "Optimal":
            plan = _extract_plan_from_solution(problem.foods, grams_var2)

    plan = _omit_near_zero(plan)
    totals = compute_totals(problem, plan)
    return plan, totals


# ---------- Greedy (respect increments & all-or-nothing packs) ----------

def solve_greedy(problem: Problem,
                 seed: Optional[int] = None,
                 special_categories: Optional[List[str]] = None
                 ) -> Tuple[Dict[str, float], Dict[str, float]]:
    rng = random.Random(seed) if seed is not None else random.Random()
    specials = set((special_categories or ["meat", "fish", "fish_canned"]))
    specials = {s.strip().lower() for s in specials}

    foods = problem.foods
    mins = problem.targets.mins
    maxes = problem.targets.maxes
    kcal_cap = problem.targets.kcal_remaining
    base_w = problem.priorities or {k: 1.0 for k in mins}
    min_w = min(base_w.values()) if base_w else 0.2
    toxin_penalty = problem.toxin_penalty
    cat_caps = problem.category_caps_g or {}

    plan: Dict[str, float] = {f.name: 0.0 for f in foods}
    totals: Dict[str, float] = {"kcal": 0.0, "cost": 0.0}
    exhausted: Dict[str, bool] = {f.name: False for f in foods}
    special_included: set[str] = set()

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
        rem = max(0.0, f.max_serving_g - plan[f.name])
        if getattr(f, "daily_max_g", None) is not None:
            rem = min(rem, max(0.0, float(f.daily_max_g) - plan[f.name]))
        if rem <= 0:
            return 0.0
        kpg = max(f.kcal_per_100g / 100.0, 1e-9)
        g_cap = min(rem, kcal_remaining() / kpg)
        # toxins
        for t, cap in maxes.items():
            per_g = f.per100.get(t, 0.0) / 100.0
            if per_g > 0:
                head = max(0.0, cap - totals.get(t, 0.0))
                if head <= 1e-12:
                    return 0.0
                g_cap = min(g_cap, head / per_g)
        # category caps
        if cat_caps:
            cat = (f.category or "").strip().lower()
            if cat in cat_caps:
                g_cap = min(g_cap, max(0.0, cat_caps[cat] - category_used_g(cat)))
        return max(0.0, g_cap)

    def dynamic_weights(d: Dict[str, float]) -> Dict[str, float]:
        return {n: (base_w.get(n, min_w) if d.get(n, 0.0) > 1e-12 else min_w) for n in mins.keys()}

    def price_per_kcal(f: Food) -> float:
        if getattr(f, "price_per_100g", None) is None or f.kcal_per_100g <= 0:
            return float("inf") if f.kcal_per_100g <= 0 else 0.0
        return (float(f.price_per_100g) / 100.0) / (f.kcal_per_100g / 100.0)

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
        base_value = (benefit_per_g / kcal_per_g) - toxin_penalty * (toxin_per_g / kcal_per_g)
        jitter = 1.0 + (rng.random() - 0.5) * 0.02
        return base_value * jitter

    for _ in range(10000):
        if not unmet_exist() or kcal_remaining() <= 1e-9:
            break
        d = deficits()
        w = dynamic_weights(d)

        candidates: List[Tuple[float, float, float, float, Food]] = []
        for f in foods:
            if exhausted[f.name]:
                continue
            # must help at least one unmet nutrient
            if not any((f.per100.get(n, 0.0) > 0 and d.get(n, 0.0) > 1e-12) for n in mins.keys()):
                continue

            f_cat = (f.category or "").strip().lower()
            # special categories: no more than 3 distinct items
            if f_cat in specials and f.name not in special_included and len(special_included) >= 3:
                continue

            g_cap = hard_cap_for_food(f)
            if g_cap <= 1e-9:
                exhausted[f.name] = True
                continue

            # per-food increment / pack logic
            step = float(f.unit_size_g or f.increment_g or 25.0)
            if f.binary_pack and f.unit_size_g:
                # allow only one unit if not already selected
                if plan[f.name] > 0:
                    exhausted[f.name] = True
                    continue
                add = f.unit_size_g if g_cap + 1e-9 >= f.unit_size_g else 0.0
            else:
                # multiples of step; prefer exactly one step at a time
                add = step if g_cap + 1e-9 >= step else 0.0

            if add <= 1e-9:
                continue

            s = score_food(f, d, w)
            ppk = price_per_kcal(f)
            candidates.append((s, ppk, add, plan.get(f.name, 0.0), f))

        if not candidates:
            break

        # Prefer: score desc, cheaper price/kcal asc, already-started first, name asc
        candidates.sort(key=lambda x: (-x[0], x[1], -x[3], x[4].name))

        progressed = False
        for s, ppk, add, current_g, f in candidates:
            if add <= 1e-9:
                continue
            plan[f.name] = plan.get(f.name, 0.0) + add
            totals["kcal"] += add * (f.kcal_per_100g / 100.0)
            if getattr(f, "price_per_100g", None) is not None:
                totals["cost"] += add * (float(f.price_per_100g) / 100.0)
            for k, v in f.per100.items():
                totals[k] = totals.get(k, 0.0) + add * (v / 100.0)
            # mark special inclusion
            if (f.category or "").strip().lower() in specials and plan[f.name] > 0:
                special_included.add(f.name)
            # for binary pack, don't pick again
            if f.binary_pack:
                exhausted[f.name] = True
            progressed = True
            break

        if not progressed:
            break

    plan = _omit_near_zero(plan)
    totals = compute_totals(problem, plan)
    return plan, totals

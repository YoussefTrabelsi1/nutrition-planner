from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from models import Problem, Food
import random


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


# ---------------- LP (add variety and special-category cardinality) ----------------

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

    # Bounds incorporate daily_max_g if present
    def up_bound(f: Food) -> float:
        m = f.max_serving_g
        if getattr(f, "daily_max_g", None) is not None:
            m = min(m, float(f.daily_max_g))
        return m

    x = {f.name: pulp.LpVariable(f"g_{f.name}", lowBound=0, upBound=up_bound(f)) for f in foods}
    model = pulp.LpProblem("nutrition_min_kcal", pulp.LpMinimize)

    # Primary objective: minimize calories (with tiny random jitter for variety)
    def ccoef(f: Food) -> float:
        base = f.kcal_per_100g / 100.0
        if rng is None:
            return base
        # up to ±0.5% jitter (does not affect feasibility, just tie-breaking)
        jitter = 1.0 + (rng.random() - 0.5) * 0.01
        return base * jitter

    calories_expr = pulp.lpSum([x[f.name] * ccoef(f) for f in foods])
    model += calories_expr

    # Kcal cap
    model += pulp.lpSum([x[f.name] * (f.kcal_per_100g / 100.0) for f in foods]) <= targets.kcal_remaining

    # Mins / Maxes
    for key, need in targets.mins.items():
        model += pulp.lpSum([x[f.name] * (f.per100.get(key, 0.0) / 100.0) for f in foods]) >= need
    for key, cap in targets.maxes.items():
        model += pulp.lpSum([x[f.name] * (f.per100.get(key, 0.0) / 100.0) for f in foods]) <= cap

    # Optional category caps
    if problem.category_caps_g:
        _add_category_caps_lp(model, x, foods, problem.category_caps_g)

    # NEW (Ticket — variety rule): cardinality on special categories via binaries
    # Require between 1 and 3 *items* chosen from categories in special_cats.
    if special_cats:
        y: Dict[str, "pulp.LpVariable"] = {}
        for f in foods:
            if (f.category or "").strip().lower() in special_cats:
                y[f.name] = pulp.LpVariable(f"y_{f.name}", cat="Binary")
                # link: x_f <= U_f * y_f
                model += x[f.name] <= up_bound(f) * y[f.name]
        if y:
            # At least 1, at most 3
            model += pulp.lpSum(list(y.values())) >= 1
            model += pulp.lpSum(list(y.values())) <= 3

    return model, x, calories_expr


def _extract_plan(foods: List[Food], x) -> Dict[str, float]:
    return {f.name: float(x[f.name].value() or 0.0) for f in foods}


def solve_lp_only(problem: Problem,
                  allow_soft: bool = False,
                  seed: Optional[int] = None,
                  special_categories: Optional[List[str]] = None
                  ) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """
    LP with:
      - tiny random jitter in calorie objective (variety),
      - optional cardinality constraint: include 1..3 items from special categories
        (meat, fish, fish_canned by default).
      - two-stage: min calories, then min cost at that calorie level (if prices available).
    """
    try:
        import pulp  # type: ignore
    except Exception:
        # Fallback handled by caller
        return None

    rng = random.Random(seed) if seed is not None else random.Random()
    specials = set((special_categories or ["meat", "fish", "fish_canned"]))
    specials = {s.strip().lower() for s in specials}

    # Stage 1: min calories (with jitter)
    model, x, calories_expr = _solve_lp_core(problem, rng, specials)
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[model.status] != "Optimal":
        return None

    best_cal = float(pulp.value(calories_expr))
    plan = _extract_plan(problem.foods, x)

    # Stage 2: min cost at fixed calories (epsilon)
    if any(getattr(f, "price_per_100g", None) is not None for f in problem.foods):
        model2, x2, calories_expr2 = _solve_lp_core(problem, rng, specials)
        epsilon = 1e-6
        model2 += calories_expr2 <= best_cal + epsilon
        # Cost objective (also with tiny jitter to diversify ties between equally priced foods)
        cost_terms = []
        for f in problem.foods:
            price = float(getattr(f, "price_per_100g", 0.0) or 0.0) / 100.0
            jitter = 1.0 + (rng.random() - 0.5) * 0.01
            cost_terms.append(x2[f.name] * (price * jitter))
        cost_expr = sum(cost_terms)
        model2.sense = pulp.LpMinimize
        model2.setObjective(cost_expr)
        model2.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[model2.status] == "Optimal":
            plan = _extract_plan(problem.foods, x2)

    plan = _omit_near_zero(plan)
    totals = compute_totals(problem, plan)
    return plan, totals


# ---------------- Greedy (variety + special-category rule) ----------------

def solve_greedy(problem: Problem,
                 seed: Optional[int] = None,
                 special_categories: Optional[List[str]] = None
                 ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Greedy with:
      - dynamic deprioritization (met mins get minimal weight),
      - per-food daily caps, toxin caps, kcal caps, optional category caps,
      - slight randomness in scoring to introduce variety across runs,
      - SPECIAL RULE: Among categories {meat, fish, fish_canned} include between 1 and 3 *items*.
        If 3 are already included, new items from these categories are prohibited,
        but increasing grams of already-selected ones is allowed.
    """
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

    CHUNK_G = 25.0

    plan: Dict[str, float] = {f.name: 0.0 for f in foods}
    totals: Dict[str, float] = {"kcal": 0.0, "cost": 0.0}
    exhausted: Dict[str, bool] = {f.name: False for f in foods}

    # Track which special-category items are included
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
        rem_serv = max(0.0, f.max_serving_g - plan[f.name])
        if rem_serv <= 0:
            return 0.0
        if getattr(f, "daily_max_g", None) is not None:
            rem_serv = min(rem_serv, max(0.0, float(f.daily_max_g) - plan[f.name]))
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
        # small random jitter ±1% to diversify choices across runs
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
            # Must help at least one unmet nutrient
            if not any((f.per100.get(n, 0.0) > 0 and d.get(n, 0.0) > 1e-12) for n in mins.keys()):
                continue

            # SPECIAL RULE: if f is in specials and is a *new* special item,
            # and we already have 3 specials selected, skip it.
            f_cat = (f.category or "").strip().lower()
            if f_cat in specials and f.name not in special_included and len(special_included) >= 3:
                continue

            g_cap = min(CHUNK_G, hard_cap_for_food(f))
            if g_cap <= 1e-9:
                exhausted[f.name] = True
                continue
            s = score_food(f, d, w)
            ppk = price_per_kcal(f)
            candidates.append((s, ppk, g_cap, plan.get(f.name, 0.0), f))

        if not candidates:
            break

        # Prefer: score desc, then cheaper price/kcal asc, then foods we already started (larger plan[g] first), then name
        candidates.sort(key=lambda x: (-x[0], x[1], -x[3], x[4].name))

        progressed = False
        for s, ppk, g_cap, current_g, f in candidates:
            add = g_cap
            if add <= 1e-9:
                exhausted[f.name] = True
                continue
            # apply
            plan[f.name] = plan.get(f.name, 0.0) + add
            totals["kcal"] += add * (f.kcal_per_100g / 100.0)
            if getattr(f, "price_per_100g", None) is not None:
                totals["cost"] += add * (float(f.price_per_100g) / 100.0)
            for k, v in f.per100.items():
                totals[k] = totals.get(k, 0.0) + add * (v / 100.0)
            # update specials set if this was a new special item
            if (f.category or "").strip().lower() in specials and plan[f.name] > 0:
                special_included.add(f.name)
            progressed = True
            break

        if not progressed:
            break

    # If we finished with 0 specials, try to include the best feasible special minimally
    if not any((f.category or "").strip().lower() in specials and plan.get(f.name, 0.0) > 0 for f in foods):
        # pick highest-score special with tiny add (up to CHUNK_G) if feasible
        d = deficits()
        w = dynamic_weights(d)
        special_cands: List[Tuple[float, Food]] = []
        for f in foods:
            if (f.category or "").strip().lower() not in specials:
                continue
            if hard_cap_for_food(f) <= 1e-9:
                continue
            special_cands.append((score_food(f, d, w), f))
        if special_cands:
            special_cands.sort(key=lambda x: (-x[0], x[1].name))
            f = special_cands[0][1]
            add = min(CHUNK_G, hard_cap_for_food(f))
            if add > 1e-9:
                plan[f.name] = plan.get(f.name, 0.0) + add
                totals["kcal"] += add * (f.kcal_per_100g / 100.0)
                if getattr(f, "price_per_100g", None) is not None:
                    totals["cost"] += add * (float(f.price_per_100g) / 100.0)
                for k, v in f.per100.items():
                    totals[k] = totals.get(k, 0.0) + add * (v / 100.0)

    plan = _omit_near_zero(plan)
    totals = compute_totals(problem, plan)
    return plan, totals

from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set
from models import (
    Food,
    Targets,
    Problem,
    FOODS_RE,
    BUDGET_MIN_RE,
    BUDGET_MAX_RE,
    REQUIRED_FOOD_BASE,
    UnitConflictError,
    SchemaError,
)
import yaml
from constraints import derive_priorities_from_ranks, normalize_to_range


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def _f(x: object, default: float = 0.0) -> float:
    if x is None:
        return default
    s = str(x).strip()
    return float(s) if s else default


def _validate_required_food_cols(headers: List[str]) -> None:
    missing = [c for c in REQUIRED_FOOD_BASE if c not in headers]
    if missing:
        raise SchemaError(f"Missing required food columns: {missing}")


def _detect_food_metrics(headers: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    seen_units: Dict[str, str] = {}
    header_to_key: Dict[str, str] = {}
    for h in headers:
        if h in REQUIRED_FOOD_BASE or h in (
            "daily_max_g",
            "price_per_100g",
            "increment_g",
            "unit_size_g",
            "unit_label",
            "binary_pack",
        ):
            continue
        if "_per_100g" in h:
            m = FOODS_RE.match(h)
            if not m:
                raise SchemaError(
                    f"Unrecognized nutrient/toxin column '{h}'; expected '<base>_(g|mg|mcg)_per_100g'."
                )
            base = m.group("base")
            unit = m.group("unit")
            if base in seen_units and seen_units[base] != unit:
                raise UnitConflictError(
                    f"Mixed units for '{base}': {seen_units[base]} and {unit}"
                )
            seen_units[base] = unit
            header_to_key[h] = f"{base}_{unit}"
        else:
            # ignore other metadata columns
            pass
    return seen_units, header_to_key


def _detect_budget_targets(headers: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    mins_units: Dict[str, str] = {}
    max_units: Dict[str, str] = {}
    for h in headers:
        if h == "kcal_remaining":
            continue
        m1 = BUDGET_MIN_RE.match(h)
        if m1:
            base, unit = m1.group("base"), m1.group("unit")
            if base in mins_units and mins_units[base] != unit:
                raise UnitConflictError(
                    f"Budget mixed units for min '{base}': {mins_units[base]} and {unit}"
                )
            mins_units[base] = unit
            continue
        m2 = BUDGET_MAX_RE.match(h)
        if m2:
            base, unit = m2.group("base"), m2.group("unit")
            if base in max_units and max_units[base] != unit:
                raise UnitConflictError(
                    f"Budget mixed units for max '{base}': {max_units[base]} and {unit}"
                )
            max_units[base] = unit
            continue
        raise SchemaError(
            f"Invalid budget column '{h}'; only '*_remaining', '*_limit', or 'kcal_remaining' are allowed."
        )
    return mins_units, max_units


def load_problem(
    foods_path: Path,
    budget_path: Path,
    priorities_path: Path | None = None,
    policies_path: Path | None = None,
) -> Problem:
    # Foods
    foods_rows = _read_csv(foods_path)
    if not foods_rows:
        raise SchemaError("foods.csv is empty")
    headers = list(foods_rows[0].keys())
    _validate_required_food_cols(headers)
    _, header_to_key = _detect_food_metrics(headers)

    foods: List[Food] = []
    all_keys_in_foods: Set[str] = set()
    for row in foods_rows:
        name = (row.get("name") or "").strip()
        kcal = _f(row.get("kcal_per_100g"), 0.0)
        maxg = _f(row.get("max_serving_g"), 0.0)
        category = (row.get("category") or "").strip()

        # Optional caps/prices & new increment/pack meta
        daily_max_g = None
        if (row.get("daily_max_g") or "").strip() != "":
            daily_max_g = _f(row.get("daily_max_g"), 0.0)
            if daily_max_g <= 0:
                raise SchemaError(f"daily_max_g must be > 0 if provided (food '{name}')")

        price_per_100g = None
        if (row.get("price_per_100g") or "").strip() != "":
            price_per_100g = _f(row.get("price_per_100g"), 0.0)
            if price_per_100g < 0:
                raise SchemaError(f"price_per_100g must be >= 0 if provided (food '{name}')")

        increment_g = None
        if (row.get("increment_g") or "").strip() != "":
            increment_g = _f(row.get("increment_g"), 0.0)
            if increment_g <= 0:
                raise SchemaError(f"increment_g must be > 0 if provided (food '{name}')")

        unit_size_g = None
        if (row.get("unit_size_g") or "").strip() != "":
            unit_size_g = _f(row.get("unit_size_g"), 0.0)
            if unit_size_g <= 0:
                raise SchemaError(f"unit_size_g must be > 0 if provided (food '{name}')")

        unit_label = (row.get("unit_label") or "").strip() or None

        binary_pack = False
        if (row.get("binary_pack") or "").strip() != "":
            val = (row.get("binary_pack") or "").strip().lower()
            binary_pack = val in ("1", "true", "yes", "y")

        if maxg <= 0 or kcal < 0:
            raise SchemaError(
                f"Invalid numeric in foods for '{name}': max_serving_g>0 and kcal>=0 required"
            )

        if binary_pack and not unit_size_g:
            raise SchemaError(f"binary_pack=1 requires unit_size_g for '{name}'")

        per100: Dict[str, float] = {}
        for h, key in header_to_key.items():
            val = row.get(h, "")
            if val == "" or val is None:
                continue
            x = _f(val, 0.0)
            if x < 0:
                raise SchemaError(f"Negative value not allowed in foods column '{h}' for '{name}'")
            per100[key] = x
            all_keys_in_foods.add(key)

        foods.append(Food(
            name=name,
            kcal_per_100g=kcal,
            max_serving_g=maxg,
            category=category,
            per100=per100,
            daily_max_g=daily_max_g,
            price_per_100g=price_per_100g,
            increment_g=increment_g,
            unit_size_g=unit_size_g,
            unit_label=unit_label,
            binary_pack=binary_pack,
        ))

    # Budget
    budget_rows = _read_csv(budget_path)
    if len(budget_rows) != 1:
        raise SchemaError("budget.csv must contain exactly one row")
    b_raw = budget_rows[0]
    b = {k: (v if v is not None else "") for k, v in b_raw.items() if k is not None}

    if "kcal_remaining" not in b:
        raise SchemaError("budget.csv missing 'kcal_remaining'")

    mins_units, max_units = _detect_budget_targets(list(b.keys()))
    kcal_remaining = _f(b.get("kcal_remaining"), 0.0)
    if kcal_remaining <= 0:
        raise SchemaError("kcal_remaining must be > 0")

    mins: Dict[str, float] = {}
    for base, unit in mins_units.items():
        col = f"{base}_{unit}_remaining"
        v = _f(b.get(col), 0.0)
        if v < 0:
            raise SchemaError(f"Negative remaining not allowed: {col}")
        mins[f"{base}_{unit}"] = v

    maxes: Dict[str, float] = {}
    for base, unit in max_units.items():
        col = f"{base}_{unit}_limit"
        v = _f(b.get(col), 0.0)
        if v < 0:
            raise SchemaError(f"Negative limit not allowed: {col}")
        maxes[f"{base}_{unit}"] = v

    # Warn about unused dimensions
    used_keys = set(mins.keys()) | set(maxes.keys())
    unused = sorted(k for k in all_keys_in_foods if k not in used_keys)
    for k in unused:
        print(f"Warning: '{k}' appears in foods.csv but not in budget.csv; it will not be targeted.")

    # Policies (optional) — unchanged passthrough
    category_caps_g: Dict[str, float] | None = None
    p_path = policies_path
    if p_path is None and priorities_path is not None:
        p_path = priorities_path.parent / "policies.yaml"
    if p_path and p_path.exists():
        with p_path.open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        caps = (y.get("per_category_daily_max_g") or {}) if isinstance(y, dict) else {}
        canonical: Dict[str, float] = {}
        for k, v in caps.items():
            try:
                cap = float(v)
            except Exception:
                continue
            if cap <= 0:
                continue
            canonical[str(k).strip().lower()] = cap
        category_caps_g = canonical if canonical else None

    # Priorities
    priorities: Dict[str, float] = {}
    toxin_penalty = 0.5
    if priorities_path and priorities_path.exists():
        with priorities_path.open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        raw_prio = y.get("priorities") or {}
        for k, v in raw_prio.items():
            try:
                w = float(v)
            except Exception:
                continue
            w = max(0.0, min(1.0, w))
            priorities[str(k)] = w
        if "toxin_penalty" in y:
            try:
                toxin_penalty = float(y["toxin_penalty"])
            except Exception:
                toxin_penalty = 0.5
        priorities = normalize_to_range(priorities, 0.2, 1.0)
    else:
        rank_map: Dict[str, int] = {k: i for i, k in enumerate(sorted(mins.keys()), start=1)}
        priorities = derive_priorities_from_ranks(rank_map)
        priorities = normalize_to_range(priorities, 0.2, 1.0)
        toxin_penalty = 0.5

    return Problem(
        foods=foods,
        targets=Targets(kcal_remaining=kcal_remaining, mins=mins, maxes=maxes),
        priorities=priorities,
        toxin_penalty=toxin_penalty,
        category_caps_g=category_caps_g,
    )

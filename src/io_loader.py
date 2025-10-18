from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Tuple
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

def _f(x: object, default: float = 0.0) -> float:
    """Coerce CSV cell to float, treating None/'' as default."""
    if x is None:
        return default
    s = str(x).strip()
    return float(s) if s else default

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def _validate_required_food_cols(headers: List[str]) -> None:
    missing = [c for c in REQUIRED_FOOD_BASE if c not in headers]
    if missing:
        raise SchemaError(f"Missing required food columns: {missing}")


def _detect_food_metrics(headers: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (base->unit) for nutrients/toxins and a map header->base_unit_key.
    base_unit_key keeps unit explicit to avoid clashes.
    """
    seen_units: Dict[str, str] = {}
    header_to_key: Dict[str, str] = {}
    for h in headers:
        m = FOODS_RE.match(h)
        if not m:
            continue
        base = m.group("base")
        unit = m.group("unit")
        if base in seen_units and seen_units[base] != unit:
            raise UnitConflictError(
                f"Mixed units for '{base}': {seen_units[base]} and {unit}"
            )
        seen_units[base] = unit
        header_to_key[h] = f"{base}_{unit}"
    return seen_units, header_to_key


def _detect_budget_targets(headers: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    mins_units: Dict[str, str] = {}
    max_units: Dict[str, str] = {}
    for h in headers:
        m1 = BUDGET_MIN_RE.match(h)
        if m1:
            base = m1.group("base")
            unit = m1.group("unit")
            if base in mins_units and mins_units[base] != unit:
                raise UnitConflictError(
                    f"Budget mixed units for min '{base}': {mins_units[base]} and {unit}"
                )
            mins_units[base] = unit
        m2 = BUDGET_MAX_RE.match(h)
        if m2:
            base = m2.group("base")
            unit = m2.group("unit")
            if base in max_units and max_units[base] != unit:
                raise UnitConflictError(
                    f"Budget mixed units for max '{base}': {max_units[base]} and {unit}"
                )
            max_units[base] = unit
    return mins_units, max_units


def load_problem(
    foods_path: Path,
    budget_path: Path,
    priorities_path: Path | None = None,
) -> Problem:
    # Foods
    foods_rows = _read_csv(foods_path)
    if not foods_rows:
        raise SchemaError("foods.csv is empty")
    headers = list(foods_rows[0].keys())
    _validate_required_food_cols(headers)
    _, header_to_key = _detect_food_metrics(headers)

    foods: List[Food] = []
    for row in foods_rows:
        name = row["name"].strip()
        kcal = float(row["kcal_per_100g"])
        maxg = float(row["max_serving_g"])
        category = row["category"].strip()
        if maxg <= 0 or kcal < 0:
            raise SchemaError(
                f"Invalid numeric in foods for '{name}': max_serving_g>0 and kcal>=0 required"
            )
        per100: Dict[str, float] = {}
        for h, key in header_to_key.items():
            val = row.get(h, "")
            if val == "" or val is None:
                continue
            x = float(val)
            if x < 0:
                raise SchemaError(f"Negative value not allowed in foods column '{h}' for '{name}'")
            per100[key] = x
        foods.append(Food(name=name, kcal_per_100g=kcal, max_serving_g=maxg, category=category, per100=per100))

    # Budget
    budget_rows = _read_csv(budget_path)
    if len(budget_rows) != 1:
        raise SchemaError("budget.csv must contain exactly one row")
    b_raw = budget_rows[0]

    # Drop None headers (can happen with trailing commas / short rows) and coerce None values
    b = {k: (v if v is not None else "") for k, v in b_raw.items() if k is not None}

    if "kcal_remaining" not in b:
        raise SchemaError("budget.csv missing 'kcal_remaining'")

    kcal_remaining = _f(b.get("kcal_remaining"), 0.0)
    if kcal_remaining <= 0:
        raise SchemaError("kcal_remaining must be > 0")

    mins_units, max_units = _detect_budget_targets(list(b.keys()))

    mins: Dict[str, float] = {}
    for base, unit in mins_units.items():
        col = f"{base}_{unit}_remaining"
        mins[f"{base}_{unit}"] = _f(b.get(col), 0.0)
        if mins[f"{base}_{unit}"] < 0:
            raise SchemaError(f"Negative remaining not allowed: {col}")

    maxes: Dict[str, float] = {}
    for base, unit in max_units.items():
        col = f"{base}_{unit}_limit"
        maxes[f"{base}_{unit}"] = _f(b.get(col), 0.0)
        if maxes[f"{base}_{unit}"] < 0:
            raise SchemaError(f"Negative limit not allowed: {col}")

    # Priorities
    priorities: Dict[str, float] = {}
    toxin_penalty = 0.5
    if priorities_path and priorities_path.exists():
        with priorities_path.open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        priorities = {str(k): float(v) for k, v in (y.get("priorities") or {}).items()}
        if "toxin_penalty" in y:
            toxin_penalty = float(y["toxin_penalty"])

    # Normalize priorities to [0,1]
    if priorities:
        mx = max(priorities.values())
        if mx > 0:
            for k in list(priorities.keys()):
                priorities[k] = priorities[k] / mx

    return Problem(
        foods=foods,
        targets=Targets(kcal_remaining=kcal_remaining, mins=mins, maxes=maxes),
        priorities=priorities,
        toxin_penalty=toxin_penalty,
    )

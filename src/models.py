from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import re

VALID_PER100_SUFFIXES = ("_g_per_100g", "_mg_per_100g", "_mcg_per_100g")
VALID_BUDGET_MIN_SUFFIX = ("_remaining",)
VALID_BUDGET_MAX_SUFFIX = ("_limit",)

FOODS_RE = re.compile(r"^(?P<base>[a-z0-9_]+)_(?P<unit>g|mg|mcg)_per_100g$")
BUDGET_MIN_RE = re.compile(r"^(?P<base>[a-z0-9_]+)_(?P<unit>g|mg|mcg)_remaining$")
BUDGET_MAX_RE = re.compile(r"^(?P<base>[a-z0-9_]+)_(?P<unit>g|mg|mcg)_limit$")

# Required base columns; optional extras supported:
# - daily_max_g: per-food daily cap (strict upper bound)
# - price_per_100g: cost (optional)
# - increment_g: step size for this food (e.g., 58 for eggs; greedy uses it, LP enforces integer multiples)
# - unit_size_g: size of a human-friendly unit (e.g., 58g/egg, 100g/pack)
# - unit_label: label to show (e.g., "egg", "pack")
# - binary_pack: 0/1 -> if 1, it's "all or nothing": either 0 or exactly 1 unit_size_g once
REQUIRED_FOOD_BASE = ["name", "kcal_per_100g", "max_serving_g", "category"]

@dataclass
class Food:
    name: str
    kcal_per_100g: float
    max_serving_g: float
    category: str
    per100: Dict[str, float]
    daily_max_g: float | None = None
    price_per_100g: float | None = None
    increment_g: float | None = None
    unit_size_g: float | None = None
    unit_label: str | None = None
    binary_pack: bool = False

@dataclass
class Targets:
    kcal_remaining: float
    mins: Dict[str, float]
    maxes: Dict[str, float]

@dataclass
class Problem:
    foods: List[Food]
    targets: Targets
    priorities: Dict[str, float]
    toxin_penalty: float
    category_caps_g: Dict[str, float] | None = None

class UnitConflictError(ValueError):
    pass

class SchemaError(ValueError):
    pass

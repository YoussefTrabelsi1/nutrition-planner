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

# Required base columns. Optional extras (if present) are enforced/used:
# - daily_max_g
# - price_per_100g
REQUIRED_FOOD_BASE = ["name", "kcal_per_100g", "max_serving_g", "category"]

@dataclass
class Food:
    name: str
    kcal_per_100g: float
    max_serving_g: float
    category: str
    per100: Dict[str, float]                 # base_with_unit -> value per 100g
    daily_max_g: float | None = None         # strict per-food daily cap, optional
    price_per_100g: float | None = None      # optional cost column (same currency across foods)

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

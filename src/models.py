from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import re

# Canonical suffixes and their multipliers (unit awareness lives in headers)
VALID_PER100_SUFFIXES = ("_g_per_100g", "_mg_per_100g", "_mcg_per_100g")
VALID_BUDGET_MIN_SUFFIX = ("_remaining",)
VALID_BUDGET_MAX_SUFFIX = ("_limit",)

# Regex to extract base name and unit from foods headers
FOODS_RE = re.compile(r"^(?P<base>[a-z0-9_]+)_(?P<unit>g|mg|mcg)_per_100g$")
BUDGET_MIN_RE = re.compile(r"^(?P<base>[a-z0-9_]+)_(?P<unit>g|mg|mcg)_remaining$")
BUDGET_MAX_RE = re.compile(r"^(?P<base>[a-z0-9_]+)_(?P<unit>g|mg|mcg)_limit$")

# Required base columns in foods.csv. "daily_max_g" is OPTIONAL and, if present, is enforced.
REQUIRED_FOOD_BASE = ["name", "kcal_per_100g", "max_serving_g", "category"]

@dataclass
class Food:
    name: str
    kcal_per_100g: float
    max_serving_g: float
    category: str
    per100: Dict[str, float]  # base_with_unit -> value per 100g
    daily_max_g: float | None = None  # optional per-food daily cap (strict upper bound)

@dataclass
class Targets:
    kcal_remaining: float
    mins: Dict[str, float]   # base_with_unit -> minimum required
    maxes: Dict[str, float]  # base_with_unit -> maximum allowed

@dataclass
class Problem:
    foods: List[Food]
    targets: Targets
    priorities: Dict[str, float]  # weights on base_with_unit for greedy/soft logic
    toxin_penalty: float
    category_caps_g: Dict[str, float] | None = None  # per-category daily max in grams (lowercased categories)

class UnitConflictError(ValueError):
    pass

class SchemaError(ValueError):
    pass

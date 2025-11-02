from __future__ import annotations

# Reusable helpers for the Streamlit UI layer.
# Keep this file free of business logic (solvers, parsing).

from pathlib import Path
from typing import Dict, List
import tempfile
import os

import streamlit as st
import pandas as pd
import yaml


# -------- Visual / UX --------

def _divider() -> None:
    """Version-safe divider: prefers st.divider() if present, else a markdown rule."""
    try:
        st.divider()  # type: ignore[attr-defined]
    except Exception:
        st.markdown("---")


def _params_container():
    """
    Return a parameters container:
      - Prefer st.popover (newer Streamlit),
      - else fall back to st.expander (closed by default).
    Usage:
        with _params_container():
            ...
    """
    try:
        return st.popover("Parameters")  # type: ignore[attr-defined]
    except Exception:
        return st.expander("Parameters", expanded=False)


# -------- Data loading / YAML --------

def _safe_load_yaml(path: Path | None) -> dict:
    """Best-effort YAML load; returns {} on any error."""
    if not path or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _subject_from_policies(pol_path: Path | None) -> dict:
    """
    Read subject info from policies.yaml:
      subject:
        age: ...
        sex: ...
        body_weight_kg: ...
    Falls back to body_weight_kg = 70.0 if missing.
    """
    y = _safe_load_yaml(pol_path)
    subj = (y.get("subject") or {}) if isinstance(y, dict) else {}
    out = {
        "age": subj.get("age", None),
        "body_weight_kg": subj.get("body_weight_kg", None),
        "sex": subj.get("sex", None),
    }
    if not out["body_weight_kg"]:
        out["body_weight_kg"] = 70.0
    return out


def _load_budget_df(path: Path) -> pd.DataFrame:
    """Read budget.csv (one row). If missing/invalid, return a minimal one-row frame."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            df = pd.DataFrame([{"kcal_remaining": 0.0}])
        return df
    except Exception:
        return pd.DataFrame([{"kcal_remaining": 0.0}])


def _edit_budget(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full budget editor (advanced/parameters panel).
    Version-safe: st.data_editor → st.experimental_data_editor → manual inputs.
    Returns a single-row DataFrame.
    """
    if df.shape[0] == 0:
        df = pd.DataFrame([{"kcal_remaining": 0.0}])

    # Normalize to numeric where possible
    row = df.iloc[0].copy()
    for c in df.columns:
        try:
            row[c] = float(row[c])
        except Exception:
            pass

    editable_df = pd.DataFrame([row])

    # Prefer the modern editor
    try:
        edited = st.data_editor(editable_df, num_rows="fixed")
    except Exception:
        # Older versions
        try:
            edited = st.experimental_data_editor(editable_df, num_rows="fixed")  # type: ignore[attr-defined]
        except Exception:
            # Very old versions: fall back to manual inputs
            cols = list(editable_df.columns)
            out = {}
            left, right = st.columns(2)
            for i, c in enumerate(cols):
                container = left if i % 2 == 0 else right
                with container:
                    val = editable_df.iloc[0][c]
                    try:
                        out[c] = st.number_input(c, value=float(val))
                    except Exception:
                        out[c] = st.text_input(c, value=str(val))
            edited = pd.DataFrame([out])

    return edited.reset_index(drop=True).iloc[:1]


# -------- Lightweight computations for the page (no solving) --------

def _foods_by_name(prob) -> Dict[str, object]:
    """Index foods by name for quick lookup."""
    return {f.name: f for f in prob.foods}


def _totals_from_session_plan(session_plan, foods_by_name: Dict[str, object]) -> Dict[str, float]:
    """
    Compute live totals (kcal + nutrients) from the session plan (manual adds).
    This is for KPIs/progress without running the solver.
    """
    totals: Dict[str, float] = {"kcal": 0.0, "cost": 0.0}
    for name, grams in session_plan.grams_by_food.items():
        f = foods_by_name.get(name)
        if not f or grams <= 0:
            continue
        totals["kcal"] += grams * (f.kcal_per_100g / 100.0)
        # cost if available
        if getattr(f, "price_per_100g", None) is not None:
            totals["cost"] += grams * (float(f.price_per_100g) / 100.0)
        # nutrients
        for k, v in f.per100.items():
            totals[k] = totals.get(k, 0.0) + grams * (v / 100.0)
    return totals


def _apply_auto_macros(edited_budget_df: pd.DataFrame, kcal_budget: float, body_weight_kg: float) -> pd.DataFrame:
    """
    Auto-calc macro mins:
      fat_g        = 1.00 g/kg
      protein_g    = 2.25 g/kg
      carbohydrates_g = remaining kcal / 4 after fat & protein kcal
    Only sets *_remaining columns; leaves other mins/limits as they are.
    """
    fat_g = max(0.0, 1.00 * float(body_weight_kg))
    protein_g = max(0.0, 2.32 * float(body_weight_kg))
    used_kcal = fat_g * 9.0 + protein_g * 4.0
    carbs_kcal = max(0.0, float(kcal_budget) - used_kcal)
    carbs_g = carbs_kcal / 4.0

    df = edited_budget_df.copy()
    if "kcal_remaining" not in df.columns:
        df["kcal_remaining"] = 0.0
    df.loc[0, "kcal_remaining"] = float(kcal_budget)

    for col in ["fat_g_remaining", "protein_g_remaining", "carbohydrates_g_remaining"]:
        if col not in df.columns:
            df[col] = 0.0

    df.loc[0, "fat_g_remaining"] = round(fat_g, 2)
    df.loc[0, "protein_g_remaining"] = round(protein_g, 2)
    df.loc[0, "carbohydrates_g_remaining"] = round(carbs_g, 2)
    return df


# -------- Thin wrapper used by the page to build a Problem for display --------

def build_problem_for_view(load_problem_func, foods_path: Path, budget_df: pd.DataFrame,
                           priorities_path: Path | None, policies_path: Path | None):
    """
    Persist the one-row budget DF to a temp CSV, then call load_problem. This keeps
    the UI layer independent from parser internals and preserves column order.
    """
    tmp_budget = Path(tempfile.gettempdir()) / f"editor_view_{os.getpid()}.csv"
    budget_df.to_csv(tmp_budget, index=False)
    return load_problem_func(foods_path, tmp_budget, priorities_path, policies_path)

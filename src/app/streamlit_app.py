from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd

# import project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # -> project root
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.io_loader import load_problem
from src.optimize import solve_lp_only, solve_greedy, compute_totals
from app.state import get_session_plan, set_flash, show_flash
from app.ui_components import (
    safe_df_display, unit_from_key, kpi_strip, progress_row, card_grid
)

# Use /data and /output at the project root level
DEFAULT_DATA = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


# ---------- small helpers (compat + formatting) ----------

def _divider() -> None:
    """Version-safe divider: prefers st.divider() if present, else a markdown rule."""
    try:
        st.divider()  # type: ignore[attr-defined]
    except Exception:
        st.markdown("---")

def _foods_by_name(prob) -> Dict[str, object]:
    return {f.name: f for f in prob.foods}

def _totals_from_session_plan(session_plan, foods_by_name: Dict[str, object]) -> Dict[str, float]:
    totals: Dict[str, float] = {"kcal": 0.0, "cost": 0.0}
    for name, grams in session_plan.grams_by_food.items():
        f = foods_by_name.get(name)
        if not f or grams <= 0:
            continue
        totals["kcal"] += grams * (f.kcal_per_100g / 100.0)
        if getattr(f, "price_per_100g", None) is not None:
            totals["cost"] += grams * (float(f.price_per_100g) / 100.0)
        for k, v in f.per100.items():
            totals[k] = totals.get(k, 0.0) + grams * (v / 100.0)
    return totals

def _top3_nutrients_for_food(food, mins_keys: List[str]) -> List[str]:
    pairs = []
    for k in mins_keys:
        v = food.per100.get(k, 0.0)
        if v > 0:
            pairs.append((k, v))
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    out = []
    for k, v in pairs[:3]:
        out.append(f"{k}:{v:.1f}{unit_from_key(k)}/100g")
    return out


def _format_units(food, grams: float) -> str:
    size = getattr(food, "unit_size_g", None) or getattr(food, "increment_g", None)
    label = getattr(food, "unit_label", None)
    if not size or size <= 0:
        return ""
    count = grams / float(size)
    if abs(round(count) - count) < 1e-6:
        cnt = f"{int(round(count))}"
    else:
        cnt = f"{count:.2f}"
    if label:
        return f" (~{cnt} {label}{'' if abs(count-1)<1e-6 else 's'})"
    return f" (~{cnt} units)"


def _build_totals_text(problem, totals: Dict[str, float]) -> str:
    lines = []
    lines.append("\nTotals:")
    cost = totals.get("cost", 0.0)
    lines.append(f"  kcal: {totals.get('kcal', 0.0):.1f}    total_cost: {cost:.2f}")
    for k, need in problem.targets.mins.items():
        got = totals.get(k, 0.0)
        unit = unit_from_key(k)
        lines.append(f"  {k}: {got:.3f}{unit} / {need:.3f}{unit}")
    any_toxin = any(totals.get(k, 0.0) > 0 for k in problem.targets.maxes.keys())
    if any_toxin:
        lines.append("  toxins:")
        for k, cap in problem.targets.maxes.items():
            got = totals.get(k, 0.0)
            if got <= 0:
                continue
            unit = unit_from_key(k)
            frac = (got / cap) if cap > 0 else 0.0
            lines.append(f"    {k}: {got:.3f}{unit} (cap {cap:.3f}{unit}, used {frac:.1%})")
    return "\n".join(lines) + "\n"


def _persist_uploaded(uploaded, suffix: str, tempdir: Path) -> Path | None:
    if not uploaded:
        return None
    p = tempdir / f"upload{suffix}"
    with open(p, "wb") as f:
        f.write(uploaded.getbuffer())
    return p


def _load_budget_df(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.empty:
            df = pd.DataFrame([{"kcal_remaining": 0.0}])
        return df
    except Exception:
        return pd.DataFrame([{"kcal_remaining": 0.0}])


def _edit_budget(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Today’s budget")
    if df.shape[0] == 0:
        df = pd.DataFrame([{"kcal_remaining": 0.0}])
    row = df.iloc[0].copy()
    for c in df.columns:
        try:
            row[c] = float(row[c])
        except Exception:
            pass
    editable_df = pd.DataFrame([row])
    # Version-safe editor
    try:
        edited = st.data_editor(editable_df, num_rows="fixed")
    except Exception:
        try:
            edited = st.experimental_data_editor(editable_df, num_rows="fixed")  # type: ignore[attr-defined]
        except Exception:
            # Fallback to manual inputs (two columns)
            cols = list(editable_df.columns)
            out = {}
            left, right = st.columns(2)
            for i, c in enumerate(cols):
                container = left if i % 2 == 0 else right
                with container:
                    val = editable_df.iloc[0][c]
                    if c == "kcal_remaining":
                        # ensure step 100 even in fallback
                        try:
                            default_val = float(val)
                        except Exception:
                            default_val = 2200.0
                        out[c] = st.number_input(c, value=float(default_val), step=100.0)
                    else:
                        try:
                            out[c] = st.number_input(c, value=float(val))
                        except Exception:
                            out[c] = st.text_input(c, value=str(val))
            edited = pd.DataFrame([out])
    edited = edited.reset_index(drop=True).iloc[:1]

    # --- NEW: explicit kcal slider with 100-kcal steps, overrides editor value ---
    try:
        current_kcal = float(edited.iloc[0].get("kcal_remaining", 0.0) or 0.0)
    except Exception:
        current_kcal = 2200.0
    kcal_val = st.slider(
        "kcal_remaining",
        min_value=800,
        max_value=6000,
        value=int(round(current_kcal)) if current_kcal > 0 else 2200,
        step=100,
        help="Adjust in 100 kcal increments",
    )
    edited.loc[0, "kcal_remaining"] = float(kcal_val)

    return edited


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Daily Nutrition Planner", page_icon="🥗", layout="wide")
st.title(f"🥗 Daily Nutrition Planner — {date.today():%a, %b %d}")

with st.sidebar:
    st.header("Data sources")
    use_defaults = st.checkbox("Use default /data files", value=True)
    tmpdir_ctx = tempfile.TemporaryDirectory()
    tempdir = Path(tmpdir_ctx.name)

    foods_upl = budget_upl = prio_upl = pol_upl = None
    if not use_defaults:
        foods_upl = st.file_uploader("foods.csv", type=["csv"])
        budget_upl = st.file_uploader("budget.csv (optional to prefill editor)", type=["csv"])
        prio_upl = st.file_uploader("priorities.yaml", type=["yaml", "yml"])
        pol_upl = st.file_uploader("policies.yaml (optional)", type=["yaml", "yml"])

    st.header("Filters")
    search = st.text_input("Search food (name/category contains)")
    low_sodium = st.checkbox("Low sodium focus", value=False)
    st.caption("Filters affect suggestions only; the solver still enforces caps.")

    run_btn = st.button("Plan")

# Load inputs & editable budget
foods_p = budget_p = prio_p = pol_p = None
if not use_defaults:
    foods_p = _persist_uploaded(foods_upl, "_foods.csv", tempdir)
    budget_p = _persist_uploaded(budget_upl, "_budget.csv", tempdir)
    prio_p = _persist_uploaded(prio_upl, "_priorities.yaml", tempdir)
    pol_p = _persist_uploaded(pol_upl, "_policies.yaml", tempdir)

budget_df = _load_budget_df(Path(budget_p) if budget_p else (DEFAULT_DATA / "budget.csv"))
edited_budget_df = _edit_budget(budget_df)

# ---- replaced st.divider() with version-safe helper ----
_divider()

# Tabs
tab_overview, tab_plan, tab_micros, tab_history = st.tabs(["Overview", "Meal plan", "Micros", "History"])

session_plan = get_session_plan()  # manual builder (user “Add” clicks)

def _load_prob_only():
    # write edited budget to tmp so names/units align; we won't solve here
    tmp_budget = Path(tempfile.gettempdir()) / f"editor_view_{os.getpid()}.csv"
    edited_budget_df.to_csv(tmp_budget, index=False)
    return load_problem(
        Path(foods_p) if foods_p else (DEFAULT_DATA / "foods.csv"),
        tmp_budget,
        Path(prio_p) if prio_p else (DEFAULT_DATA / "priorities.yaml"),
        Path(pol_p) if pol_p else None,
    )

def _solve_now():
    edited_budget_path = (Path(tempfile.gettempdir()) / f"edited_budget_{os.getpid()}.csv")
    edited_budget_df.to_csv(edited_budget_path, index=False)
    prob = load_problem(
        Path(foods_p) if foods_p else (DEFAULT_DATA / "foods.csv"),
        edited_budget_path,
        Path(prio_p) if prio_p else (DEFAULT_DATA / "priorities.yaml"),
        Path(pol_p) if pol_p else None,
    )
    # auto LP→Greedy
    res = solve_lp_only(prob, allow_soft=False, seed=None, special_categories=["meat", "fish", "fish_canned"])
    used = "LP"
    if res is None:
        plan, totals = solve_greedy(prob, seed=None, special_categories=["meat", "fish", "fish_canned"])
        used = "greedy"
    else:
        plan, totals = res
    return prob, used, plan, totals

# ---------------- Overview Tab ----------------
with tab_overview:
    st.subheader("KPI")
    try:
        prob_for_view = _load_prob_only()
        foods_map = _foods_by_name(prob_for_view)
        live = _totals_from_session_plan(session_plan, foods_map)  # ← live from added foods

        consumed = {
            "kcal": live.get("kcal", 0.0),
            "protein": live.get("protein_g", 0.0),
            "carbs": live.get("carbohydrates_g", 0.0),
            "fat": live.get("fat_g", 0.0),
            "fiber": live.get("fiber_g", 0.0),
        }
        targets = {
            "kcal": float(edited_budget_df.iloc[0].get("kcal_remaining", 0.0)),
            "protein": prob_for_view.targets.mins.get("protein_g", 0.0),
            "carbs": prob_for_view.targets.mins.get("carbohydrates_g", 0.0),
            "fat": prob_for_view.targets.mins.get("fat_g", 0.0),
            "fiber": prob_for_view.targets.mins.get("fiber_g", 0.0),
        }
        kpi_strip(consumed, targets, [
            ("Calories", "kcal"), ("Protein (g)", "protein"),
            ("Carbs (g)", "carbs"), ("Fat (g)", "fat"), ("Fiber (g)", "fiber")
        ])
        progress_row(consumed, targets, ["protein", "carbs", "fat", "fiber"])
    except Exception as e:
        st.warning(f"KPI unavailable: {e}")

    # -------- Suggested foods (compact + pagination) --------
    st.markdown("### Suggested foods")
    show_why = st.checkbox("Show why (top nutrients)", value=False)
    page_size = st.selectbox("Items per page", [6, 9, 12], index=0, key="sugg_page_size")
    # Build suggestion list (no solve)
    try:
        prob = prob_for_view
        mins_keys = list(prob.targets.mins.keys())
        foods = prob.foods

        def include_food(f) -> bool:
            name = (f.name or "").lower()
            cat = (f.category or "").lower()
            if search and (search.lower() not in name) and (search.lower() not in cat):
                return False
            if low_sodium and f.per100.get("sodium_mg", 0.0) > 400:
                return False
            return True

        # Simple relevance: protein per kcal, then fiber per kcal, then low kcal
        def score(f):
            kcal = max(f.kcal_per_100g, 1e-9)
            ppk = f.per100.get("protein_g", 0.0) / kcal
            fibk = f.per100.get("fiber_g", 0.0) / kcal
            return (ppk * 2.0 + fibk * 0.8, -kcal)

        items = [f for f in foods if include_food(f)]
        items.sort(key=score, reverse=True)

        # Pagination
        total = len(items)
        pages = max(1, (total + page_size - 1) // page_size)
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="sugg_page")
        start = (page - 1) * page_size
        shown = items[start:start + page_size]

        # Build grid rows
        def _top3(food):
            pairs = [(k, food.per100.get(k, 0.0)) for k in mins_keys if food.per100.get(k, 0.0) > 0]
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            return ", ".join(f"{k}:{v:.1f}{unit_from_key(k)}/100g" for k, v in pairs[:3])

        grid_rows: List[Dict[str, str]] = []
        for idx, f in enumerate(shown):
            step = float(getattr(f, "unit_size_g", None) or getattr(f, "increment_g", None) or 50.0)
            why = _top3(f) if show_why else None
            def _make_add(name=f.name, grams=step):
                def _do():
                    session_plan.add(name, grams)
                    set_flash(f"Added {grams:.0f} g {name}", "success")
                return _do
            grid_rows.append({
                "key": f"sugg_{start+idx}",
                "name": f.name,
                "serving": f"{int(step)} g",
                "kcal": int(round(f.kcal_per_100g * step / 100.0)),
                "P": round(f.per100.get("protein_g", 0.0) * step / 100.0, 1),
                "C": round(f.per100.get("carbohydrates_g", 0.0) * step / 100.0, 1),
                "F": round(f.per100.get("fat_g", 0.0) * step / 100.0, 1),
                "why": why,
                "on_click": _make_add(),
            })
        card_grid(grid_rows, cols=3, show_why=show_why)
    except Exception as e:
        st.warning(f"Suggestions unavailable: {e}")

# ---------------- Meal plan Tab ----------------
with tab_plan:
    st.subheader("Your meal plan (manual)")
    show_flash()
    if not session_plan.grams_by_food:
        st.info("No items yet. Add foods from **Overview → Suggested foods**.")
    else:
        names = list(session_plan.grams_by_food.keys())
        grams = [session_plan.grams_by_food[n] for n in names]
        df = pd.DataFrame({"Food": names, "Grams": grams})
        safe_df_display(df)
        cols = st.columns(4)
        with cols[0]:
            if st.button("Clear all"):
                session_plan.clear()
                set_flash("Cleared manual meal plan.", "warning")
        try:
            prob, _, _, _ = _solve_now()
            foods_by_name = {f.name: f for f in prob.foods}
            manual_plan = {n: session_plan.grams_by_food[n] for n in session_plan.grams_by_food}
            manual_totals = {"kcal": 0.0, "cost": 0.0}
            for name, g in manual_plan.items():
                f = foods_by_name.get(name)
                if not f or g <= 0:
                    continue
                manual_totals["kcal"] += g * (f.kcal_per_100g / 100.0)
                if getattr(f, "price_per_100g", None) is not None:
                    manual_totals["cost"] += g * (float(f.price_per_100g) / 100.0)
                for k, v in f.per100.items():
                    manual_totals[k] = manual_totals.get(k, 0.0) + g * (v / 100.0)
            kcal_need = float(edited_budget_df.iloc[0].get("kcal_remaining", 0.0))
            st.caption(f"Manual plan calories: {manual_totals.get('kcal', 0.0):.0f} / {kcal_need:.0f} kcal")
        except Exception:
            pass

# ---------------- Micros Tab ----------------
with tab_micros:
    st.subheader("Micronutrient gaps (after solve)")
    st.caption("Run the solver to see deficits and suggestions.")

# ---------------- History Tab ----------------
with tab_history:
    st.subheader("History")
    st.info("Persisting history/templates is not implemented yet.")

# ---- replaced st.divider() with version-safe helper ----
# (bottom section separator)
_divider()

# ---------- Run solver & show results ----------
if run_btn:
    with st.spinner("Building problem and solving…"):
        try:
            prob, used, plan, totals = _solve_now()
        except Exception as e:
            st.error(f"Failed to build/solve: {e}")
            plan = totals = None

    if plan is None:
        st.error("No feasible plan found by LP or Greedy.")
    else:
        st.subheader(f"Solver result ({used})")
        consumed = {
            "kcal": totals.get("kcal", 0.0),
            "protein": totals.get("protein_g", 0.0),
            "carbs": totals.get("carbohydrates_g", 0.0),
            "fat": totals.get("fat_g", 0.0),
            "fiber": totals.get("fiber_g", 0.0),
        }
        targets = {
            "kcal": float(edited_budget_df.iloc[0].get("kcal_remaining", 0.0)),
            "protein": prob.targets.mins.get("protein_g", 0.0),
            "carbs": prob.targets.mins.get("carbohydrates_g", 0.0),
            "fat": prob.targets.mins.get("fat_g", 0.0),
            "fiber": prob.targets.mins.get("fiber_g", 0.0),
        }
        kpi_strip(consumed, targets, [
            ("Calories", "kcal"),
            ("Protein (g)", "protein"),
            ("Carbs (g)", "carbs"),
            ("Fat (g)", "fat"),
            ("Fiber (g)", "fiber"),
        ])
        progress_row(consumed, targets, ["protein", "carbs", "fat", "fiber"])

        foods_by_name = {f.name: f for f in prob.foods}
        rows = []
        for name, grams in sorted(plan.items(), key=lambda kv: (-kv[1], kv[0])):
            f = foods_by_name[name]
            price100 = getattr(f, "price_per_100g", None)
            cost = (grams * (float(price100) / 100.0)) if (price100 is not None) else None
            top3 = ", ".join(_top3_nutrients_for_food(f, list(prob.targets.mins.keys())))
            units = _format_units(f, grams)
            rows.append({
                "Food": name,
                "Grams": round(grams, 1),
                "Cost": None if cost is None else round(cost, 2),
                "Top nutrients (per 100g)": top3,
                "Units": units.strip(),
                "Category": (f.category or "").lower(),
            })
        df = pd.DataFrame(rows)
        safe_df_display(df)
        st.code(_build_totals_text(prob, totals), language="text")

        # Save report to /output + download
        try:
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = DEFAULT_OUTPUT_DIR / f"plan_{ts}.txt"
            with out_path.open("w", encoding="utf-8") as f:
                f.write("Chosen plan\n")
                f.write(df.to_csv(index=False))
                f.write("\n")
                f.write(_build_totals_text(prob, totals))
            st.success(f"Saved report to: {out_path}")
        except Exception as e:
            st.warning(f"Failed to save report: {e}")

# Cleanup temp uploads
try:
    tmpdir_ctx.cleanup()
except Exception:
    pass

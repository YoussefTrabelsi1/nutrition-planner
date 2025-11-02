from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List

import streamlit as st
import pandas as pd
import yaml  # used to read policies.yaml

# ==== PATHS (project root two levels up) ====
# We are in: <project>/src/app/streamlit_app.py
# Project root is two levels up from here.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Core solver imports (unchanged)
from src.io_loader import load_problem
from src.optimize import solve_lp_only, solve_greedy
from app.state import get_session_plan, set_flash, show_flash

# Local UI helpers
from app.ui_components import (
    safe_df_display, unit_from_key, kpi_strip, progress_row, card_grid
)
from app.ui_helpers import (
    _divider, _params_container, _safe_load_yaml, _subject_from_policies,
    _load_budget_df, _edit_budget, _foods_by_name, _totals_from_session_plan,
    _apply_auto_macros, build_problem_for_view,
)

# Defaults
DEFAULT_DATA = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

# ---------- UI ----------
st.set_page_config(page_title="Daily Nutrition Planner", page_icon="🥗", layout="wide")
st.title(f"🥗 Daily Nutrition Planner — {date.today():%a, %b %d}")

# Session keys for latest plan results (so we can show plan BEFORE suggestions)
if "latest_plan" not in st.session_state:
    st.session_state["latest_plan"] = None
if "latest_totals" not in st.session_state:
    st.session_state["latest_totals"] = None
if "latest_used" not in st.session_state:
    st.session_state["latest_used"] = None

# Sidebar: data sources + kcal budget only (keep main page clean)
with st.sidebar:
    st.header("Data sources")
    use_defaults = st.checkbox("Use default /data files", value=True)

    # Temporary dir for uploads
    tmpdir_ctx = tempfile.TemporaryDirectory()
    tempdir = Path(tmpdir_ctx.name)

    foods_upl = budget_upl = prio_upl = pol_upl = None
    if not use_defaults:
        foods_upl = st.file_uploader("foods.csv", type=["csv"])
        budget_upl = st.file_uploader("budget.csv (prefill for parameters)", type=["csv"])
        prio_upl = st.file_uploader("priorities.yaml", type=["yaml", "yml"])
        pol_upl = st.file_uploader("policies.yaml", type=["yaml", "yml"])

    st.header("Calorie budget")
    # Clean main control: user only sets kcal here (macros are auto from policies)
    kcal_budget = st.slider("kcal_remaining", min_value=800, max_value=6000, value=2700, step=100)

    # Solver trigger
    run_btn = st.button("Plan")

# Resolve uploaded file paths (if any)
foods_p = budget_p = prio_p = pol_p = None
if not use_defaults:
    if foods_upl:
        foods_p = tempdir / "upload_foods.csv"
        foods_p.write_bytes(foods_upl.getbuffer())
    if budget_upl:
        budget_p = tempdir / "upload_budget.csv"
        budget_p.write_bytes(budget_upl.getbuffer())
    if prio_upl:
        prio_p = tempdir / "upload_priorities.yaml"
        prio_p.write_bytes(prio_upl.getbuffer())
    if pol_upl:
        pol_p = tempdir / "upload_policies.yaml"
        pol_p.write_bytes(pol_upl.getbuffer())

# Load base budget (for advanced editor) and policies (for auto macros)
base_budget_df = _load_budget_df(budget_p if budget_p else (DEFAULT_DATA / "budget.csv"))
policies_path = pol_p if pol_p else (DEFAULT_DATA / "policies.yaml")
subject = _subject_from_policies(policies_path)
body_weight_kg = float(subject.get("body_weight_kg", 70.0))

# Parameters panel (advanced). We keep a default for auto_macros in session_state.
if "auto_macros" not in st.session_state:
    st.session_state["auto_macros"] = True

with _params_container():
    st.caption("Tune full budget (mins/limits). If auto-macros is on, Protein/Fat/Carbs are recalculated from calories & body weight.")
    st.text(
        f"Detected body weight: {body_weight_kg:.1f} kg"
        + (f" • age: {subject.get('age')}" if subject.get("age") else "")
    )
    st.session_state["auto_macros"] = st.checkbox(
        "Auto-calculate macros from policies (recommended)",
        value=st.session_state["auto_macros"]
    )
    advanced_budget_df = _edit_budget(base_budget_df)

# Build the working budget from parameters + auto macros (if enabled)
edited_budget_df = advanced_budget_df.copy()
if st.session_state["auto_macros"]:
    edited_budget_df = _apply_auto_macros(edited_budget_df, kcal_budget, body_weight_kg)
else:
    # Even when auto is off, we keep kcal synchronized to the main slider.
    if "kcal_remaining" not in edited_budget_df.columns:
        edited_budget_df["kcal_remaining"] = 0.0
    edited_budget_df.loc[0, "kcal_remaining"] = float(kcal_budget)

# Separator
__ = _divider()

# Tabs
tab_overview, tab_plan, tab_micros, tab_history = st.tabs(["Overview", "Meal plan", "Micros", "History"])
session_plan = get_session_plan()  # holds manual additions

# Lightweight problem for food list & KPIs (no solve)
def _load_prob_only():
    return build_problem_for_view(
        load_problem_func=load_problem,
        foods_path=foods_p if foods_p else (DEFAULT_DATA / "foods.csv"),
        budget_df=edited_budget_df,
        priorities_path=prio_p if prio_p else (DEFAULT_DATA / "priorities.yaml"),
        policies_path=pol_p if pol_p else None,
    )

# ---------------- PLAN ACTION (compute & store in session so we can show first) ----------
if run_btn:
    with st.spinner("Building problem and solving…"):
        try:
            # Persist the working budget to a temp CSV for the solver
            tmp_budget = Path(tempfile.gettempdir()) / f"solve_budget_{os.getpid()}.csv"
            edited_budget_df.to_csv(tmp_budget, index=False)

            # Build problem
            prob = load_problem(
                foods_p if foods_p else (DEFAULT_DATA / "foods.csv"),
                tmp_budget,
                prio_p if prio_p else (DEFAULT_DATA / "priorities.yaml"),
                pol_p if pol_p else None,
            )

            # Auto: LP first → Greedy fallback
            res = solve_lp_only(prob, allow_soft=False, seed=None, special_categories=["meat", "fish", "fish_canned"])
            used = "LP"
            if res is None:
                plan, totals = solve_greedy(prob, seed=None, special_categories=["meat", "fish", "fish_canned"])
                used = "greedy"
            else:
                plan, totals = res

            # Save to session for display at top of Overview
            st.session_state["latest_plan"] = plan
            st.session_state["latest_totals"] = totals
            st.session_state["latest_used"] = used

            # Also write a text report like CLI
            try:
                os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = DEFAULT_OUTPUT_DIR / f"plan_{ts}.txt"
                foods_by_name = _foods_by_name(prob)
                # Build a CSV-like text
                rows = []
                for name, grams in sorted(plan.items(), key=lambda kv: (-kv[1], kv[0])):
                    f = foods_by_name[name]
                    price100 = getattr(f, "price_per_100g", None)
                    cost = (grams * (float(price100) / 100.0)) if (price100 is not None) else None
                    rows.append((name, grams, cost))
                with out_path.open("w", encoding="utf-8") as f:
                    f.write("Chosen plan\nFood,Grams,Cost\n")
                    for n, g, c in rows:
                        f.write(f"{n},{g:.1f},{'' if c is None else round(c,2)}\n")
                set_flash(f"Saved plan to: {out_path}", "success")
            except Exception:
                pass

        except Exception as e:
            st.error(f"Failed to build/solve: {e}")
            st.session_state["latest_plan"] = None
            st.session_state["latest_totals"] = None
            st.session_state["latest_used"] = None

# ---------------- Overview Tab ----------------
with tab_overview:
    st.subheader("KPI")
    try:
        prob_for_view = _load_prob_only()
        foods_map = _foods_by_name(prob_for_view)
        live = _totals_from_session_plan(session_plan, foods_map)  # KPIs move when you add foods

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
            ("Calories", "kcal"),
            ("Protein (g)", "protein"),
            ("Carbs (g)", "carbs"),
            ("Fat (g)", "fat"),
            ("Fiber (g)", "fiber")
        ])
        progress_row(consumed, targets, ["protein", "carbs", "fat", "fiber"])
    except Exception as e:
        st.warning(f"KPI unavailable: {e}")

    # ===== Show latest PLAN results FIRST (with 'Add planned items') =====
    if st.session_state.get("latest_plan"):
        _divider()
        st.markdown(f"### Plan (algorithm recommendation) — {st.session_state.get('latest_used')}")
        # Render table
        try:
            prob = _load_prob_only()  # for foods metadata
            foods_by_name = _foods_by_name(prob)
            plan = st.session_state["latest_plan"]
            rows = []
            for name, grams in sorted(plan.items(), key=lambda kv: (-kv[1], kv[0])):
                f = foods_by_name.get(name)
                if not f:
                    continue
                price100 = getattr(f, "price_per_100g", None)
                cost = (grams * (float(price100) / 100.0)) if (price100 is not None) else None
                pairs = [(k, f.per100.get(k, 0.0)) for k in prob.targets.mins.keys() if f.per100.get(k, 0.0) > 0]
                pairs.sort(key=lambda kv: kv[1], reverse=True)
                top3 = ", ".join(f"{k}:{v:.1f}{unit_from_key(k)}/100g" for k, v in pairs[:3])
                size = getattr(f, "unit_size_g", None) or getattr(f, "increment_g", None)
                label = getattr(f, "unit_label", None)
                units = ""
                if size and size > 0:
                    cnt = grams / float(size)
                    units = f" (~{int(round(cnt)) if abs(cnt-round(cnt))<1e-6 else f'{cnt:.2f}'} {label or 'units'}{'s' if label and abs(cnt-1.0)>1e-6 else ''})"
                rows.append({
                    "Food": name,
                    "Grams": round(grams, 1),
                    "Cost": None if cost is None else round(cost, 2),
                    "Top nutrients (per 100g)": top3,
                    "Units": units.strip(),
                    "Category": (f.category or "").lower(),
                })
            df_plan = pd.DataFrame(rows)
            safe_df_display(df_plan)

            # Add ALL planned items to manual plan (this adds the algorithm's plan, not the suggestions)
            if st.button("➕ Add planned items"):
                for name, grams in st.session_state["latest_plan"].items():
                    session_plan.add(name, float(grams))
                set_flash(f"Added {len(st.session_state['latest_plan'])} planned items.", "success")

            # Totals summary
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

            totals = st.session_state.get("latest_totals") or {}
            st.code(_build_totals_text(prob, totals), language="text")
        except Exception as e:
            st.warning(f"Could not render plan: {e}")

    # ===== Suggested foods (ALL items + live search; no pagination) =====
    _divider()
    st.markdown("### Suggested foods")
    search_q = st.text_input("Search foods (live)", value=st.session_state.get("food_search", ""), key="food_search")
    show_why = st.checkbox("Show why (top nutrients)", value=False)

    try:
        # Ensure we have a problem instance (from earlier KPI, else load now)
        if "prob_for_view" not in locals():
            prob_for_view = _load_prob_only()
        prob = prob_for_view
        mins_keys = list(prob.targets.mins.keys())
        foods = prob.foods

        # Filter by live search (contains match on name or category)
        q = (search_q or "").strip().lower()
        def include_food(f) -> bool:
            if not q:
                return True
            name = (f.name or "").lower()
            cat = (f.category or "").lower()
            return (q in name) or (q in cat)

        # Relevance: protein per kcal, then fiber per kcal, then lower kcal
        def _score(food):
            kcal = max(food.kcal_per_100g, 1e-9)
            ppk = food.per100.get("protein_g", 0.0) / kcal
            fibk = food.per100.get("fiber_g", 0.0) / kcal
            return (ppk * 2.0 + fibk * 0.8, -kcal)

        items = [f for f in foods if include_food(f)]
        items.sort(key=_score, reverse=True)

        # Build a single list (no pagination)
        def _top3(food):
            pairs = [(k, food.per100.get(k, 0.0)) for k in mins_keys if food.per100.get(k, 0.0) > 0]
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            return ", ".join(f"{k}:{v:.1f}{unit_from_key(k)}/100g" for k, v in pairs[:3])

        grid_rows: List[Dict[str, str]] = []
        for idx, f in enumerate(items):
            step = float(getattr(f, "unit_size_g", None) or getattr(f, "increment_g", None) or 50.0)
            why = _top3(f) if show_why else None

            def _make_add(name=f.name, grams=step):
                def _do():
                    session_plan.add(name, grams)
                    set_flash(f"Added {grams:.0f} g {name}", "success")
                return _do

            grid_rows.append({
                "key": f"sugg_{idx}",
                "name": f.name,
                "serving": f"{int(step)} g",
                "kcal": int(round(f.kcal_per_100g * step / 100.0)),
                "P": round(f.per100.get("protein_g", 0.0) * step / 100.0, 1),
                "C": round(f.per100.get("carbohydrates_g", 0.0) * step / 100.0, 1),
                "F": round(f.per100.get("fat_g", 0.0) * step / 100.0, 1),
                "why": why,
                "on_click": _make_add(),
            })

        # 3-column compact grid; "Why" text optional via checkbox
        card_grid(grid_rows, cols=3, show_why=show_why)

    except Exception as e:
        st.warning(f"Suggestions unavailable: {e}")

# ---------------- Meal plan Tab ----------------
with tab_plan:
    st.subheader("Your meal plan (manual)")
    show_flash()
    if not session_plan.grams_by_food:
        st.info("No items yet. Add foods from **Overview → Suggested foods** or click **Add planned items**.")
    else:
        names = list(session_plan.grams_by_food.keys())
        grams = [session_plan.grams_by_food[n] for n in names]
        df = pd.DataFrame({"Food": names, "Grams": grams})
        safe_df_display(df)

        # Quick clear
        cols = st.columns(4)
        with cols[0]:
            if st.button("Clear all"):
                session_plan.clear()
                set_flash("Cleared manual meal plan.", "warning")

# ---------------- Micros Tab ----------------
with tab_micros:
    st.subheader("Micronutrient gaps (after solve)")
    st.caption("Run the solver to see deficits and suggestions.")

# ---------------- History Tab ----------------
with tab_history:
    st.subheader("History")
    st.info("Persisting history/templates is not implemented yet.")

# Bottom separator
__ = _divider()

# Cleanup temp uploads
try:
    tmpdir_ctx.cleanup()
except Exception:
    pass

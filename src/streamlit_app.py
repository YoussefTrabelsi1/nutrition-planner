from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd  # NEW: for robust table rendering


import streamlit as st

# Ensure we can import from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.io_loader import load_problem
from src.optimize import solve_lp_only, solve_greedy, compute_totals

DEFAULT_DATA = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "output"


# ---------- Formatting helpers (copied from CLI style to avoid changing existing files) ----------

def _unit_from_key(key: str) -> str:
    if key.endswith("_g"): return "g"
    if key.endswith("_mg"): return "mg"
    if key.endswith("_mcg"): return "mcg"
    return ""


def _top3_nutrients_for_food(food, mins_keys: List[str]) -> List[str]:
    pairs = []
    for k in mins_keys:
        v = food.per100.get(k, 0.0)
        if v > 0:
            pairs.append((k, v))
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    top = []
    for k, v in pairs[:3]:
        unit = _unit_from_key(k)
        top.append(f"{k}:{v:.1f}{unit}/100g")
    return top


def _format_units(food, grams: float) -> str:
    size = getattr(food, "unit_size_g", None) or getattr(food, "increment_g", None)
    label = getattr(food, "unit_label", None)
    if not size or size <= 0:
        return ""
    count = grams / float(size)
    if abs(round(count) - count) < 1e-6:
        count_str = f"{int(round(count))}"
    else:
        count_str = f"{count:.2f}"
    if label:
        suffix = "" if abs(count - 1.0) < 1e-6 else "s"
        return f" (~{count_str} {label}{suffix})"
    return f" (~{count_str} units)"


def _build_plan_table_text(plan: Dict[str, float], foods_obj: Dict[str, object], mins_keys: List[str]) -> str:
    if not plan:
        return "No foods selected.\n"
    items = sorted(plan.items(), key=lambda kv: (-kv[1], kv[0]))
    name_w = max(len("Food"), *(len(k) for k, _ in items))
    lines = []
    lines.append(f"{'Food'.ljust(name_w)}  Grams    Cost     Top nutrients (per 100g)         Units")
    lines.append(f"{'-'*name_w}  -----    ----     -----------------------------         -----")
    total_cost = 0.0
    for name, grams in items:
        f = foods_obj[name]
        price100 = getattr(f, "price_per_100g", None)
        cost = (grams * (float(price100) / 100.0)) if (price100 is not None) else 0.0
        total_cost += cost
        cost_str = f"{cost:7.2f}" if price100 is not None else "   --  "
        tops = ", ".join(_top3_nutrients_for_food(f, mins_keys)) or "-"
        units = _format_units(f, grams)
        lines.append(f"{name.ljust(name_w)}  {grams:7.1f}  {cost_str}  {tops:<35}{units}")
    lines.append(f"{'Total'.ljust(name_w)}  {'':7}  {total_cost:7.2f}")
    return "\n".join(lines) + "\n"


def _build_totals_text(problem, totals: Dict[str, float]) -> str:
    lines = []
    lines.append("\nTotals:")
    cost = totals.get("cost", 0.0)
    lines.append(f"  kcal: {totals.get('kcal', 0.0):.1f}    total_cost: {cost:.2f}")
    for k, need in problem.targets.mins.items():
        got = totals.get(k, 0.0)
        unit = _unit_from_key(k)
        lines.append(f"  {k}: {got:.3f}{unit} / {need:.3f}{unit}")

    any_toxin = any(totals.get(k, 0.0) > 0 for k in problem.targets.maxes.keys())
    if any_toxin:
        lines.append("  toxins:")
        for k, cap in problem.targets.maxes.items():
            got = totals.get(k, 0.0)
            if got <= 0:
                continue
            unit = _unit_from_key(k)
            frac = (got / cap) if cap > 0 else 0.0
            lines.append(f"    {k}: {got:.3f}{unit} (cap {cap:.3f}{unit}, used {frac:.1%})")
    return "\n".join(lines) + "\n"


def _all_mins_satisfied(problem, totals: Dict[str, float]) -> bool:
    return all(totals.get(k, 0.0) + 1e-9 >= need for k, need in problem.targets.mins.items())


def _build_greedy_miss_text(problem, totals: Dict[str, float]) -> str:
    misses = []
    for k, need in problem.targets.mins.items():
        got = totals.get(k, 0.0)
        if got + 1e-9 < need:
            unit = _unit_from_key(k)
            misses.append(f"Unmet: {k} short by {(need-got):.3f}{unit}")
    if not misses:
        return ""
    return "\n".join(misses[:5]) + "\n"


def _problem_from_inputs(
    foods_file: Path | None,
    budget_file: Path | None,
    priorities_file: Path | None,
    policies_file: Path | None,
):
    return load_problem(
        foods_file or (DEFAULT_DATA / "foods.csv"),
        budget_file or (DEFAULT_DATA / "budget.csv"),
        priorities_file or (DEFAULT_DATA / "priorities.yaml"),
        policies_file if (policies_file and policies_file.exists()) else None,
    )


def _persist_uploaded(uploaded, suffix: str, tempdir: Path) -> Path | None:
    if not uploaded:
        return None
    p = tempdir / f"upload{suffix}"
    with open(p, "wb") as f:
        f.write(uploaded.getbuffer())
    return p


# ... keep existing imports ...

# (no changes above)

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Nutrition Planner", layout="wide")

st.title("🥗 Nutrition Planner — LP + Greedy")
st.caption("Low-calorie ingredient planner that meets mins, respects caps, and honors category/pack rules.")

with st.sidebar:
    st.header("Inputs")

    use_defaults = st.checkbox("Use default /data files", value=True)

    foods_upl = budget_upl = prio_upl = pol_upl = None
    tmpdir_ctx = tempfile.TemporaryDirectory()
    tempdir = Path(tmpdir_ctx.name)

    if not use_defaults:
        foods_upl = st.file_uploader("foods.csv", type=["csv"])
        budget_upl = st.file_uploader("budget.csv", type=["csv"])
        prio_upl = st.file_uploader("priorities.yaml", type=["yaml", "yml"])
        pol_upl = st.file_uploader("policies.yaml (optional)", type=["yaml", "yml"])

    st.header("Options")
    solver_choice = st.selectbox("Solver", ["Auto (LP→Greedy)", "LP only", "Greedy only"], index=0)
    allow_soft = st.checkbox("Allow soft constraints (LP)", value=False)
    seed = st.number_input("Random seed (optional)", value=0, step=1)
    use_seed = st.checkbox("Enable seed", value=False)

    run_btn = st.button("Plan")

# Placeholders for results
plan_container = st.container()
totals_container = st.container()
download_container = st.container()

if run_btn:
    # Prepare files
    with st.spinner("Building problem..."):
        foods_p = budget_p = prio_p = pol_p = None
        if not use_defaults:
            foods_p = _persist_uploaded(foods_upl, "_foods.csv", tempdir)
            budget_p = _persist_uploaded(budget_upl, "_budget.csv", tempdir)
            prio_p = _persist_uploaded(prio_upl, "_priorities.yaml", tempdir)
            pol_p = _persist_uploaded(pol_upl, "_policies.yaml", tempdir)

        prob = _problem_from_inputs(
            Path(foods_p) if foods_p else None,
            Path(budget_p) if budget_p else None,
            Path(prio_p) if prio_p else None,
            Path(pol_p) if pol_p else None,
        )

    # ---- CHANGED: replace st.toast with a version-safe info message ----
    st.info("Solving…")  # works across Streamlit versions

    s_val = int(seed) if use_seed else None

    plan = totals = None
    used = None

    if solver_choice in ("Auto (LP→Greedy)", "LP only"):
        res = solve_lp_only(prob, allow_soft=allow_soft, seed=s_val, special_categories=["meat", "fish", "fish_canned"])
        if res is not None:
            plan, totals = res
            used = "LP"

    if plan is None:
        if solver_choice in ("Auto (LP→Greedy)", "Greedy only"):
            plan, totals = solve_greedy(prob, seed=s_val, special_categories=["meat", "fish", "fish_canned"])
            used = "greedy"

    if plan is None:
        st.error("No feasible plan found by LP or Greedy.")
    else:
        # ... rest of file unchanged ...
        foods_by_name = {f.name: f for f in prob.foods}
        header = f"\nChosen plan ({used})\n"
        body_plan = _build_plan_table_text(plan, foods_by_name, list(prob.targets.mins.keys()))
        body_totals = _build_totals_text(prob, totals)
        footer = ""
        if not _all_mins_satisfied(prob, totals) and used in {"greedy", "LP-soft"}:
            footer = _build_greedy_miss_text(prob, totals)
        report = header + body_plan + body_totals + footer

        with plan_container:
            st.subheader(f"Chosen plan ({used})")
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
            try:
                # Newer Streamlit
                st.dataframe(df, use_container_width=True)
            except TypeError:
                try:
                    # Older Streamlit
                    st.dataframe(df)
                except TypeError:
                    # Very old versions: fallback to static table
                    st.table(df)

        with totals_container:
            st.subheader("Totals")
            st.code(body_totals, language="text")

        # Save & download (unchanged)
        try:
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = DEFAULT_OUTPUT_DIR / f"plan_{ts}.txt"
            with out_path.open("w", encoding="utf-8") as f:
                f.write(report)
            st.success(f"Saved report to: {out_path}")
        except Exception as e:
            st.warning(f"Failed to save report: {e}")

        with download_container:
            st.download_button(
                "Download report (.txt)",
                data=report.encode("utf-8"),
                file_name=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

    # Cleanup temp uploads
    try:
        tmpdir_ctx.cleanup()
    except Exception:
        pass

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from datetime import datetime
import os

from io_loader import load_problem
from optimize import solve_lp_only, solve_greedy, compute_totals

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = DEFAULT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "output"


def _unit_from_key(key: str) -> str:
    if key.endswith("_g"): return "g"
    if key.endswith("_mg"): return "mg"
    if key.endswith("_mcg"): return "mcg"
    return ""


def _top3_nutrients_for_food(food, mins_keys: list[str]) -> list[str]:
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
    # Prefer unit_size_g if provided; else fall back to increment_g for display
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
        # pluralize naively when not equal to 1
        suffix = "" if abs(count - 1.0) < 1e-6 else "s"
        return f" (~{count_str} {label}{suffix})"
    return f" (~{count_str} units)"


def _build_plan_table_text(plan: dict[str, float], foods_obj: dict[str, object], mins_keys: list[str]) -> str:
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


def _build_totals_text(problem, totals: dict[str, float]) -> str:
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


def _all_mins_satisfied(problem, totals: dict[str, float]) -> bool:
    return all(totals.get(k, 0.0) + 1e-9 >= need for k, need in problem.targets.mins.items())


def _build_greedy_miss_text(problem, totals: dict[str, float]) -> str:
    misses = []
    for k, need in problem.targets.mins.items():
        got = totals.get(k, 0.0)
        if got + 1e-9 < need:
            unit = _unit_from_key(k)
            misses.append(f"Unmet: {k} short by {(need-got):.3f}{unit}")
    if not misses:
        return ""
    return "\n".join(misses[:5]) + "\n"


def run(
    foods: Path | None = None,
    budget: Path | None = None,
    priorities: Path | None = None,
    policies: Path | None = None,
    allow_soft: bool = False,
    seed: int | None = None,
) -> int:
    foods_p = foods or (DEFAULT_DATA / "foods.csv")
    budget_p = budget or (DEFAULT_DATA / "budget.csv")
    prio_p = priorities or (DEFAULT_DATA / "priorities.yaml")
    pol_p = policies or (DEFAULT_DATA / "policies.yaml")
    prob = load_problem(foods_p, budget_p, prio_p, pol_p if pol_p.exists() else None)

    res = solve_lp_only(
        prob,
        allow_soft=allow_soft,
        seed=seed,
        special_categories=["meat", "fish", "fish_canned"],
    )
    used = "LP"
    if res is None:
        plan, totals = solve_greedy(
            prob, seed=seed, special_categories=["meat", "fish", "fish_canned"]
        )
        used = "greedy"
    else:
        plan, totals = res

    foods_by_name = {f.name: f for f in prob.foods}
    header = f"\nChosen plan ({used})\n"
    body_plan = _build_plan_table_text(plan, foods_by_name, list(prob.targets.mins.keys()))
    body_totals = _build_totals_text(prob, totals)
    footer = ""
    if not _all_mins_satisfied(prob, totals) and used in {"greedy", "LP-soft"}:
        footer = _build_greedy_miss_text(prob, totals)

    report = header + body_plan + body_totals + footer
    print(report, end="")

    # Save to /output with timestamp seconds
    try:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = DEFAULT_OUTPUT_DIR / f"plan_{ts}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved report to: {out_path}")
    except Exception as e:
        print(f"Warning: failed to save report: {e}")

    return 0 if _all_mins_satisfied(prob, totals) else 2


def main() -> None:
    ap = argparse.ArgumentParser(description="Low-calorie nutrition planner (LP + greedy fallback)")
    ap.add_argument("--foods", type=Path, default=None, help="Path to foods.csv")
    ap.add_argument("--budget", type=Path, default=None, help="Path to budget.csv")
    ap.add_argument("--priorities", type=Path, default=None, help="Path to priorities.yaml")
    ap.add_argument("--policies", type=Path, default=None, help="Path to policies.yaml (optional)")
    ap.add_argument("--allow-soft", action="store_true", help="Allow soft constraints (best-effort when infeasible)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible variety")
    args = ap.parse_args()
    code = run(args.foods, args.budget, args.priorities, args.policies, args.allow_soft, args.seed)
    sys.exit(code)


if __name__ == "__main__":
    main()

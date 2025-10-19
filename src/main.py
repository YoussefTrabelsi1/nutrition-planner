from __future__ import annotations
import argparse
import sys
from pathlib import Path
from io_loader import load_problem
from optimize import solve_lp_only, solve_greedy, compute_totals

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = DEFAULT_ROOT / "data"

def _unit_from_key(key: str) -> str:
    if key.endswith("_g"): return "g"
    if key.endswith("_mg"): return "mg"
    if key.endswith("_mcg"): return "mcg"
    return ""

def _print_plan_table(plan: dict[str, float], foods_by_name: dict[str, dict]) -> None:
    if not plan:
        print("No foods selected.")
        return
    name_w = max(len("Food"), *(len(k) for k in plan.keys()))
    print(f"{'Food'.ljust(name_w)}  Grams    Cost")
    print(f"{'-'*name_w}  -----    ----")
    total_cost = 0.0
    for k in sorted(plan.keys()):
        grams = plan[k]
        price100 = foods_by_name[k].get("price_per_100g")
        cost = (grams * (price100 / 100.0)) if (price100 is not None) else 0.0
        total_cost += cost
        cost_str = f"{cost:7.2f}" if price100 is not None else "   --  "
        print(f"{k.ljust(name_w)}  {grams:7.1f}  {cost_str}")
    print(f"{'Total'.ljust(name_w)}  {'':7}  {total_cost:7.2f}")

def _print_totals(problem, totals: dict[str, float]) -> None:
    print("\nTotals:")
    cost = totals.get("cost", 0.0)
    print(f"  kcal: {totals.get('kcal', 0.0):.1f}    total_cost: {cost:.2f}")
    for k, need in problem.targets.mins.items():
        got = totals.get(k, 0.0)
        unit = _unit_from_key(k)
        print(f"  {k}: {got:.3f}{unit} / {need:.3f}{unit}")

    # Show toxins only if any non-zero usage
    any_toxin = False
    for k, cap in problem.targets.maxes.items():
        got = totals.get(k, 0.0)
        if got > 0:
            any_toxin = True
            break
    if any_toxin:
        print("  toxins:")
        for k, cap in problem.targets.maxes.items():
            got = totals.get(k, 0.0)
            if got <= 0:
                continue  # hide 0.0% rows
            unit = _unit_from_key(k)
            frac = (got / cap) if cap > 0 else 0.0
            print(f"    {k}: {got:.3f}{unit} (cap {cap:.3f}{unit}, used {frac:.1%})")

def _all_mins_satisfied(problem, totals: dict[str, float]) -> bool:
    return all(totals.get(k, 0.0) + 1e-9 >= need for k, need in problem.targets.mins.items())

def _print_greedy_miss(problem, totals: dict[str, float]) -> None:
    misses = []
    for k, need in problem.targets.mins.items():
        got = totals.get(k, 0.0)
        if got + 1e-9 < need:
            unit = _unit_from_key(k)
            misses.append(f"Unmet: {k} short by {(need-got):.3f}{unit}")
    for line in misses[:5]:
        print(line)

def run(
    foods: Path | None = None,
    budget: Path | None = None,
    priorities: Path | None = None,
    policies: Path | None = None,
    allow_soft: bool = False,
) -> int:
    foods_p = foods or (DEFAULT_DATA / "foods.csv")
    budget_p = budget or (DEFAULT_DATA / "budget.csv")
    prio_p = priorities or (DEFAULT_DATA / "priorities.yaml")
    pol_p = policies or (DEFAULT_DATA / "policies.yaml")
    prob = load_problem(foods_p, budget_p, prio_p, pol_p if pol_p.exists() else None)

    # Attempt LP (with optional soft); if prices exist, LP will second-stage minimize cost.
    res = solve_lp_only(prob, allow_soft=allow_soft)
    used = "LP"
    if res is None:
        plan, totals = solve_greedy(prob)
        used = "greedy"
    else:
        plan, totals = res

    print(f"\nChosen plan ({used})")
    # Build lookup for price column in table
    foods_by_name = {
        f.name: {"price_per_100g": f.price_per_100g} for f in prob.foods
    }
    _print_plan_table(plan, foods_by_name)
    _print_totals(prob, totals)

    ok = _all_mins_satisfied(prob, totals)
    if not ok and used in {"greedy", "LP-soft"}:
        _print_greedy_miss(prob, totals)

    return 0 if ok else 2

def main() -> None:
    ap = argparse.ArgumentParser(description="Low-calorie nutrition planner (LP + greedy fallback)")
    ap.add_argument("--foods", type=Path, default=None, help="Path to foods.csv")
    ap.add_argument("--budget", type=Path, default=None, help="Path to budget.csv")
    ap.add_argument("--priorities", type=Path, default=None, help="Path to priorities.yaml")
    ap.add_argument("--policies", type=Path, default=None, help="Path to policies.yaml (optional)")
    ap.add_argument("--allow-soft", action="store_true", help="Allow soft constraints (best-effort when infeasible)")
    args = ap.parse_args()
    code = run(args.foods, args.budget, args.priorities, args.policies, args.allow_soft)
    sys.exit(code)

if __name__ == "__main__":
    main()

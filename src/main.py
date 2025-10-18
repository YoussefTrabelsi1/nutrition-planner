from __future__ import annotations
import argparse
from pathlib import Path
from io_loader import load_problem
from optimize import solve_lp_only, solve_greedy

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = DEFAULT_ROOT / "data"


def run(foods: Path | None = None, budget: Path | None = None, priorities: Path | None = None) -> None:
    foods_p = foods or (DEFAULT_DATA / "foods.csv")
    budget_p = budget or (DEFAULT_DATA / "budget.csv")
    prio_p = priorities or (DEFAULT_DATA / "priorities.yaml")
    prob = load_problem(foods_p, budget_p, prio_p)

    res = solve_lp_only(prob)
    if res is None:
        # Fallback
        plan, totals = solve_greedy(prob)
        used = "greedy"
    else:
        plan, totals = res
        used = "LP"

    print(f"Chosen plan ({used}) (food -> grams):")
    if not plan:
        print("  <empty>")
    for k, v in sorted(plan.items()):
        print(f"  {k}: {v:.1f} g")

    print("\nTotals:")
    print(f"  kcal: {totals.get('kcal', 0.0):.1f}")
    # show satisfied mins and used toxin fractions
    for k, need in prob.targets.mins.items():
        got = totals.get(k, 0.0)
        print(f"  {k}: {got:.3f} / {need:.3f}")
    for k, cap in prob.targets.maxes.items():
        got = totals.get(k, 0.0)
        frac = (got / cap) if cap > 0 else 0.0
        print(f"  {k}: {got:.3f} (cap {cap:.3f}, used {frac:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Low-calorie nutrition planner (LP + greedy fallback)")
    ap.add_argument("--foods", type=Path, default=None, help="Path to foods.csv")
    ap.add_argument("--budget", type=Path, default=None, help="Path to budget.csv")
    ap.add_argument("--priorities", type=Path, default=None, help="Path to priorities.yaml")
    args = ap.parse_args()
    run(args.foods, args.budget, args.priorities)


if __name__ == "__main__":
    main()

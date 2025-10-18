from __future__ import annotations
import argparse
import sys
from pathlib import Path
from io_loader import load_problem
from optimize import solve_lp_only, solve_greedy, compute_totals

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = DEFAULT_ROOT / "data"


def _unit_from_key(key: str) -> str:
    if key.endswith("_g"):
        return "g"
    if key.endswith("_mg"):
        return "mg"
    if key.endswith("_mcg"):
        return "mcg"
    return ""


def _print_plan_table(plan: dict[str, float]) -> None:
    if not plan:
        print("No foods selected.")
        return
    name_w = max(len("Food"), *(len(k) for k in plan.keys()))
    print(f"{'Food'.ljust(name_w)}  Grams")
    print(f"{'-'*name_w}  -----")
    for k in sorted(plan.keys()):
        print(f"{k.ljust(name_w)}  {plan[k]:7.1f}")


def _print_totals(problem, totals: dict[str, float]) -> None:
    print("\nTotals:")
    print(f"  kcal: {totals.get('kcal', 0.0):.1f}")
    for k, need in problem.targets.mins.items():
        got = totals.get(k, 0.0)
        unit = _unit_from_key(k)
        print(f"  {k}: {got:.3f}{unit} / {need:.3f}{unit}")
    for k, cap in problem.targets.maxes.items():
        got = totals.get(k, 0.0)
        unit = _unit_from_key(k)
        frac = (got / cap) if cap > 0 else 0.0
        print(f"  {k}: {got:.3f}{unit} (cap {cap:.3f}{unit}, used {frac:.1%})")


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

    # Try LP (with optional soft objective)
    res = solve_lp_only(prob, allow_soft=allow_soft)
    used = "LP-soft" if (allow_soft and res is not None and not _all_mins_satisfied(prob, res[1])) else "LP"

    if res is None:
        print("LP status: infeasible or unavailable.")
        plan, totals = solve_greedy(prob)
        used = "greedy"
    else:
        plan, totals = res

    print(f"\nChosen plan ({used})")
    _print_plan_table(plan)
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

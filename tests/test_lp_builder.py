import tempfile, textwrap
from pathlib import Path
from src.io_loader import load_problem
from src.optimize import solve_lp_only


def write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content.strip() + "\n", encoding="utf-8")
    return p


def test_generic_lp_constraints_parametrized():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # Two foods: HighProteinSafe (low toxin), HighProteinToxic (high toxin)
        foods = write(root, "foods.csv", textwrap.dedent(
            """
            name,kcal_per_100g,max_serving_g,category,protein_g_per_100g,lead_mcg_per_100g
            HighProteinSafe,200,300,Test,50,1
            HighProteinToxic,200,300,Test,60,10
            """
        ))
        budget = write(root, "budget.csv", textwrap.dedent(
            """
            kcal_remaining,protein_g_remaining,lead_mcg_limit
            2000,60,5
            """
        ))
        prio = write(root, "priorities.yaml", "priorities: {protein_g: 1.0}")

        prob = load_problem(foods, budget, prio)
        res = solve_lp_only(prob)
        assert res is not None, "LP should be feasible"
        plan, totals = res
        # Check mins/caps respected
        assert totals.get("protein_g", 0.0) >= 60 - 1e-6
        assert totals.get("lead_mcg", 0.0) <= 5 + 1e-6
        # Ensure the safe food is used and toxic food is not needed
        assert plan.get("HighProteinSafe", 0.0) > 0

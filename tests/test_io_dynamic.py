from pathlib import Path
from src.io_loader import load_problem
import tempfile, textwrap


def write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content.strip() + "\n", encoding="utf-8")
    return p


def test_dynamic_add_vitd_parses():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        foods = write(root, "foods.csv", textwrap.dedent(
            """
            name,kcal_per_100g,max_serving_g,category,vitd_mcg_per_100g
            Test,100,100,Cat,5
            """
        ))
        budget = write(root, "budget.csv", textwrap.dedent(
            """
            kcal_remaining,vitd_mcg_remaining
            1000,10
            """
        ))
        prio = write(root, "priorities.yaml", "priorities: {vitd_mcg: 1.0}")
        prob = load_problem(foods, budget, prio)
        assert "vitd_mcg" in prob.targets.mins
        assert prob.targets.mins["vitd_mcg"] == 10
        assert prob.foods[0].per100["vitd_mcg"] == 5


def test_mixed_units_raises():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        foods = write(root, "foods.csv", textwrap.dedent(
            """
            name,kcal_per_100g,max_serving_g,category,sodium_g_per_100g,sodium_mg_per_100g
            Test,100,100,Cat,0.1,10
            """
        ))
        budget = write(root, "budget.csv", textwrap.dedent(
            """
            kcal_remaining
            1000
            """
        ))
        try:
            load_problem(foods, budget, None)
            assert False, "Expected exception"
        except Exception as e:
            assert "Mixed units for 'sodium'" in str(e)

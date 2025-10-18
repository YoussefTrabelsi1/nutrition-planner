from pathlib import Path
from src.main import run


def test_smoke_runs(tmp_path: Path, capsys):
    # Copy example data from repo data/
    from shutil import copyfile
    import pkgutil
    import os
    repo_root = Path(__file__).resolve().parents[1]
    foods = repo_root / "data" / "foods.csv"
    budget = repo_root / "data" / "budget.csv"
    prio = repo_root / "data" / "priorities.yaml"

    run(foods, budget, prio)
    out = capsys.readouterr().out
    assert "Chosen plan" in out

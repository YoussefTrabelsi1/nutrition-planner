````markdown
# nutrition-planner

Low-calorie ingredient planner that **meets daily macro/micro targets** and **respects toxicity caps** with a **protein-first bias** by default. CSV-driven and unit-safe.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.main
````

**Overrides:**

```bash
python -m src.main --foods data/foods.csv --budget data/budget.csv --priorities data/priorities.yaml
```

**Exit codes:**

* `0` = feasible (all minimums met)
* `2` = infeasible / miss

---

## Schema conventions & units

### `foods.csv` (per 100 g)

**Required columns:**
`name,kcal_per_100g,max_serving_g,category`

**Dynamic nutrients/toxins columns:**
`*_g_per_100g`, `*_mg_per_100g`, `*_mcg_per_100g`
Examples: `protein_g_per_100g`, `iron_mg_per_100g`, `methylmercury_mcg_per_100g`

**Rules:**

* No negatives.
* `max_serving_g > 0`, `kcal_per_100g >= 0`.
* No mixed units for the same base (e.g., don’t use both `sodium_g_per_100g` and `sodium_mg_per_100g`).

### `budget.csv` (single row, “remaining today”)

**Required column:**
`kcal_remaining`

**Dynamic minimums (≥):** `<base>_remaining`
**Dynamic caps (≤):** `<base>_limit`

Include units in base names, e.g.:
`protein_g_remaining`, `sodium_mg_remaining`, `methylmercury_mcg_limit`.

---

## Optimization

* **LP (PuLP/CBC):** minimize calories subject to all mins/caps + kcal budget.
* **Greedy fallback:** deterministic, priority-aware; used when LP is unavailable or infeasible.

---

## Diagnostics (Ticket 4)

* **On LP infeasible:** prints up to 3 blocking mins with “need vs max achievable” and levers such as:

  * Lever: increase `kcal_remaining` by ~X
  * Lever: add high-<nutrient> foods
* **On greedy miss:** prints unmet minimums with shortfall.

Messages are concise (1–5 lines), no stack traces.

---

## Output format

* Pretty table of `(Food, Grams)`.
* Totals show kcal and every min/max with units (`g`, `mg`, `mcg`).

---

## Tests

```bash
pytest -q
```

Covers smoke LP/greedy, dynamic I/O parsing, generic LP builder, and greedy behavior in simple scenarios.

---

If you want a tiny test for the vitamin-C infeasible case (as in the ticket), say the word and I’ll add `tests/test_diagnostics.py`.

```
```

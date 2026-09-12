# Intelligent Resource Allocation & Scenario Optimization Platform

A constrained optimization system that decides which projects to fund and which
employee gets staffed where — under budget, headcount, skill, and regional
constraints — with a predictive "expected value" layer and a live what-if
simulator. Built as portfolio project #3 for a Data Science Associate
application, to learn the **predict-then-optimize** pattern used throughout
real-world allocation and resourcing systems.

## Problem framing

Given N candidate projects and M employees (a fixed budget/headcount pool),
determine the allocation that maximizes expected portfolio value — subject to
a money budget, employee capacity, skill-matching, minimum regional coverage,
and a set of "must-fund" strategic/compliance projects — with a live simulator
so a decision-maker can move sliders and see the optimizer re-solve in
real time.

## Why this is a legitimate synthetic-data project

All data here is generated, not scraped or leaked from any company — which is
standard for optimization portfolio projects, since real internal allocation
data is never public. What makes it more than arbitrary `np.random` noise:

- **Calibrated against a real reference dataset.** The expected-value spread
  for each priority tier is drawn from the **CAPM dataset** (`Ecdat::Capm` on
  [Rdatasets](https://github.com/vincentarelbundock/Rdatasets)) — 516 months
  of real historical industry-portfolio stock returns (durables, consumables,
  food staples). Annualized mean/volatility per industry maps onto our three
  priority tiers:

  | CAPM industry | Annualized return / volatility | Maps to tier | Why |
  |---|---|---|---|
  | Durables (`rdur`) | 6.3% / **20.1%** (widest) | `revenue` | Growth bets — highest upside, highest risk |
  | Consumables (`rcon`) | 5.1% / 20.1% | `social_impact` | Moderate, harder to predict |
  | Food staples (`rfood`) | 8.0% / **15.7%** (tightest) | `risk_reduction` | Defensive, stable, predictable value |

  This gives a real, citable reason why revenue-tier projects have the widest
  expected-value spread — not a hand-tuned choice.

- **Deliberate correlations, not independent noise.** Project cost scales
  with priority tier and region cost-of-living; employee cost scales with
  skill scarcity and region; required skills have tier-affinity (e.g.
  `risk_reduction` projects skew toward Finance/Legal, `revenue` projects
  skew toward Marketing/Engineering).

## Repository structure

```
resource-allocation-optimizer/
├── data/
│   ├── raw/
│   │   └── capm_reference.csv          # real reference data (cached)
│   └── processed/                       # generated + derived datasets
│       ├── projects.csv
│       ├── employees.csv
│       ├── projects_with_predictions.csv
│       ├── allocation_result.csv
│       └── sensitivity_*.csv
├── src/
│   ├── generate_data.py                 # Step 1: synthetic data generator
│   ├── predict_value.py                 # Step 2: predictive layer
│   ├── optimize.py                      # Step 3: CP-SAT optimizer + baseline
│   └── sensitivity.py                   # Step 4: sensitivity sweeps
├── app/
│   └── streamlit_app.py                 # Step 5: live what-if simulator
├── models/
│   └── best_model.joblib                # trained predictive model
├── reports/                              # plots (PNG + interactive HTML)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Running the pipeline (in order)

Each step writes its output into `data/processed/`, which the next step reads.

```bash
python src/generate_data.py       # -> projects.csv, employees.csv
python src/predict_value.py       # -> projects_with_predictions.csv, models/best_model.joblib
python src/optimize.py            # -> allocation_result.csv (prints CP-SAT vs. baseline comparison)
python src/sensitivity.py         # -> sensitivity_*.csv + reports/*.html (+ PNG if Chrome is installed)
streamlit run app/streamlit_app.py  # live simulator in the browser
```

`optimize.py` accepts flags for experimentation:
```bash
python src/optimize.py --budget_fraction 0.3 --min_projects_per_region 5 --time_limit 60
```

## Architecture

```
Synthetic project/employee data (calibrated against real CAPM returns)
        │
        ▼
Predictive layer: Linear Regression vs. Random Forest
  → predicts expected_value_k from project attributes ONLY
  → (cost, tier, region, skills, duration — never the ROI/value-std columns,
     which would leak the answer)
        │
        ▼
CP-SAT optimization model (OR-Tools)
  decision variables : fund[project] (bool), assign[employee, project] (bool)
  objective          : maximize total predicted value of funded projects
  constraints        : money budget · employee capacity (1 project per employee)
                        · strict skill coverage · regional minimums
                        · must-fund mandates
        │
        ▼
Optimal allocation  ──vs.──  naive "highest-value-first" greedy baseline
        │
        ▼
Streamlit what-if simulator (sliders re-trigger the solver live)
```

## Modeling decisions worth knowing (for the write-up / interview)

- **CP-SAT, not a pure LP (PuLP)** — chosen because the problem is
  fundamentally a combined **assignment problem** (which employee → which
  project) and **knapsack problem** (which projects to fund), both of which
  need integer/boolean variables that CP-SAT models naturally.
- **Budget and headcount are two separate constrained resources**, not one
  merged pool — mirroring how real portfolio management works: you're capped
  by cash *and* by people, and skills are what tie the two together.
- **1:1 staffing**: each employee is assigned to at most one project.
- **Strict skill-matching**: a project can only be funded if every skill it
  requires is covered by at least one assigned employee — this makes
  headcount (not budget) the usual binding constraint, since 80 employees
  can staff far fewer than 80 multi-skill projects.
- **The optimizer maximizes *predicted* value, not true value** — deliberately,
  because a real system never knows a project's true future value in advance.

## Results (baseline scenario: 25% budget, 80 employees, min. 3 projects/region)

- **Predictive layer**: Linear Regression (R² = 0.837) *beat* Random Forest
  (R² = 0.790) on held-out test data. Not a bug — the true value-generating
  process is close to linear (`value ≈ cost × (1 + tier's average ROI)`), so
  the extra flexibility of a Random Forest added variance without benefit.
  Knowing when *not* to reach for the more complex model is itself the point.
- **Feature importance**: `estimated_cost_k` dominates (>90% of Random Forest
  importance) — cost's absolute range ($20K–$700K+) swamps the few-percentage-point
  ROI difference between tiers. Predicting ROI% instead of absolute value
  would surface tier effects more clearly.
- **Optimizer vs. naive baseline**: the CP-SAT optimizer improved total
  portfolio value by **roughly 19–21%** over a naive "fund the biggest numbers
  first" baseline (exact figure varies slightly run-to-run — see caveats below).
  The naive baseline also **violated the regional minimum** for at least one
  region, since it has no concept of business constraints beyond raw value.
- **Headcount is the usual binding constraint**, not budget: at baseline
  settings, all 80 employees get used while a meaningful chunk of budget
  can remain unspent.
- **Cutting headcount ~20% below baseline can make the problem INFEASIBLE**,
  not just worse — a must-fund compliance project can lose its only available
  skill match entirely. This is a sharper, more useful finding than a smooth
  value-vs-headcount curve: it identifies a real hiring-gap risk.
- **Priority-tier weighting reallocates the mix, not just the total**: doubling
  the weight on `revenue` roughly doubled the number of revenue projects
  funded, while total *raw* value barely moved — a clean trade-off dial for
  the what-if simulator.

## Honest caveats

- **CP-SAT often returns `FEASIBLE`, not `OPTIMAL`**, within the default time
  limits used for interactive/sweep speed. This means results can vary
  slightly between runs (parallel search finds a different, still-strong,
  solution). Longer `--time_limit` values give more reliable optimality
  guarantees for a final report.
- **Employee counts above 80 in the sensitivity sweep and simulator are
  simulated** by bootstrap-resampling the existing 80 employees' skill/cost
  profiles — a reasonable approximation for "what if we hired more people
  like the ones we have," not a claim about specific new hires.
- All underlying data is synthetic; the CAPM calibration justifies the
  *shape* of the value distributions, not the literal dollar figures.


## Tech stack

Python · pandas · scikit-learn · OR-Tools (CP-SAT) · Streamlit · Plotly
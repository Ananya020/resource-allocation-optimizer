"""
sensitivity.py
---------------
Systematically varies three things and re-solves the optimizer each time:

  1. Budget        : +/-20% around the baseline (25% of total project cost)
  2. Headcount     : +/-20% around the baseline (80 employees)
  3. Priority tier weighting : what if leadership cared 2x more about one tier?

For (2), employee counts ABOVE 80 don't exist in our generated data, so
"hiring more" is simulated by bootstrap-resampling additional employees from
the empirical skill/cost distribution we already have (sampling existing
rows with replacement, given new IDs). This is a reasonable, explainable
approximation for "what if we hired more people like the ones we already
have" — not a claim about literally which individuals we'd hire.

Each run uses a short time limit (10s) since we only need directional
trends here, not certified-optimal solutions for every single point.

Outputs:
  - data/processed/sensitivity_budget.csv
  - data/processed/sensitivity_headcount.csv
  - data/processed/sensitivity_weights.csv
  - reports/sensitivity_budget.png
  - reports/sensitivity_headcount.png
  - reports/sensitivity_weights.png

Run:
    python src/sensitivity.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

from optimize import load_data, solve

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_BUDGET_FRACTION = 0.25
BASELINE_MIN_REGION = 3
TIME_LIMIT = 10  # short, for sweep speed — trends matter more than proven optimality here


def bootstrap_employees(employees: pd.DataFrame, target_n: int) -> pd.DataFrame:
    """Return an employee pool of exactly target_n rows. Subsamples without
    replacement if target_n <= current size; bootstraps extra rows (sampled
    with replacement from the existing pool, given fresh IDs) if target_n is
    larger, to simulate hiring more people with a similar skill/cost mix."""
    n = len(employees)
    if target_n <= n:
        return employees.sample(n=target_n, random_state=SEED).reset_index(drop=True)
    extra_needed = target_n - n
    extra = employees.sample(n=extra_needed, replace=True, random_state=SEED).reset_index(drop=True)
    extra["employee_id"] = [f"EMP-BOOT-{i:03d}" for i in range(extra_needed)]
    return pd.concat([employees, extra], ignore_index=True)


def run_budget_sweep(projects, employees):
    fractions = [round(BASELINE_BUDGET_FRACTION * m, 3) for m in [0.8, 0.9, 1.0, 1.1, 1.2]]
    rows = []
    for frac in fractions:
        budget_k = projects["estimated_cost_k"].sum() * frac
        result, _ = solve(projects, employees, budget_k, BASELINE_MIN_REGION, TIME_LIMIT)
        funded = {p for p, v in result.get("fund", {}).items() if v}
        rows.append({
            "budget_fraction": frac,
            "budget_k": budget_k,
            "status": result["status"],
            "projects_funded": len(funded),
            "total_value_k": result.get("objective_value_k", 0),
            "employees_used": len(result.get("assign", {})),
        })
        print(f"  budget_fraction={frac:.3f} -> value=${rows[-1]['total_value_k']:,.0f}K, "
              f"funded={rows[-1]['projects_funded']}, status={result['status']}")
    return pd.DataFrame(rows)


def run_headcount_sweep(projects, employees):
    base_n = len(employees)
    counts = [int(round(base_n * m)) for m in [0.8, 0.9, 1.0, 1.1, 1.2]]
    budget_k = projects["estimated_cost_k"].sum() * BASELINE_BUDGET_FRACTION
    rows = []
    for n in counts:
        pool = bootstrap_employees(employees, n)
        result, _ = solve(projects, pool, budget_k, BASELINE_MIN_REGION, TIME_LIMIT)
        funded = {p for p, v in result.get("fund", {}).items() if v}
        rows.append({
            "employee_count": n,
            "status": result["status"],
            "projects_funded": len(funded),
            "total_value_k": result.get("objective_value_k", 0),
            "employees_used": len(result.get("assign", {})),
        })
        print(f"  employee_count={n} -> value=${rows[-1]['total_value_k']:,.0f}K, "
              f"funded={rows[-1]['projects_funded']}, status={result['status']}")
    return pd.DataFrame(rows)


def run_weight_sweep(projects, employees):
    budget_k = projects["estimated_cost_k"].sum() * BASELINE_BUDGET_FRACTION
    scenarios = {
        "baseline (1,1,1)": {},
        "revenue x2": {"revenue": 2.0},
        "social_impact x2": {"social_impact": 2.0},
        "risk_reduction x2": {"risk_reduction": 2.0},
    }
    rows = []
    for label, weights in scenarios.items():
        result, _ = solve(projects, employees, budget_k, BASELINE_MIN_REGION, TIME_LIMIT, tier_weights=weights)
        funded = {p for p, v in result.get("fund", {}).items() if v}
        # report RAW (unweighted) value so scenarios are comparable to each other
        raw_value = projects.loc[projects["project_id"].isin(funded), "predicted_value_k"].sum()
        tier_mix = projects.loc[projects["project_id"].isin(funded)].groupby("priority_tier").size().to_dict()
        rows.append({
            "scenario": label,
            "status": result["status"],
            "projects_funded": len(funded),
            "raw_total_value_k": raw_value,
            "revenue_funded": tier_mix.get("revenue", 0),
            "social_impact_funded": tier_mix.get("social_impact", 0),
            "risk_reduction_funded": tier_mix.get("risk_reduction", 0),
        })
        print(f"  {label:20s} -> raw_value=${raw_value:,.0f}K, mix={tier_mix}")
    return pd.DataFrame(rows)


def plot_line(df, x_col, y_col, title, xaxis_title, out_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines+markers"))
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=y_col, template="plotly_white")
    fig.write_html(str(REPORTS_DIR / out_name.replace(".png", ".html")))
    try:
        fig.write_image(str(REPORTS_DIR / out_name), width=800, height=500, scale=2)
    except Exception as e:
        print(f"  (skipped PNG export for {out_name} — Chrome/kaleido not available: {type(e).__name__}. "
              f"HTML version saved instead — open it in a browser.)")


def plot_weight_mix(df, out_name):
    fig = go.Figure()
    for tier, color in [("revenue_funded", "#EF553B"), ("social_impact_funded", "#00CC96"),
                         ("risk_reduction_funded", "#636EFA")]:
        fig.add_trace(go.Bar(name=tier.replace("_funded", ""), x=df["scenario"], y=df[tier]))
    fig.update_layout(title="How priority-tier weighting shifts the funded mix", barmode="stack",
                       xaxis_title="Weighting scenario", yaxis_title="Projects funded", template="plotly_white")
    fig.write_html(str(REPORTS_DIR / out_name.replace(".png", ".html")))
    try:
        fig.write_image(str(REPORTS_DIR / out_name), width=900, height=500, scale=2)
    except Exception as e:
        print(f"  (skipped PNG export for {out_name} — Chrome/kaleido not available: {type(e).__name__}. "
              f"HTML version saved instead — open it in a browser.)")


def main():
    projects, employees = load_data()

    print("--- Budget sensitivity (+/-20%) ---")
    budget_df = run_budget_sweep(projects, employees)
    budget_df.to_csv(OUT_DIR / "sensitivity_budget.csv", index=False)
    plot_line(budget_df, "budget_k", "total_value_k", "Portfolio value vs. budget",
              "Budget ($K)", "sensitivity_budget.png")

    print("\n--- Headcount sensitivity (+/-20%) ---")
    headcount_df = run_headcount_sweep(projects, employees)
    headcount_df.to_csv(OUT_DIR / "sensitivity_headcount.csv", index=False)
    plot_line(headcount_df, "employee_count", "total_value_k", "Portfolio value vs. headcount",
              "Number of employees", "sensitivity_headcount.png")

    print("\n--- Priority weighting sensitivity ---")
    weight_df = run_weight_sweep(projects, employees)
    weight_df.to_csv(OUT_DIR / "sensitivity_weights.csv", index=False)
    plot_weight_mix(weight_df, "sensitivity_weights.png")

    print(f"\nSaved sensitivity CSVs and plots -> {OUT_DIR} / {REPORTS_DIR}")


if __name__ == "__main__":
    main()
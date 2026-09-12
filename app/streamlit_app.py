"""
streamlit_app.py
-----------------
Live what-if simulator for the resource allocation optimizer.

Move the sliders -> the CP-SAT solver re-runs -> you see the reallocated
portfolio and how it compares to a naive greedy baseline.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# app/ is a sibling of src/, so add src/ to the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from optimize import load_data, solve, greedy_baseline
from sensitivity import bootstrap_employees

st.set_page_config(page_title="Resource Allocation Optimizer", layout="wide")

WEIGHT_PRESETS = {
    "Balanced (no bias)": {},
    "Revenue-focused (2x)": {"revenue": 2.0},
    "Social-impact-focused (2x)": {"social_impact": 2.0},
    "Risk-reduction-focused (2x)": {"risk_reduction": 2.0},
}


@st.cache_data
def get_base_data():
    return load_data()


@st.cache_data(show_spinner=False)
def run_scenario(budget_fraction, employee_count, min_region, weight_label, time_limit):
    projects, employees = get_base_data()
    budget_k = projects["estimated_cost_k"].sum() * budget_fraction
    pool = bootstrap_employees(employees, employee_count)
    weights = WEIGHT_PRESETS[weight_label]

    result, _ = solve(projects, pool, budget_k, min_region, time_limit_sec=time_limit, tier_weights=weights)
    baseline = greedy_baseline(projects, pool, budget_k, min_region)
    return projects, pool, budget_k, result, baseline


def region_tier_chart(funded_df, all_df, group_col, title):
    fig = go.Figure()
    funded_counts = funded_df.groupby(group_col).size()
    all_regions = sorted(all_df[group_col].unique())
    fig.add_trace(go.Bar(
        x=all_regions,
        y=[funded_counts.get(r, 0) for r in all_regions],
        marker_color="#636EFA",
    ))
    fig.update_layout(title=title, template="plotly_white", height=350, showlegend=False)
    return fig


def main():
    st.title("Intelligent Resource Allocation & Scenario Optimizer")
    st.caption(
        "Move the sliders in the sidebar. Each change re-solves a CP-SAT constrained "
        "optimization problem (project selection + employee-to-project assignment) "
        "and compares it against a naive 'fund the highest-value projects first' baseline."
    )

    projects_all, employees_all = get_base_data()
    total_cost_all = projects_all["estimated_cost_k"].sum()

    with st.sidebar:
        st.header("Scenario controls")
        budget_fraction = st.slider(
            "Budget (% of total project cost)", min_value=10, max_value=50, value=25, step=1,
        ) / 100
        st.caption(f"= ${total_cost_all * budget_fraction:,.0f}K of ${total_cost_all:,.0f}K total")

        employee_count = st.slider(
            "Employee headcount", min_value=40, max_value=120, value=80, step=4,
            help="Above 80 (the actual generated headcount), extra staff are simulated by "
                 "resampling the existing skill/cost profiles — an approximation for 'what if we hired more'.",
        )

        min_region = st.slider("Minimum funded projects per region", 0, 10, 3)

        weight_label = st.selectbox("Priority weighting", list(WEIGHT_PRESETS.keys()))

        with st.expander("Advanced"):
            time_limit = st.slider("Solver time limit (seconds)", 3, 30, 8,
                                    help="Longer = more likely to prove optimality, but slower to interact with.")

    with st.spinner("Re-solving the optimization model..."):
        projects, pool, budget_k, result, baseline = run_scenario(
            budget_fraction, employee_count, min_region, weight_label, time_limit
        )

    if result["status"] not in ("OPTIMAL", "FEASIBLE"):
        st.error(
            "No feasible allocation exists for this combination of sliders. "
            "This usually means a must-fund (compliance/strategic) project needs a skill "
            "that isn't available at this headcount level, or the regional minimum can't be "
            "met. Try raising headcount or lowering the regional minimum."
        )
        return

    opt_funded_ids = {p for p, v in result["fund"].items() if v}
    opt_funded_df = projects[projects["project_id"].isin(opt_funded_ids)]
    opt_value = result["objective_value_k"] if not WEIGHT_PRESETS[weight_label] else \
        opt_funded_df["predicted_value_k"].sum()  # show raw value when a weighting bias is active
    opt_cost = opt_funded_df["estimated_cost_k"].sum()
    opt_employees_used = len(result["assign"])

    base_funded_df = projects[projects["project_id"].isin(baseline["funded_ids"])]
    base_value = baseline["total_value_k"]
    base_cost = baseline["total_cost_k"]
    base_employees_used = baseline["employees_used"]

    lift = (opt_value - base_value) / base_value * 100 if base_value > 0 else float("nan")

    st.subheader("Optimizer vs. naive baseline")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimizer value", f"${opt_value:,.0f}K")
    col2.metric("Baseline value", f"${base_value:,.0f}K")
    col3.metric("Lift", f"{lift:+.1f}%")
    col4.metric("Solver status", result["status"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Projects funded (optimizer)", len(opt_funded_ids), delta=len(opt_funded_ids) - len(baseline["funded_ids"]))
    col2.metric("Budget used (optimizer)", f"${opt_cost:,.0f}K", delta=f"{opt_cost - base_cost:+,.0f}K vs baseline")
    col3.metric("Employees used (optimizer)", f"{opt_employees_used}/{employee_count}",
                delta=opt_employees_used - base_employees_used)

    st.divider()
    st.subheader("Funded portfolio breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(region_tier_chart(opt_funded_df, projects, "region", "By region — Optimizer"),
                         use_container_width=True)
    with c2:
        st.plotly_chart(region_tier_chart(opt_funded_df, projects, "priority_tier", "By priority tier — Optimizer"),
                         use_container_width=True)

    region_shortfall = opt_funded_df.groupby("region").size()
    shortfalls = {r: c for r in projects["region"].unique()
                  for c in [region_shortfall.get(r, 0)] if c < min_region}
    if shortfalls:
        st.warning(f"Regions below the minimum in the optimizer result: {shortfalls} "
                   "(can happen if must-fund projects crowd out that region's budget/headcount share).")

    st.divider()
    st.subheader("Funded projects (optimizer)")
    display_cols = ["project_id", "priority_tier", "region", "required_skills",
                     "estimated_cost_k", "predicted_value_k", "must_fund"]
    st.dataframe(
        opt_funded_df[display_cols].sort_values("predicted_value_k", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )

    with st.expander("How this works"):
        st.markdown(
            "- **Predict**: a regression model (`predict_value.py`) estimates each project's expected "
            "value from its attributes (cost, tier, region, skills, duration) — not its true outcome.\n"
            "- **Optimize**: OR-Tools CP-SAT (`optimize.py`) chooses which projects to fund and which "
            "employee covers which required skill, maximizing total predicted value subject to budget, "
            "headcount, skill-matching, regional minimums, and must-fund mandates.\n"
            "- **Baseline**: a naive heuristic that simply funds the highest-predicted-value projects "
            "first, with no lookahead and no attempt to balance regions — this is what most "
            "spreadsheet-driven prioritization looks like in practice.\n"
            "- Employee counts above 80 are simulated by bootstrap-resampling the existing employee "
            "skill/cost profiles, not real additional hires."
        )


if __name__ == "__main__":
    main()
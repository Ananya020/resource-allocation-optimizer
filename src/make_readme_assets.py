"""
make_readme_assets.py
----------------------
Generates a handful of polished, presentation-quality PNGs for the README,
built from files the pipeline already produced:
  - data/processed/allocation_result.csv + projects_with_predictions.csv
  - data/processed/sensitivity_budget.csv / sensitivity_headcount.csv / sensitivity_weights.csv

These are matplotlib (not Plotly) so they don't need Chrome/kaleido to export
— useful since the interactive Plotly HTML files already cover the "explore
it yourself" use case; these PNGs are just for the README to look good on
GitHub without anyone needing to open an HTML file.

Run:
    python src/make_readme_assets.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

COLORS = {"revenue": "#EF553B", "social_impact": "#00CC96", "risk_reduction": "#636EFA",
          "optimizer": "#2E5EAA", "baseline": "#B0B0B0"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def chart_optimizer_vs_baseline():
    alloc = pd.read_csv(DATA / "allocation_result.csv")
    projects = pd.read_csv(DATA / "projects_with_predictions.csv")
    merged = alloc.merge(projects, on="project_id")

    opt = merged[merged["funded_optimal"]]
    base = merged[merged["funded_baseline"]]

    metrics = ["Projects funded", "Total value ($K)"]
    opt_vals = [len(opt), opt["predicted_value_k"].sum()]
    base_vals = [len(base), base["predicted_value_k"].sum()]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, metric, ov, bv in zip(axes, metrics, opt_vals, base_vals):
        bars = ax.bar(["Optimizer\n(CP-SAT)", "Naive\nBaseline"], [ov, bv],
                       color=[COLORS["optimizer"], COLORS["baseline"]], width=0.55)
        ax.set_title(metric, fontweight="bold")
        for b, v in zip(bars, [ov, bv]):
            ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(ov, bv) * 1.2)
    lift = (opt_vals[1] - base_vals[1]) / base_vals[1] * 100
    fig.suptitle(f"Optimizer beats the naive baseline by {lift:.1f}% in total portfolio value",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS / "optimizer_vs_baseline.png", dpi=160)
    plt.close()
    print(f"Saved optimizer_vs_baseline.png (lift = {lift:.1f}%)")


def chart_budget_sensitivity():
    df = pd.read_csv(DATA / "sensitivity_budget.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2E5EAA" if s == "OPTIMAL" else "#F2A104" for s in df["status"]]
    ax.plot(df["budget_fraction"] * 100, df["total_value_k"], color="#B0B0B0", zorder=1, linewidth=1.5)
    ax.scatter(df["budget_fraction"] * 100, df["total_value_k"], c=colors, s=90, zorder=2, edgecolor="white")
    ax.set_xlabel("Budget (% of total project cost)")
    ax.set_ylabel("Total portfolio value ($K)")
    ax.set_title("Portfolio value vs. budget (\u00b120% sweep)", fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E5EAA", markersize=9, label="OPTIMAL"),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#F2A104", markersize=9, label="FEASIBLE")]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(ASSETS / "sensitivity_budget.png", dpi=160)
    plt.close()
    print("Saved sensitivity_budget.png")


def chart_headcount_sensitivity():
    df = pd.read_csv(DATA / "sensitivity_headcount.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    feasible = df[df["status"] != "INFEASIBLE"]
    infeasible = df[df["status"] == "INFEASIBLE"]

    colors = ["#2E5EAA" if s == "OPTIMAL" else "#F2A104" for s in feasible["status"]]
    ax.plot(feasible["employee_count"], feasible["total_value_k"], color="#B0B0B0", zorder=1, linewidth=1.5)
    ax.scatter(feasible["employee_count"], feasible["total_value_k"], c=colors, s=90, zorder=2, edgecolor="white")

    if len(infeasible):
        ax.scatter(infeasible["employee_count"], [0] * len(infeasible), marker="x", color="#D62728",
                   s=140, linewidths=3, zorder=3, label="INFEASIBLE")
        for _, row in infeasible.iterrows():
            ax.annotate("infeasible\n(skill gap)", (row["employee_count"], 0),
                        textcoords="offset points", xytext=(0, 12), ha="center",
                        fontsize=9, color="#D62728", fontweight="bold")

    ax.set_xlabel("Employee headcount")
    ax.set_ylabel("Total portfolio value ($K)")
    ax.set_title("Portfolio value vs. headcount (\u00b120% sweep)", fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E5EAA", markersize=9, label="OPTIMAL"),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#F2A104", markersize=9, label="FEASIBLE"),
               plt.Line2D([0], [0], marker="x", color="#D62728", markersize=10, linewidth=0, label="INFEASIBLE")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(ASSETS / "sensitivity_headcount.png", dpi=160)
    plt.close()
    print("Saved sensitivity_headcount.png")


def chart_weight_sensitivity():
    df = pd.read_csv(DATA / "sensitivity_weights.csv")
    tiers = ["revenue_funded", "social_impact_funded", "risk_reduction_funded"]
    labels = ["Revenue", "Social impact", "Risk reduction"]
    colors = [COLORS["revenue"], COLORS["social_impact"], COLORS["risk_reduction"]]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = [0] * len(df)
    for tier, label, color in zip(tiers, labels, colors):
        ax.bar(df["scenario"], df[tier], bottom=bottom, label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, df[tier])]

    ax.set_ylabel("Projects funded")
    ax.set_title("How priority-tier weighting reshapes the funded mix", fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(ASSETS / "sensitivity_weights.png", dpi=160)
    plt.close()
    print("Saved sensitivity_weights.png")


if __name__ == "__main__":
    chart_optimizer_vs_baseline()
    chart_budget_sensitivity()
    chart_headcount_sensitivity()
    chart_weight_sensitivity()
    print(f"\nAll README assets saved to {ASSETS}")
"""
generate_data.py
-----------------
Generates the two synthetic datasets for the Intelligent Resource Allocation
& Scenario Optimization Platform:

  1. projects.csv   — 450 candidate projects
  2. employees.csv  — 80 employee/budget units

This is NOT plain np.random noise. Every distribution is either:
  (a) calibrated against a REAL reference dataset (CAPM industry stock
      returns, from Rdatasets), or
  (b) deliberately correlated with another column, mirroring how real
      portfolios behave (e.g. revenue-tier bets are riskier than
      risk-reduction bets; scarce skills cost more; certain regions have
      higher labor costs).

Reference data
--------------
We use the CAPM dataset (Ecdat::Capm, mirrored on Rdatasets/GitHub):
monthly excess returns for three real industry portfolios from 1978-2019.
We annualize each column's mean and std, and use those numbers as the
(mean %, std %) of the expected-ROI distribution for one priority tier:

    CAPM column   annualized mean / std     -> priority_tier
    rdur (durables)     6.3% / 20.1% (widest spread) -> revenue
    rcon (consumables)  5.1% / 20.1%                 -> social_impact
    rfood (staples)     8.0% / 15.7% (tightest)       -> risk_reduction

This gives a real, citable reason why revenue-tier projects have the
widest expected-value spread (growth bets) while risk_reduction projects
are the most predictable (defensive bets) — instead of an arbitrary choice.

Run:
    python src/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "capm_reference.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PROJECTS = 450
N_EMPLOYEES = 80

REGIONS = ["North America", "Europe", "APAC", "LATAM", "MEA"]
# relative labor cost-of-living multiplier per region (illustrative, real-ish ordering)
REGION_COST_MULT = {"North America": 1.30, "Europe": 1.15, "APAC": 0.85, "LATAM": 0.70, "MEA": 0.75}

SKILLS = [
    "Data Science", "Cloud Engineering", "Digital Marketing", "Operations",
    "Finance", "Legal & Compliance", "UX/Product Design", "Software Engineering",
]
# how expensive/scarce each skill is (monthly cost multiplier), roughly realistic
SKILL_COST_MULT = {
    "Data Science": 1.35, "Cloud Engineering": 1.30, "Software Engineering": 1.20,
    "UX/Product Design": 1.05, "Finance": 1.10, "Digital Marketing": 0.95,
    "Operations": 0.90, "Legal & Compliance": 1.15,
}

TIERS = ["revenue", "social_impact", "risk_reduction"]
TIER_PROBS = [0.45, 0.25, 0.30]

# which skills each tier is more likely to need (affinity weights, not exclusive)
TIER_SKILL_AFFINITY = {
    "revenue": {"Digital Marketing": 3, "Software Engineering": 3, "Data Science": 2,
                "Cloud Engineering": 2, "UX/Product Design": 2, "Finance": 1,
                "Operations": 1, "Legal & Compliance": 1},
    "social_impact": {"Operations": 3, "Legal & Compliance": 2, "UX/Product Design": 2,
                       "Data Science": 1, "Digital Marketing": 1, "Cloud Engineering": 1,
                       "Finance": 1, "Software Engineering": 1},
    "risk_reduction": {"Finance": 3, "Legal & Compliance": 3, "Operations": 2,
                        "Cloud Engineering": 2, "Data Science": 1, "Software Engineering": 1,
                        "Digital Marketing": 1, "UX/Product Design": 1},
}


def get_calibration():
    """Pull annualized mean/std of real returns per tier from the CAPM reference data."""
    capm = pd.read_csv(RAW_PATH)
    mapping = {"revenue": "rdur", "social_impact": "rcon", "risk_reduction": "rfood"}
    calib = {}
    for tier, col in mapping.items():
        monthly_mean, monthly_std = capm[col].mean(), capm[col].std()
        calib[tier] = {
            "roi_mean_pct": monthly_mean * 12,
            "roi_std_pct": monthly_std * np.sqrt(12),
        }
    return calib


def weighted_skill_sample(tier, k):
    """Sample k distinct skills, weighted by this tier's affinity."""
    skills = list(TIER_SKILL_AFFINITY[tier].keys())
    weights = np.array(list(TIER_SKILL_AFFINITY[tier].values()), dtype=float)
    weights /= weights.sum()
    k = min(k, len(skills))
    return list(rng.choice(skills, size=k, replace=False, p=weights))


def generate_projects(calib: dict) -> pd.DataFrame:
    rows = []
    for i in range(1, N_PROJECTS + 1):
        tier = rng.choice(TIERS, p=TIER_PROBS)
        region = rng.choice(REGIONS)
        n_skills = rng.integers(1, 4)  # 1-3 required skills
        req_skills = weighted_skill_sample(tier, n_skills)

        # cost: revenue-tier projects tend to be bigger bets (bigger budgets)
        tier_cost_base = {"revenue": 220, "social_impact": 110, "risk_reduction": 140}[tier]
        estimated_cost = max(20, rng.lognormal(mean=np.log(tier_cost_base), sigma=0.55))
        estimated_cost *= REGION_COST_MULT[region]  # regional cost-of-doing-business
        estimated_cost = round(estimated_cost, 1)  # in $K

        duration_months = int(np.clip(rng.normal(6, 2.5), 1, 18))

        # ROI drawn from the tier's real-calibrated distribution, clipped to a sane floor
        roi_pct = rng.normal(calib[tier]["roi_mean_pct"], calib[tier]["roi_std_pct"])
        roi_pct = float(np.clip(roi_pct, -40, 80))

        expected_value = round(estimated_cost * (1 + roi_pct / 100), 1)
        value_std = round(estimated_cost * (calib[tier]["roi_std_pct"] / 100), 1)

        # ~5% are strategic/compliance projects that must be funded regardless of ROI
        must_fund = bool(rng.random() < (0.08 if tier == "risk_reduction" else 0.03))

        rows.append({
            "project_id": f"PRJ-{i:04d}",
            "priority_tier": tier,
            "region": region,
            "required_skills": "|".join(req_skills),
            "num_skills_required": len(req_skills),
            "estimated_cost_k": estimated_cost,
            "duration_months": duration_months,
            "expected_roi_pct": round(roi_pct, 2),
            "expected_value_k": expected_value,
            "value_std_k": value_std,
            "must_fund": must_fund,
        })
    return pd.DataFrame(rows)


def generate_employees() -> pd.DataFrame:
    rows = []
    for i in range(1, N_EMPLOYEES + 1):
        region = rng.choice(REGIONS)
        primary_skill = rng.choice(SKILLS)
        has_secondary = rng.random() < 0.4
        remaining = [s for s in SKILLS if s != primary_skill]
        secondary_skill = rng.choice(remaining) if has_secondary else None

        # capacity: most are full-time, some part-time
        capacity_fte = rng.choice([1.0, 0.5], p=[0.8, 0.2])

        base_monthly_cost = 8.0  # $K baseline
        monthly_cost = base_monthly_cost * SKILL_COST_MULT[primary_skill] * REGION_COST_MULT[region]
        monthly_cost *= (0.9 + 0.2 * rng.random())  # individual noise
        monthly_cost = round(monthly_cost * capacity_fte, 2)

        rows.append({
            "employee_id": f"EMP-{i:03d}",
            "region": region,
            "primary_skill": primary_skill,
            "secondary_skill": secondary_skill if secondary_skill else "",
            "capacity_fte": capacity_fte,
            "monthly_cost_k": monthly_cost,
        })
    return pd.DataFrame(rows)


def main():
    calib = get_calibration()
    print("Calibration pulled from real CAPM data (annualized):")
    for tier, stats in calib.items():
        print(f"  {tier:15s} roi_mean={stats['roi_mean_pct']:.2f}%  roi_std={stats['roi_std_pct']:.2f}%")

    projects = generate_projects(calib)
    employees = generate_employees()

    projects.to_csv(OUT_DIR / "projects.csv", index=False)
    employees.to_csv(OUT_DIR / "employees.csv", index=False)

    print(f"\nGenerated {len(projects)} projects -> {OUT_DIR / 'projects.csv'}")
    print(f"Generated {len(employees)} employees -> {OUT_DIR / 'employees.csv'}")

    print("\n--- projects.csv summary ---")
    print(projects.groupby("priority_tier")[["estimated_cost_k", "expected_value_k", "expected_roi_pct"]]
          .agg(["mean", "std"]).round(1))
    print("\nTotal budget needed to fund every project: ${:.1f}K".format(projects["estimated_cost_k"].sum()))
    print("Must-fund projects:", projects["must_fund"].sum())

    print("\n--- employees.csv summary ---")
    print(employees.groupby("primary_skill")["monthly_cost_k"].agg(["count", "mean"]).round(2))


if __name__ == "__main__":
    main()
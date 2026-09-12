"""
optimize.py
------------
The "optimize" half of predict-then-optimize.

Two intertwined decisions, solved jointly with OR-Tools CP-SAT:
  fund[p]      -> should project p be funded at all? (bool)
  assign[e,p]  -> is employee e staffed on project p? (bool)

Two SEPARATE constrained resources (deliberately not merged into one budget):
  - Money budget   : sum of estimated_cost_k over funded projects <= BUDGET_K
  - Headcount      : each of the 80 employees can be assigned to at most 1 project
Skills are what tie them together: a project can only be funded if, for EVERY
skill it requires, at least one assigned employee actually has that skill.

Hard business constraints:
  - must_fund projects are forced to fund[p] = 1 (compliance/strategic mandates)
  - every region must get at least MIN_PROJECTS_PER_REGION funded projects

Objective: maximize total PREDICTED value of funded projects (predicted_value_k,
from predict_value.py) — not the true expected_value_k. We only ever get to
act on what we predicted, same as a real system would.

A naive "greedy highest-value-first" baseline is also computed, so we can
report the % lift the optimizer provides — the number that actually makes
this project readable to a non-technical hiring manager.

Run:
    python src/optimize.py
"""

import argparse
import pandas as pd
from pathlib import Path
from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_PATH = ROOT / "data" / "processed" / "projects_with_predictions.csv"
EMPLOYEES_PATH = ROOT / "data" / "processed" / "employees.csv"
OUT_DIR = ROOT / "data" / "processed"

SCALE = 10  # CP-SAT needs integers; we keep 1 decimal of precision on $K values


def load_data():
    projects = pd.read_csv(PROJECTS_PATH)
    employees = pd.read_csv(EMPLOYEES_PATH)
    employees["secondary_skill"] = employees["secondary_skill"].fillna("")
    projects["required_skills"] = projects["required_skills"].fillna("")
    return projects, employees


def employee_skills(emp_row):
    skills = {emp_row["primary_skill"]}
    if emp_row["secondary_skill"]:
        skills.add(emp_row["secondary_skill"])
    return skills


def build_valid_pairs(projects, employees):
    """Only create assign[e,p] variables where e actually has a skill p needs
    (keeps the model small — no point modeling irrelevant assignments)."""
    emp_skills = {row["employee_id"]: employee_skills(row) for _, row in employees.iterrows()}
    pairs = []
    for _, prow in projects.iterrows():
        req = set(s for s in prow["required_skills"].split("|") if s)
        if not req:
            continue
        for eid, eskills in emp_skills.items():
            if eskills & req:
                pairs.append((eid, prow["project_id"]))
    return pairs, emp_skills


def solve(projects, employees, budget_k, min_projects_per_region, time_limit_sec=30, tier_weights=None):
    """tier_weights: optional dict like {"revenue": 2.0} to bias the objective
    toward a priority tier. Defaults to 1.0 (no bias) for every tier not listed."""
    if tier_weights is None:
        tier_weights = {}
    pairs, emp_skills = build_valid_pairs(projects, employees)

    model = cp_model.CpModel()

    fund = {p: model.NewBoolVar(f"fund_{p}") for p in projects["project_id"]}
    assign = {(e, p): model.NewBoolVar(f"assign_{e}_{p}") for (e, p) in pairs}

    # index helpers
    assign_by_emp = {}
    assign_by_proj = {}
    for (e, p) in pairs:
        assign_by_emp.setdefault(e, []).append(assign[(e, p)])
        assign_by_proj.setdefault(p, []).append((e, assign[(e, p)]))

    # 1. each employee staffed on at most one project
    for e, var_list in assign_by_emp.items():
        model.Add(sum(var_list) <= 1)

    # 2. can't staff an unfunded project
    for (e, p), var in assign.items():
        model.Add(var <= fund[p])

    # 3. skill coverage: every required skill of a funded project needs >=1 matching employee
    for _, prow in projects.iterrows():
        p = prow["project_id"]
        req = set(s for s in prow["required_skills"].split("|") if s)
        if not req:
            model.Add(fund[p] == 0)  # malformed row safety net, shouldn't occur
            continue
        for skill in req:
            matching_vars = [assign[(e, p)] for (e, _v) in assign_by_proj.get(p, []) if skill in emp_skills[e]]
            if matching_vars:
                model.Add(sum(matching_vars) >= fund[p])
            else:
                model.Add(fund[p] == 0)  # nobody on staff has this skill at all -> can never be funded

    # 4. money budget
    cost_int = {p: int(round(c * SCALE)) for p, c in zip(projects["project_id"], projects["estimated_cost_k"])}
    model.Add(sum(cost_int[p] * fund[p] for p in fund) <= int(round(budget_k * SCALE)))

    # 5. regional minimums
    for region, group in projects.groupby("region"):
        region_projects = list(group["project_id"])
        model.Add(sum(fund[p] for p in region_projects) >= min_projects_per_region)

    # 6. must-fund mandates
    must_fund_ids = projects.loc[projects["must_fund"], "project_id"].tolist()
    for p in must_fund_ids:
        model.Add(fund[p] == 1)

    # objective: maximize (weighted) predicted value of funded projects
    tier_by_project = dict(zip(projects["project_id"], projects["priority_tier"]))
    value_int = {}
    for p, v in zip(projects["project_id"], projects["predicted_value_k"]):
        w = tier_weights.get(tier_by_project[p], 1.0)
        value_int[p] = int(round(v * w * SCALE))
    model.Maximize(sum(value_int[p] * fund[p] for p in fund))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    status_name = solver.StatusName(status)
    result = {"status": status_name, "fund": {}, "assign": {}}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result["fund"] = {p: bool(solver.Value(fund[p])) for p in fund}
        result["assign"] = {(e, p): bool(solver.Value(v)) for (e, p), v in assign.items() if solver.Value(v)}
        result["objective_value_k"] = solver.ObjectiveValue() / SCALE
    return result, must_fund_ids


def greedy_baseline(projects, employees, budget_k, min_projects_per_region):
    """Naive baseline: fund highest predicted-value projects first, first-come
    skill/budget availability, no lookahead, no attempt at balancing regions.
    Must-fund projects are still forced (any real manager would honor those)."""
    emp_skills = {row["employee_id"]: employee_skills(row) for _, row in employees.iterrows()}
    available_emp = set(employees["employee_id"])
    budget_left = budget_k
    funded = []

    must_fund_rows = projects[projects["must_fund"]].sort_values("predicted_value_k", ascending=False)
    rest_rows = projects[~projects["must_fund"]].sort_values("predicted_value_k", ascending=False)
    ordered = pd.concat([must_fund_rows, rest_rows])

    for _, prow in ordered.iterrows():
        req = set(s for s in prow["required_skills"].split("|") if s)
        cost = prow["estimated_cost_k"]
        is_must = prow["must_fund"]

        if not is_must and cost > budget_left:
            continue

        # try to find one available employee per required skill
        chosen = {}
        ok = True
        for skill in req:
            match = next((e for e in available_emp
                          if e not in chosen.values() and skill in emp_skills[e]), None)
            if match is None:
                ok = False
                break
            chosen[skill] = match

        if not ok:
            if is_must:
                funded.append({"project_id": prow["project_id"], "staffed": False})
            continue

        for e in chosen.values():
            available_emp.discard(e)
        budget_left -= cost
        funded.append({"project_id": prow["project_id"], "staffed": True})

    funded_ids = {f["project_id"] for f in funded if f["staffed"]}
    total_value = projects.loc[projects["project_id"].isin(funded_ids), "predicted_value_k"].sum()
    total_cost = projects.loc[projects["project_id"].isin(funded_ids), "estimated_cost_k"].sum()
    region_counts = projects.loc[projects["project_id"].isin(funded_ids)].groupby("region").size()

    return {
        "funded_ids": funded_ids,
        "total_value_k": total_value,
        "total_cost_k": total_cost,
        "employees_used": len(employees) - len(available_emp),
        "region_counts": region_counts.to_dict(),
        "region_min_violations": {r: c for r, c in region_counts.items() if c < min_projects_per_region},
    }


def summarize(label, funded_ids, projects, employees_used, total_cost_k, total_value_k, min_projects_per_region):
    funded_df = projects[projects["project_id"].isin(funded_ids)]
    region_counts = funded_df.groupby("region").size().to_dict()
    print(f"\n=== {label} ===")
    print(f"Projects funded : {len(funded_ids)} / {len(projects)}")
    print(f"Employees used  : {employees_used} / 80")
    print(f"Budget spent    : ${total_cost_k:,.1f}K")
    print(f"Total value     : ${total_value_k:,.1f}K")
    print(f"By region       : {region_counts}")
    shortfalls = {r: c for r, c in region_counts.items() if c < min_projects_per_region}
    if shortfalls:
        print(f"  ! Below regional minimum ({min_projects_per_region}): {shortfalls}")
    by_tier = funded_df.groupby("priority_tier").size().to_dict()
    print(f"By tier         : {by_tier}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget_fraction", type=float, default=0.25,
                         help="Fraction of total project cost available as budget")
    parser.add_argument("--min_projects_per_region", type=int, default=3)
    parser.add_argument("--time_limit", type=int, default=30)
    args = parser.parse_args()

    projects, employees = load_data()
    budget_k = projects["estimated_cost_k"].sum() * args.budget_fraction
    print(f"Total cost of ALL projects : ${projects['estimated_cost_k'].sum():,.1f}K")
    print(f"Budget available ({args.budget_fraction:.0%})   : ${budget_k:,.1f}K")
    print(f"Employees available        : {len(employees)}")
    print(f"Must-fund projects         : {projects['must_fund'].sum()}")

    result, must_fund_ids = solve(projects, employees, budget_k, args.min_projects_per_region, args.time_limit)
    print(f"\nCP-SAT solver status: {result['status']}")

    if result["status"] not in ("OPTIMAL", "FEASIBLE"):
        print("No feasible solution found. Likely cause: a must-fund project needs a "
              "skill no employee has, or the regional minimum can't be met alongside "
              "the must-fund set. This itself is a real finding (a hiring gap) — "
              "next step would be relaxing constraints one at a time to locate it.")
        return

    opt_funded_ids = {p for p, v in result["fund"].items() if v}
    opt_employees_used = len(result["assign"])
    opt_cost = projects.loc[projects["project_id"].isin(opt_funded_ids), "estimated_cost_k"].sum()
    summarize("CP-SAT Optimal Allocation", opt_funded_ids, projects, opt_employees_used,
               opt_cost, result["objective_value_k"], args.min_projects_per_region)

    baseline = greedy_baseline(projects, employees, budget_k, args.min_projects_per_region)
    summarize("Naive Greedy Baseline", baseline["funded_ids"], projects, baseline["employees_used"],
               baseline["total_cost_k"], baseline["total_value_k"], args.min_projects_per_region)

    lift = (result["objective_value_k"] - baseline["total_value_k"]) / baseline["total_value_k"] * 100
    print(f"\n>>> Optimizer improves total portfolio value by {lift:.1f}% over the naive baseline <<<")

    # save the allocation for the Streamlit app / sensitivity analysis to reuse
    alloc_rows = []
    for p in projects["project_id"]:
        alloc_rows.append({
            "project_id": p,
            "funded_optimal": p in opt_funded_ids,
            "funded_baseline": p in baseline["funded_ids"],
        })
    pd.DataFrame(alloc_rows).to_csv(OUT_DIR / "allocation_result.csv", index=False)
    print(f"\nSaved allocation -> {OUT_DIR / 'allocation_result.csv'}")


if __name__ == "__main__":
    main()
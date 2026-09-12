"""
predict_value.py
-----------------
The "predict" half of predict-then-optimize.

In a real allocation system you never know a project's true future value in
advance — you only know its attributes (cost, tier, region, skills needed,
duration). So here we pretend expected_value_k is unknown, and train a
regression model to predict it from those attributes alone.

Important: we deliberately EXCLUDE `expected_roi_pct` and `value_std_k` from
the features. Both are downstream of the same random draw used to build the
true expected_value_k, so including them would leak the answer straight into
the model (it would just learn value = cost * (1 + roi/100) instead of
learning real patterns). That would give a fake, useless R^2 close to 1.0.

Two models are trained and compared:
  1. Linear Regression  -> honest, interpretable baseline
  2. Random Forest      -> captures non-linear interactions, gives
                            feature importances for free

Outputs:
  - data/processed/projects_with_predictions.csv  (adds predicted_value_k)
  - models/best_model.joblib                       (the winning model)
  - reports/feature_importance.png                 (Random Forest importances)
  - reports/actual_vs_predicted.png                (model fit sanity check)

Run:
    python src/predict_value.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "projects.csv"
OUT_CSV = ROOT / "data" / "processed" / "projects_with_predictions.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(parents=True, exist_ok=True)

SKILLS = [
    "Data Science", "Cloud Engineering", "Digital Marketing", "Operations",
    "Finance", "Legal & Compliance", "UX/Product Design", "Software Engineering",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw project columns into a model-ready feature matrix."""
    feats = df.copy()

    # multi-hot encode required_skills ("Data Science|Finance" -> has_skill_Data_Science=1, ...)
    for skill in SKILLS:
        col = "has_skill_" + skill.replace(" ", "_").replace("&", "and").replace("/", "_")
        feats[col] = feats["required_skills"].apply(lambda s: int(skill in str(s).split("|")))

    feats["must_fund_flag"] = feats["must_fund"].astype(int)

    # one-hot encode tier and region
    feats = pd.get_dummies(feats, columns=["priority_tier", "region"], prefix=["tier", "region"])

    feature_cols = (
        ["estimated_cost_k", "duration_months", "num_skills_required", "must_fund_flag"]
        + [c for c in feats.columns if c.startswith("has_skill_")]
        + [c for c in feats.columns if c.startswith("tier_") or c.startswith("region_")]
    )
    return feats, feature_cols


def main():
    df = pd.read_csv(IN_PATH)
    feats, feature_cols = build_features(df)

    X = feats[feature_cols]
    y = feats["expected_value_k"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=SEED),
    }

    results = {}
    fitted = {}
    test_preds_by_model = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        test_preds_by_model[name] = preds  # keep the honest, held-out predictions for plotting
        results[name] = {
            "R2": r2_score(y_test, preds),
            "MAE_k": mean_absolute_error(y_test, preds),
            "RMSE_k": np.sqrt(mean_squared_error(y_test, preds)),
        }
        fitted[name] = model

    print("--- Model comparison (test set, $K units) ---")
    print(pd.DataFrame(results).T.round(3))

    best_name = max(results, key=lambda n: results[n]["R2"])
    best_test_preds = test_preds_by_model[best_name]  # honest predictions, saved before refit
    print(f"\nBest model: {best_name} (R2={results[best_name]['R2']:.3f})")

    # refit the winning model on ALL data for the final production predictions
    # (this is a fresh, separate model instance for full deployment; the test-set
    # predictions above are kept untouched for an honest actual-vs-predicted plot)
    best_model = models[best_name]
    best_model.fit(X, y)
    df["predicted_value_k"] = best_model.predict(X)
    df.to_csv(OUT_CSV, index=False)
    joblib.dump({"model": best_model, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"\nSaved predictions -> {OUT_CSV}")
    print(f"Saved model -> {MODEL_PATH}")

    # feature importance plot (Random Forest only — Linear Reg coefficients aren't
    # directly comparable across differently-scaled features)
    rf = fitted["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)
    plt.figure(figsize=(8, 6))
    importances.iloc[::-1].plot(kind="barh")
    plt.title("Random Forest — top feature importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # actual vs predicted sanity check plot — uses the honest, held-out test predictions
    # saved BEFORE the full-data refit above, so this is not data-leaked
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_test_preds, alpha=0.4)
    lims = [min(y_test.min(), 0), y_test.max()]
    plt.plot(lims, lims, "r--", label="perfect prediction")
    plt.xlabel("Actual expected_value_k")
    plt.ylabel("Predicted expected_value_k")
    plt.title(f"{best_name}: Actual vs Predicted (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    print(f"Saved plots -> {REPORTS_DIR}/feature_importance.png, actual_vs_predicted.png")


if __name__ == "__main__":
    main()

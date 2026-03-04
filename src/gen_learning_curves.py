"""Generate learning curve data for all classification models.
Trains each model on increasing fractions of training data and records
train/val scores at each step. Outputs JSON for the web dashboard.
"""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _derive_insurance_quality(df):
    net_map = {"gold": 3, "silver": 2, "bronze": 1, "tier1": 3, "tier 1": 3, "tier2": 2, "tier 2": 2, "tier3": 1, "tier 3": 1}
    plan_map = {"ppo": 3, "epo": 2, "hmo": 1, "premium": 3, "standard": 2, "basic": 1}
    net = df["network_tier"].astype(str).str.strip().str.lower().map(net_map).fillna(2).astype(float)
    plan = df["plan_type"].astype(str).str.strip().str.lower().map(plan_map).fillna(2).astype(float)
    deductible = pd.to_numeric(df["deductible"], errors="coerce")
    premium = pd.to_numeric(df["monthly_premium"], errors="coerce")

    def _qscore(s, q=3, lower_is_better=True):
        r = s.rank(method="average")
        bins = pd.qcut(r, q=q, duplicates="drop")
        k = bins.cat.categories.size
        out = pd.qcut(r, q=q, labels=list(range(1, k + 1)), duplicates="drop").astype(float)
        return (k + 1) - out if lower_is_better else out

    score = 0.35 * plan + 0.30 * net + 0.20 * _qscore(deductible).fillna(2) + 0.15 * _qscore(premium).fillna(2)
    qr = score.rank(method="average")
    return pd.qcut(qr, q=3, labels=["low", "medium", "high"], duplicates="drop")


def main():
    df = pd.read_csv("medical_insurance.csv")
    df["insurance_quality"] = _derive_insurance_quality(df)

    experiments = {
        "original": {"target": "insurance_quality", "drop": ["insurance_quality"]},
        "high_risk": {"target": "is_high_risk", "drop": ["is_high_risk"]},
        "removed_stuff": {"target": "insurance_quality", "drop": ["insurance_quality", "is_high_risk", "risk_score"]},
    }

    models = {
        "gbr": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "logreg_l1": LogisticRegression(solver="saga", penalty="l1", C=10.0, max_iter=3000),
        "logreg_l2": LogisticRegression(solver="saga", penalty="l2", C=10.0, max_iter=3000),
    }

    fractions = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    results = {}

    for exp_name, exp_cfg in experiments.items():
        print(f"Experiment: {exp_name}")
        y = df[exp_cfg["target"]].astype(str)
        X = df.drop(columns=exp_cfg["drop"], errors="ignore")

        # Only keep a subsample for speed (20K rows)
        if len(X) > 20000:
            X, _, y, _ = train_test_split(X, y, train_size=20000, random_state=42, stratify=y)

        numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ])

        exp_results = {}
        for model_name, model in models.items():
            print(f"  Model: {model_name}...", end=" ", flush=True)
            pipe = Pipeline([("preprocess", pre), ("model", model)])

            train_sizes_abs, train_scores, val_scores = learning_curve(
                pipe, X, y,
                train_sizes=fractions,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                random_state=42,
            )

            exp_results[model_name] = {
                "train_sizes": train_sizes_abs.tolist(),
                "train_mean": train_scores.mean(axis=1).tolist(),
                "train_std": train_scores.std(axis=1).tolist(),
                "val_mean": val_scores.mean(axis=1).tolist(),
                "val_std": val_scores.std(axis=1).tolist(),
            }
            print(f"done (final val acc: {val_scores.mean(axis=1)[-1]:.4f})")

        results[exp_name] = exp_results

    out_path = "web/public/data/learning_curves.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

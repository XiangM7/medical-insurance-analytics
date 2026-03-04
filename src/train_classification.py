# src/train_classification.py
import argparse
import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# silence noisy warnings so terminal output is readable
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _infer_numeric_and_categorical(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def _derive_insurance_quality(
    df: pd.DataFrame,
    deductible_col="deductible",
    network_col="network_tier",
    plan_col="plan_type",
    monthly_premium_col="monthly_premium",
    annual_premium_col="annual_premium",
):
    """
    Derive insurance_quality (low/medium/high) from plan/network and cost-sharing proxies.
    Robust to repeated values (no qcut label mismatch). Does NOT require copay.
    """
    missing = [c for c in [deductible_col, network_col, plan_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot derive insurance_quality; missing columns: {missing}")

    net_map = {
        "gold": 3, "silver": 2, "bronze": 1,
        "tier1": 3, "tier 1": 3, "tier2": 2, "tier 2": 2, "tier3": 1, "tier 3": 1,
    }
    plan_map = {
        "ppo": 3, "epo": 2, "hmo": 1,
        "premium": 3, "standard": 2, "basic": 1,
    }

    net = df[network_col].astype(str).str.strip().str.lower().map(net_map).fillna(2).astype(float)
    plan = df[plan_col].astype(str).str.strip().str.lower().map(plan_map).fillna(2).astype(float)

    deductible = pd.to_numeric(df[deductible_col], errors="coerce")

    if monthly_premium_col in df.columns:
        premium = pd.to_numeric(df[monthly_premium_col], errors="coerce")
    elif annual_premium_col in df.columns:
        premium = pd.to_numeric(df[annual_premium_col], errors="coerce")
    else:
        premium = pd.Series([float("nan")] * len(df), index=df.index)

    def _robust_quantile_score(series: pd.Series, q: int = 3, lower_is_better: bool = True) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        r = s.rank(method="average")
        bins = pd.qcut(r, q=q, duplicates="drop")
        k = bins.cat.categories.size
        labels = list(range(1, k + 1))
        out = pd.qcut(r, q=q, labels=labels, duplicates="drop").astype(float)
        if lower_is_better:
            return (k + 1) - out  # invert so lower values -> higher score
        return out

    ded_score = _robust_quantile_score(deductible, q=3, lower_is_better=True)
    prem_score = _robust_quantile_score(premium, q=3, lower_is_better=True)

    # composite score (higher = better)
    score = (
        0.35 * plan
        + 0.30 * net
        + 0.20 * ded_score.fillna(ded_score.median())
        + 0.15 * prem_score.fillna(prem_score.median())
    )

    qr = score.rank(method="average")
    bins = pd.qcut(qr, q=3, duplicates="drop")
    k = bins.cat.categories.size

    if k == 1:
        return pd.Series(["medium"] * len(df), index=df.index, dtype=str)
    if k == 2:
        q2 = pd.qcut(qr, q=2, labels=["low", "high"], duplicates="drop")
        return q2.astype(str)

    q3 = pd.qcut(qr, q=3, labels=["low", "medium", "high"], duplicates="drop")
    return q3.astype(str)


def _evaluate(y_true, y_pred, y_proba=None):
    out = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    labels = np.unique(np.array(list(y_true) + list(y_pred), dtype=object)).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["confusion_labels"] = labels
    out["confusion_matrix"] = cm.tolist()

    if y_proba is not None:
        if len(labels) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            out["roc_auc_ovr_macro"] = float(
            )
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="medical_insurance.csv")
    ap.add_argument("--target", type=str, default="is_high_risk")
    ap.add_argument("--derive-target", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--models", type=str, default="logreg_l2,logreg_l1")
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--outdir", type=str, default="results_cls")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)

    if args.derive_target:
        df = df.copy()
        df[args.target] = _derive_insurance_quality(df)

    if args.target not in df.columns:
        raise SystemExit(
            f"Target column '{args.target}' not found. Use --derive-target or choose an existing target."
        )

    y = df[args.target].astype(str)
    # removed automatically selecting target temporarily, was not working with removing mutliple columns
    X = df.drop(columns=["is_high_risk", "risk_score"])

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    val_frac_of_trainval = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac_of_trainval,
        random_state=args.random_state,
        stratify=y_trainval,
    )

    numeric_cols, categorical_cols = _infer_numeric_and_categorical(X_train)
    pre = _build_preprocessor(numeric_cols, categorical_cols)

    # Use saga for both L2 and L1 so multiclass is supported.
    model_defs = {
        "logreg_l2": LogisticRegression(
            solver="saga",
            max_iter=5000,
            penalty="l2",
        ),
        "logreg_l1": LogisticRegression(
            solver="saga",
            max_iter=5000,
            penalty="l1",
        ),
        "rf": RandomForestClassifier(random_state=args.random_state, n_jobs=-1),
        "gbr": GradientBoostingClassifier(random_state=args.random_state),
    }

    param_grids = {
        "logreg_l2": {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        "logreg_l1": {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        "rf": {"model__n_estimators": [400], "model__max_depth": [None, 12]},
        "gbr": {"model__n_estimators": [300], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
    }

    chosen = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in chosen:
        if m not in model_defs:
            raise SystemExit(f"Unknown model '{m}'. Choose from: {sorted(model_defs.keys())}")

    rows = []
    best_overall = None
    best_key = None
    best_metric = None

    for key in chosen:
        pipe = Pipeline(steps=[("preprocess", pre), ("model", model_defs[key])])
        grid = param_grids.get(key, {})

        search = GridSearchCV(
            pipe,
            param_grid=grid,
            scoring="f1_macro",
            cv=args.cv,
            n_jobs=args.n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        best_params = search.best_params_

        val_pred = fitted.predict(X_val)
        test_pred = fitted.predict(X_test)

        y_val_proba = fitted.predict_proba(X_val) if hasattr(fitted, "predict_proba") else None
        y_test_proba = fitted.predict_proba(X_test) if hasattr(fitted, "predict_proba") else None

        val_metrics = _evaluate(y_val, val_pred, y_val_proba)
        test_metrics = _evaluate(y_test, test_pred, y_test_proba)

        row = {
            "model": key,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_mcc": val_metrics["mcc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_mcc": test_metrics["mcc"],
            "best_params": json.dumps(best_params, sort_keys=True),
        }
        if "roc_auc" in test_metrics:
            row["test_roc_auc"] = test_metrics["roc_auc"]
        if "roc_auc_ovr_macro" in test_metrics:
            row["test_roc_auc_ovr_macro"] = test_metrics["roc_auc_ovr_macro"]

        rows.append(row)

        current = test_metrics["mcc"]
        if best_metric is None or current > best_metric:
            best_metric = current
            best_overall = fitted
            best_key = key

        cm_out = {
            "model": key,
            "val": {"labels": val_metrics["confusion_labels"], "matrix": val_metrics["confusion_matrix"]},
            "test": {"labels": test_metrics["confusion_labels"], "matrix": test_metrics["confusion_matrix"]},
        }
        with open(os.path.join(args.outdir, f"confusion_{key}.json"), "w") as f:
            json.dump(cm_out, f, indent=2)

    metrics_df = pd.DataFrame(rows).sort_values(by="test_mcc", ascending=False)
    metrics_path = os.path.join(args.outdir, "metrics_cls.csv")
    metrics_df.to_csv(metrics_path, index=False)

    if best_metric is None:
        best_metric = 0.0

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.outdir, f"best_cls_model_{best_key}_{stamp}.joblib")
    joblib.dump(best_overall, model_path)

    summary = {
        "best_model": best_key,
        "best_test_mcc": float(best_metric),
        "metrics_csv": metrics_path,
        "model_artifact": model_path,
        "target": args.target,
        "data": args.data,
        "derived_target": bool(args.derive_target),
    }
    with open(os.path.join(args.outdir, "run_cls_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
    

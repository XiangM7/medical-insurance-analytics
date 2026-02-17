# train_classification.py
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib


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


def _derive_insurance_quality(df: pd.DataFrame,
                             deductible_col="deductible",
                             copay_col="copay",
                             network_col="network_tier",
                             plan_col="plan_type"):
    missing = [c for c in [deductible_col, copay_col, network_col, plan_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot derive insurance_quality; missing columns: {missing}")

    net_map = {
        "gold": 3, "silver": 2, "bronze": 1,
        "tier1": 3, "tier 1": 3, "tier2": 2, "tier 2": 2, "tier3": 1, "tier 3": 1
    }
    plan_map = {
        "ppo": 3, "epo": 2, "hmo": 1,
        "premium": 3, "standard": 2, "basic": 1
    }

    net = df[network_col].astype(str).str.strip().str.lower().map(net_map).fillna(2)
    plan = df[plan_col].astype(str).str.strip().str.lower().map(plan_map).fillna(2)

    deductible = pd.to_numeric(df[deductible_col], errors="coerce")
    copay = pd.to_numeric(df[copay_col], errors="coerce")

    ded_score = pd.qcut(deductible.rank(method="average"), q=3, labels=[3, 2, 1])
    copay_score = pd.qcut(copay.rank(method="average"), q=3, labels=[3, 2, 1])

    score = (
        0.30 * plan.astype(float)
        + 0.25 * net.astype(float)
        + 0.25 * ded_score.astype(float)
        + 0.20 * copay_score.astype(float)
    )

    quality = pd.qcut(score.rank(method="average"), q=3, labels=["low", "medium", "high"])
    return quality.astype(str)


def _evaluate(y_true, y_pred, y_proba=None, average="macro"):
    out = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    labels = np.unique(np.array(list(y_true) + list(y_pred), dtype=object)).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["confusion_labels"] = labels
    out["confusion_matrix"] = cm.tolist()

    if y_proba is not None:
        classes = labels
        if len(classes) == 2:
            pos_idx = 1
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, pos_idx]))
            except Exception:
                out["roc_auc"] = None
        else:
            try:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
            except Exception:
                out["roc_auc_ovr_macro"] = None

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="medical_insurance.csv")
    ap.add_argument("--target", type=str, default="insurance_quality")
    ap.add_argument("--derive-target", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--models", type=str, default="logreg_l2,logreg_l1,rf,gbr")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)

    if args.derive_target:
        df = df.copy()
        df[args.target] = _derive_insurance_quality(df)

    if args.target not in df.columns:
        raise SystemExit(
            f"Target column '{args.target}' not found. "
            f"Use --derive-target to create it, or choose an existing target."
        )

    y = df[args.target].astype(str)
    X = df.drop(columns=[args.target])

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

    model_defs = {
        "logreg_l2": LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=5000, n_jobs=-1
        ),
        "logreg_l1": LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=5000
        ),
        "rf": RandomForestClassifier(random_state=args.random_state, n_jobs=-1),
        "gbr": GradientBoostingClassifier(random_state=args.random_state),
    }

    param_grids = {
        "logreg_l2": {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        "logreg_l1": {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        "rf": {"model__n_estimators": [300, 600], "model__max_depth": [None, 8, 14]},
        "gbr": {"model__n_estimators": [200, 500], "model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3]},
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

        scoring = "f1_macro"
        search = GridSearchCV(
            pipe,
            param_grid=grid,
            scoring=scoring,
            cv=args.cv,
            n_jobs=-1,
            refit=True,
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
            row["test_roc_auc"] = test_metrics.get("roc_auc")
        if "roc_auc_ovr_macro" in test_metrics:
            row["test_roc_auc_ovr_macro"] = test_metrics.get("roc_auc_ovr_macro")

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

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.outdir, f"best_cls_model_{best_key}_{stamp}.joblib")
    joblib.dump(best_overall, model_path)

    if best_metric is None:
        best_metric = 0.0
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

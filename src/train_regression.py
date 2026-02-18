# src/train_regression.py
import argparse
import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def _make_models(random_state: int):
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(random_state=random_state),
        "lasso": Lasso(random_state=random_state, max_iter=100000),
        "elasticnet": ElasticNet(random_state=random_state, max_iter=100000),
        "rf": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "gbr": GradientBoostingRegressor(random_state=random_state),
    }


def _param_grids():
    # keep grids small so it runs fast and stable
    return {
        "linear": {},
        "ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        "lasso": {"model__alpha": [0.01, 0.1, 1.0, 10.0]},
        "elasticnet": {
            "model__alpha": [0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        },
        "rf": {
            "model__n_estimators": [300],
            "model__max_depth": [None, 12],
            "model__min_samples_split": [2, 10],
        },
        "gbr": {
            "model__n_estimators": [300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    }


def _evaluate(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="medical_insurance.csv")
    ap.add_argument("--target", type=str, default="annual_medical_cost")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--models", type=str, default="linear,ridge,lasso")
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--scoring", type=str, default="neg_root_mean_squared_error")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--n-jobs", type=int, default=1)  # keep stable (parallel can be slower/noisy)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise SystemExit(
            f"Target column '{args.target}' not found. Available columns: {list(df.columns)[:25]}..."
        )

    y = df[args.target]
    X = df.drop(columns=[args.target])

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    val_frac_of_trainval = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac_of_trainval,
        random_state=args.random_state,
    )

    numeric_cols, categorical_cols = _infer_numeric_and_categorical(X_train)
    pre = _build_preprocessor(numeric_cols, categorical_cols)

    all_models = _make_models(args.random_state)
    grids = _param_grids()

    chosen = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in chosen:
        if m not in all_models:
            raise SystemExit(f"Unknown model '{m}'. Choose from: {sorted(all_models.keys())}")

    rows = []
    best_overall = None
    best_key = None
    best_metric = None

    for key in chosen:
        model = all_models[key]
        pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
        grid = grids.get(key, {})

        if grid:
            search = GridSearchCV(
                pipe,
                param_grid=grid,
                scoring=args.scoring,
                cv=args.cv,
                n_jobs=args.n_jobs,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            best_params = search.best_params_
        else:
            fitted = pipe.fit(X_train, y_train)
            best_params = {}

        val_pred = fitted.predict(X_val)
        test_pred = fitted.predict(X_test)

        val_metrics = _evaluate(y_val, val_pred)
        test_metrics = _evaluate(y_test, test_pred)

        rows.append(
            {
                "model": key,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "best_params": json.dumps(best_params, sort_keys=True),
            }
        )

        current = test_metrics["rmse"]
        if best_metric is None or current < best_metric:
            best_metric = current
            best_overall = fitted
            best_key = key

    metrics_df = pd.DataFrame(rows).sort_values(by="test_rmse", ascending=True)
    metrics_path = os.path.join(args.outdir, "metrics_reg.csv")
    metrics_df.to_csv(metrics_path, index=False)

    if best_metric is None:
        best_metric = 0.0

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.outdir, f"best_reg_model_{best_key}_{stamp}.joblib")
    joblib.dump(best_overall, model_path)

    summary = {
        "best_model": best_key,
        "best_test_rmse": float(best_metric),
        "metrics_csv": metrics_path,
        "model_artifact": model_path,
        "target": args.target,
        "data": args.data,
    }
    with open(os.path.join(args.outdir, "run_reg_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

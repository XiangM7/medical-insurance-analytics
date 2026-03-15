"""
Flask API for live model predictions.
Loads the Ridge regression model and GBR high-risk classification model,
serves predictions via REST endpoints.

Usage:
    source .venv/bin/activate
    python src/predict_api.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Load data & models at startup
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "medical_insurance.csv")

df = pd.read_csv(DATA_PATH)

# Reproduce the test split for regression (same random_state as training)
y_reg_all = df["annual_medical_cost"]
X_reg_all = df.drop(columns=["annual_medical_cost"])
_, X_test_reg, _, y_test_reg = train_test_split(
    X_reg_all, y_reg_all, test_size=0.2, random_state=42
)

# Reproduce the test split for classification
y_cls_all = df["is_high_risk"].astype(str)
X_cls_all = df.drop(columns=["is_high_risk", "risk_score"])
_, X_test_cls, _, y_test_cls = train_test_split(
    X_cls_all, y_cls_all, test_size=0.2, random_state=42, stratify=y_cls_all
)

# Load models
reg_model = joblib.load(
    os.path.join(BASE, "results", "best_reg_model_ridge_20260217_223041.joblib")
)
cls_model = joblib.load(
    os.path.join(BASE, "high_risk_all_models", "best_cls_model_gbr_retrained.joblib")
)

# Feature display names for the UI
KEY_FEATURES = [
    "age", "sex", "income", "bmi", "smoker", "chronic_count",
    "systolic_bp", "diastolic_bp", "hba1c", "ldl",
    "claims_count", "annual_medical_cost", "deductible",
    "medication_count", "hypertension", "diabetes",
    "cardiovascular_disease", "cancer_history",
]


def _patient_to_dict(row, idx):
    """Convert a DataFrame row to a JSON-safe dict with all features."""
    d = {}
    for col in row.index:
        val = row[col]
        if pd.isna(val):
            d[col] = None
        elif isinstance(val, (np.integer,)):
            d[col] = int(val)
        elif isinstance(val, (np.floating,)):
            d[col] = float(val)
        elif isinstance(val, (np.bool_,)):
            d[col] = bool(val)
        else:
            d[col] = val
    d["_index"] = int(idx)
    return d


@app.route("/api/random-patient", methods=["GET"])
def random_patient():
    """Pick a random patient from the test set."""
    idx = np.random.choice(X_test_reg.index)
    row = df.loc[idx]
    return jsonify(_patient_to_dict(row, idx))


@app.route("/api/predict", methods=["POST"])
def predict():
    """Run both models on the provided patient data."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Build DataFrames matching what each model expects
    patient_df = pd.DataFrame([data])

    # --- Regression: predict annual_medical_cost ---
    reg_cols = [c for c in X_reg_all.columns if c in patient_df.columns]
    missing_reg = [c for c in X_reg_all.columns if c not in patient_df.columns]
    reg_input = patient_df[reg_cols].copy()
    for c in missing_reg:
        reg_input[c] = np.nan
    reg_input = reg_input[X_reg_all.columns]

    pred_cost = float(reg_model.predict(reg_input)[0])
    actual_cost = float(data.get("annual_medical_cost", 0))

    # --- Classification: predict is_high_risk ---
    cls_cols = [c for c in X_cls_all.columns if c in patient_df.columns]
    missing_cls = [c for c in X_cls_all.columns if c not in patient_df.columns]
    cls_input = patient_df[cls_cols].copy()
    for c in missing_cls:
        cls_input[c] = np.nan
    cls_input = cls_input[X_cls_all.columns]

    pred_risk = cls_model.predict(cls_input)[0]
    raw_proba = cls_model.predict_proba(cls_input)[0]
    # Cap probabilities so nothing displays as 100%
    pred_proba = np.clip(raw_proba, 0.005, 0.995).tolist()
    risk_classes = cls_model.classes_.tolist()
    actual_risk = str(data.get("is_high_risk", ""))

    # --- Feature importance from GBR ---
    gbr = cls_model.named_steps["model"]
    pre = cls_model.named_steps["preprocess"]
    try:
        feat_names = pre.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"f{i}" for i in range(len(gbr.feature_importances_))]

    importances = gbr.feature_importances_.tolist()
    feat_imp = sorted(
        zip(feat_names, importances), key=lambda x: x[1], reverse=True
    )[:10]
    top_features = [{"name": n.replace("num__", "").replace("cat__", ""), "importance": round(v, 4)} for n, v in feat_imp]

    return jsonify({
        "regression": {
            "predicted_cost": round(pred_cost, 2),
            "actual_cost": round(actual_cost, 2),
            "error": round(abs(pred_cost - actual_cost), 2),
        },
        "classification": {
            "predicted_risk": pred_risk,
            "actual_risk": actual_risk,
            "probabilities": dict(zip(risk_classes, [round(p, 4) for p in pred_proba])),
            "correct": str(pred_risk) == str(actual_risk),
        },
        "top_features": top_features,
    })


if __name__ == "__main__":
    print("Models loaded. Starting API on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)

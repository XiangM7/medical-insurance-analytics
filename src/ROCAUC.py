import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc

# Load data
df = pd.read_csv("medical_insurance.csv")

# Basic structure 
print("Shape (rows, columns):", df.shape)
print("\nColumn names:\n", df.columns.tolist())

print("\nData types:\n", df.dtypes)

#  Missing values 
print("\nMissing values per column:\n", df.isna().sum())

print("\nSummary statistics (numeric and categorical):\n")
print(df.describe(include="all").transpose())

#  Correlation matrix for numeric features 
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
corr = df[num_cols].corr()
print("\nCorrelation matrix (numeric features):\n")
print(corr)

# Optionally, save key EDA outputs to CSV
corr.to_csv("eda_corr.csv", index=True)
df.describe(include="all").transpose().to_csv("eda_summary.csv")
df.isna().sum().to_csv("eda_missing.csv")
df.dtypes.to_csv("eda_dtypes.csv")

import joblib
import glob
import os

# Load the best saved classifier from train_classification.py's output 
model_files = glob.glob("results_cls/best_cls_model_*.joblib")

# Pick the most recently saved model
model_path = max(model_files, key=os.path.getmtime)
clf = joblib.load(model_path)
print(f"Loaded model: {model_path}")

#  train_classification.py drops is_high_risk and risk_score from X
TARGET_COL     = "is_high_risk"
PRED_LABEL_COL = "high_cost_pred"
PRED_SCORE_COL = "high_cost_score"

X_full = df.drop(columns=["is_high_risk", "risk_score"])

#  Generate prediction columns  
df[PRED_LABEL_COL] = clf.predict(X_full)
df[PRED_SCORE_COL] = clf.predict_proba(X_full)[:, 1]


# Confusion matrix + MCC + rates 
if (
    TARGET_COL is not None
    and PRED_LABEL_COL is not None
    and TARGET_COL in df.columns
    and PRED_LABEL_COL in df.columns
):
    y_true = df[TARGET_COL].values
    y_pred = df[PRED_LABEL_COL].values

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion matrix (TN, FP, FN, TP):")
    print(cm)

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    print("\nMatthews Correlation Coefficient (MCC):", mcc)

    # Basic rates (linked to ROC definition in slides)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # True Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan  # True Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan  # False Negative Rate

    print("True Positive Rate (Recall):", tpr)
    print("True Negative Rate (Specificity):", tnr)
    print("False Positive Rate:", fpr)
    print("False Negative Rate:", fnr)

else:
    print("\n[MCC NOTE] Set TARGET_COL and PRED_LABEL_COL to compute confusion matrix and MCC.")

# ROC curve + AUC 
if (
    TARGET_COL is not None
    and PRED_SCORE_COL is not None
    and TARGET_COL in df.columns
    and PRED_SCORE_COL in df.columns
):
    y_true = df[TARGET_COL].values
    y_score = df[PRED_SCORE_COL].values

    fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr_vals, tpr_vals)

    print("\nROC curve points (first 10):")
    for i in range(min(10, len(thresholds))):
        print(
            f"Threshold={thresholds[i]:.3f}, "
            f"FPR={fpr_vals[i]:.3f}, "
            f"TPR={tpr_vals[i]:.3f}"
        )

    print("\nArea Under the ROC Curve (AUC):", roc_auc)

else:
    print("\n[ROC/AUC NOTE] Set TARGET_COL and PRED_SCORE_COL (probabilities) to compute ROC and AUC.")



import matplotlib.pyplot as plt

def visualize_roc(fpr_vals, tpr_vals, thresholds, roc_auc, output_path="roc_curve.png"):
    fig, ax = plt.subplots(figsize=(7, 6))

    # ROC curve with fill
    ax.fill_between(fpr_vals, tpr_vals, alpha=0.10)
    ax.plot(fpr_vals, tpr_vals, lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")

    # Random-chance diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="gray", label="Random chance")

    # Threshold markers at 5 evenly-spaced points
    idx = np.linspace(0, len(thresholds) - 1, 7, dtype=int)[1:-1]
    for i in idx:
        ax.scatter(fpr_vals[i], tpr_vals[i], s=60, zorder=5)
        ax.annotate(
            f"t={thresholds[i]:.2f}",
            (fpr_vals[i], tpr_vals[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("FPR (1 − Specificity)", fontsize=11)
    ax.set_ylabel("TPR (Recall / Sensitivity)", fontsize=11)
    ax.set_title("ROC Curve — Medical Insurance Classifier", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if (
    TARGET_COL is not None
    and PRED_SCORE_COL is not None
    and TARGET_COL in df.columns
    and PRED_SCORE_COL in df.columns
):
    y_true  = df[TARGET_COL].values
    y_score = df[PRED_SCORE_COL].values

    fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr_vals, tpr_vals)

    print("\nROC curve points (first 10):")
    for i in range(min(10, len(thresholds))):
        print(f"  Threshold={thresholds[i]:.3f}, FPR={fpr_vals[i]:.3f}, TPR={tpr_vals[i]:.3f}")

    #path-force printing ROC graph
    from pathlib import Path
    import os

    downloads = Path.home() / "Downloads"
    downloads.mkdir(exist_ok=True)  
    output_path = str(downloads / "roc_curve.png")
    print("Saving ROC curve to:", output_path)

    visualize_roc(fpr_vals, tpr_vals, thresholds, roc_auc, output_path=output_path)

    print("File exists:", os.path.exists(output_path))

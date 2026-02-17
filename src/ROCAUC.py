import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc

# Load data
df = pd.read_csv("medical_insurance.csv")

# ---- Basic structure ----
print("Shape (rows, columns):", df.shape)
print("\nColumn names:\n", df.columns.tolist())

print("\nData types:\n", df.dtypes)

# ---- Missing values ----
print("\nMissing values per column:\n", df.isna().sum())

# ---- Descriptive statistics ----
print("\nSummary statistics (numeric and categorical):\n")
print(df.describe(include="all").transpose())

# ---- Correlation matrix for numeric features ----
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
corr = df[num_cols].corr()
print("\nCorrelation matrix (numeric features):\n")
print(corr)

# Optionally, save key EDA outputs to CSV
corr.to_csv("eda_corr.csv", index=True)
df.describe(include="all").transpose().to_csv("eda_summary.csv")
df.isna().sum().to_csv("eda_missing.csv")
df.dtypes.to_csv("eda_dtypes.csv")

TARGET_COL = None      # e.g., "high_cost"
PRED_LABEL_COL = None  # e.g., "high_cost_pred" (0/1)
PRED_SCORE_COL = None  # e.g., "high_cost_score" (probability of class 1)

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

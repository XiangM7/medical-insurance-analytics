#!/usr/bin/env python3
"""
EDA script for ECS171 Medical Insurance project
Run in terminal: python EDA.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "medical_insurance.csv"   # CSV must be in same folder
OUTPUT_DIR = "eda_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=== Dataset Shape ===")
print(df.shape)

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Missing Value Rate (Top 15) ===")
print(df.isna().mean().sort_values(ascending=False).head(15))

print("\n=== Duplicate Rows ===")
print(df.duplicated().sum())

print("\n=== Numeric Summary ===")
print(df.describe())

if "is_high_risk" in df.columns:
    print("\nUsing existing column: is_high_risk as label")
    df["risk_label"] = df["is_high_risk"]
else:
    print("\nNo is_high_risk column found, defining risk using cost")
    cost_col = None
    for c in ["annual_medical_cost", "charges", "total_claims_paid"]:
        if c in df.columns:
            cost_col = c
            break

    if cost_col is None:
        raise ValueError("No cost column found to define risk")

    threshold = df[cost_col].median()
    df["risk_label"] = (df[cost_col] > threshold).astype(int)

print("\n=== Risk Label Distribution ===")
print(df["risk_label"].value_counts())
print(df["risk_label"].value_counts(normalize=True))

plt.figure()
df["risk_label"].value_counts().plot(kind="bar")
plt.title("Risk Label Distribution (0 = Low, 1 = High)")
plt.xlabel("Risk Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/risk_distribution.png")
plt.close()

numeric_features = [
    "age", "income", "bmi", "systolic_bp", "diastolic_bp",
    "ldl", "hba1c", "deductible", "copay",
    "annual_medical_cost", "annual_premium", "monthly_premium",
    "claims_count", "total_claims_paid", "chronic_count"
]

categorical_features = [
    "sex", "region", "urban_rural", "education",
    "employment_status", "smoker", "plan_type", "network_tier"
]

numeric_features = [c for c in numeric_features if c in df.columns]
categorical_features = [c for c in categorical_features if c in df.columns]

print("\nNumeric features used:", numeric_features)
print("Categorical features used:", categorical_features)

for col in numeric_features:
    plt.figure()
    plt.hist(df[col].dropna(), bins=40)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hist_{col}.png")
    plt.close()

    print(f"\n--- {col} by risk level ---")
    print(df.groupby("risk_label")[col].describe())

    plt.figure()
    low = df[df["risk_label"] == 0][col].dropna()
    high = df[df["risk_label"] == 1][col].dropna()
    plt.boxplot([low, high], labels=["Low Risk", "High Risk"])
    plt.title(f"{col} vs Risk Level")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/box_{col}.png")
    plt.close()

for col in categorical_features:
    print(f"\n--- {col} vs risk level ---")
    ct = pd.crosstab(df[col], df["risk_label"], normalize="index")
    print(ct)

    if 1 in ct.columns:
        plt.figure(figsize=(8, 4))
        ct[1].sort_values(ascending=False).plot(kind="bar")
        plt.title(f"High Risk Rate by {col}")
        plt.ylabel("P(High Risk)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/highrisk_{col}.png")
        plt.close()

corr_cols = [c for c in numeric_features if pd.api.types.is_numeric_dtype(df[c])]
if len(corr_cols) > 1:
    corr = df[corr_cols].corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=90)
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()

print("\nEDA finished.")
print(f"All figures saved to ./{OUTPUT_DIR}/")

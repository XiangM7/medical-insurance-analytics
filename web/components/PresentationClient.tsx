"use client";

import { useState, useEffect, useCallback } from "react";
import type { Experiment, RegressionMetric, FeatureStat } from "@/lib/types";
import { modelLabel, pct, fmt } from "@/lib/utils";
import AnimatedCounter from "./AnimatedCounter";
import ConfusionMatrix from "./ConfusionMatrix";

interface Props {
  experiments: Experiment[];
  regMetrics: RegressionMetric[];
  features: FeatureStat[];
  corrFeatures: string[];
  corrMatrix: number[][];
}

function SlideShell({
  children,
  slideNum,
  total,
}: {
  children: React.ReactNode;
  slideNum: number;
  total: number;
}) {
  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex-1 flex items-center justify-center p-8 overflow-y-auto">
        <div className="w-full max-w-4xl">{children}</div>
      </div>
      <div className="flex items-center justify-between px-8 pb-4 text-xs text-muted">
        <span>ECS 171  - Medical Insurance ML Project</span>
        <span>
          {slideNum} / {total}
        </span>
      </div>
    </div>
  );
}

// Speaker notes per slide (Q&A defense prep)
const speakerNotes: string[] = [
  // 0: Title
  `PROBLEM FRAMING: Insurers and healthcare providers must accurately determine a patient's financial risk and insurance quality tier. The complexity of human health, lifestyle, and socioeconomic factors makes this inherently difficult manually. Our goal is to identify patient-level features that most strongly predict risk.`,

  // 1: Problem & Applications
  `REAL-WORLD USE CASES: (1) Insurance Underwriter Tool - underwriter inputs applicant data, model recommends premium tier and risk rating, replacing slow manual underwriting. (2) Hospital Risk Stratification - administrator runs batch of patient profiles to flag high-risk patients before expensive intervention, enabling proactive care management.`,

  // 2: Dataset
  `MISSING VALUES & CLASS IMBALANCE: Only alcohol_freq has missing values at ~30% - imputed with most frequent value inside a sklearn Pipeline to prevent data leakage. For class imbalance (36.8% high-risk vs. 63.2% low-risk), using F1-macro and MCC as evaluation metrics instead of raw accuracy already accounts for this, since both metrics penalize models that simply predict the majority class.`,

  // 3: EDA - Distributions
  `OUTLIERS & FEATURE SELECTION: Income, annual medical cost, and claims are right-skewed with heavy outliers. Applied standard scaling on numeric features, which reduces their leverage without discarding real patient variation. Features like deductible and region that showed weak signal in EDA may still contribute in combination with others. We kept outliers because they represent real high-cost patients - exactly who we want to identify. Tree-based models (GBR, RF) are robust to outliers; StandardScaler handles it for logistic regression.`,

  // 4: EDA - Risk groups & correlations
  `FEATURE CORRELATIONS: Strong correlations among cost features (annual_medical_cost, claims_count, total_claims_paid) - potential redundancy/multicollinearity. High-risk patients show systematically higher costs, more claims, more chronic conditions. Current smokers had a 73% high-risk rate vs. ~31% for non-smokers, and high-risk patients averaged 59 years old vs. 41 for low-risk. We kept correlated features because tree models handle collinearity well; for logistic regression, regularization (L1/L2) handles it.`,

  // 5: Pipeline
  `OPTIMIZATION & HYPERPARAMETERS: L1 and L2 regularization via the SAGA solver for logistic regression. C (inverse of lambda) is the primary hyperparameter, tuned via grid search with cross-validation. Ridge Regression on the cost side, best alpha=0.01 found via 3-fold CV. For Random Forest and Gradient Boosting, key hyperparameters are n_estimators, max_depth, and learning_rate.

WHY LOGISTIC REGRESSION? Linear Regression is inappropriate because the target is categorical (risk tier), not continuous - it doesn't produce valid class probabilities. A single Decision Tree is prone to overfitting without pruning. LogReg with regularization offers a strong bias-variance tradeoff, produces calibrated probabilities, and its coefficients are directly interpretable - critical for healthcare where stakeholders need to understand why a patient is flagged. RF and GBR added as more expressive baselines.

VALIDATION: 80/20 train-test split with an additional 20% validation set and 3-fold cross-validation. Overfitting detected when train accuracy is significantly higher than validation accuracy. Underfitting would appear as uniformly poor performance across both sets - remedy is adding interaction terms or switching to a more expressive model.`,

  // 6: Classification Results
  `EPOCHS & TRAINING: Logistic regression and tree-based models don't train in epochs like neural networks - they converge via optimization algorithms. For SAGA, max_iter=1000 as a safe upper bound with early stopping built into the convergence criterion. For Gradient Boosting, early stopping is done by monitoring validation loss across boosting rounds and halting when it stops improving.

HYPERPARAMETER GRIDS: GBR: n_estimators=[300], learning_rate=[0.05,0.1], max_depth=[2,3]. RF: n_estimators=[400], max_depth=[None,12]. LogReg: C=[0.1,1.0,10.0], class_weight=[None,"balanced"]. Scoring metric: f1_macro. Best params selected by CV score, then evaluated on held-out val and test sets.`,

  // 7: Confusion Matrices
  `SPECIFIC PREDICTIONS: Large positive coefficient on smoker_status_current or age means those directly push the log-odds toward high-risk. Current smokers had 73% high-risk rate vs ~31% for non-smokers, high-risk patients averaged 59 years old vs 41 for low-risk. For unexpected predictions, the answer usually lies in rare feature combinations - e.g., a young patient flagged as high-risk likely has multiple chronic conditions overriding the age signal.

MODEL FAILURE CASE: Atypical high-risk patients - e.g., a young, non-smoking patient with rare combination of kidney disease + cancer history who is genuinely high-risk but looks low-risk on dominant features (age, smoking). This reveals the model's reliance on population-level statistical patterns rather than rare but clinically meaningful combinations. Also exposes logistic regression's linearity limitation - can't capture complex interaction effects without explicitly engineering them.`,

  // 8: Regression
  `MODEL SIZE REDUCTION FOR MOBILE: Start by dropping low-importance features via L1 sparsity - logistic regression with L1 regularization zeros out irrelevant coefficients, performing automatic feature selection. EDA flagged as uninformative: sex, region, urban/rural status, plan type, employment status are first candidates. Reduce from 54 features to ~15-20 high-signal features (age, smoking, chronic conditions, blood pressure, claims count) with minimal performance loss. For tree-based models, reduce max_depth and n_estimators. Ridge regression is already tiny and achieves R2=0.966 - natural choice for deployment.`,

  // 9: Ablation
  `OVERFITTING/UNDERFITTING DETECTION: 80/20 train-test split with additional 20% validation set and 3-fold cross-validation. Overfitting detected when train accuracy significantly exceeds validation accuracy. Regularization directly mitigates overfitting. Underfitting would appear as uniformly poor performance across both sets - if that occurs, add interaction terms or switch to more expressive model (RF, GBR).

The ablation study IS our data leakage detection. Original experiment: 99.9% accuracy looks too good. Removing risk_score and is_high_risk drops to ~40%, proving those features were leaking target information. The removed-features experiment IS a failure case - insurance_quality derived from plan/network/deductible/premium is NOT predictable from health/demographic features alone.`,

  // 10: High Risk
  `BIAS CHECK: A model trained on this data may learn to flag lower-income patients as high-risk not purely due to health factors, but due to correlated socioeconomic disadvantage. Should run disaggregated performance metrics (F1, false positive rate) broken down by sex, region, and income bracket. The model may underperform on demographic subgroups underrepresented in the 100K training set - if deployed on a different population (e.g., rural-only dataset).

POTENTIAL CONCERNS: (1) Model uses features like income, region, education that correlate with race/ethnicity and could encode demographic bias. (2) is_high_risk may embed historical biases in how risk was originally assigned. (3) Model could perpetuate existing disparities - e.g., if historically underserved populations have fewer claims due to access barriers (not lower risk), model incorrectly labels them as low-risk. Mitigation requires fairness metrics across protected groups, disparate impact analysis, and potentially removing or decorrelating sensitive features.`,

  // 11: Takeaways
  `SUMMARY: Feature engineering and data integrity matter more than model complexity. The "best" model is meaningless if the features leak the target. Our ablation study demonstrates this clearly. For real deployment, you'd need features available BEFORE the risk determination is made - not derived from it. Insurance_quality as derived from plan/network/deductible/premium is not predictable from health/demographic features alone - the original high accuracy was an artifact of data leakage.`,

  // 12: Thank you
  `ADDITIONAL NOTES: The entire dashboard is built in Next.js with server-side data loading. All metrics are computed from pre-generated CSV/JSON files produced by our sklearn pipeline. The correlation heatmap, confusion matrices, and feature tables are all interactive and pull from real data.`,
];

export default function PresentationClient({
  experiments,
  regMetrics,
  features,
  corrFeatures,
  corrMatrix,
}: Props) {
  const [slide, setSlide] = useState(0);
  const [showNotes, setShowNotes] = useState(false);

  const orig = experiments.find((e) => e.id === "original")!;
  const highRisk = experiments.find((e) => e.id === "high_risk")!;
  const ablation = experiments.find((e) => e.id === "removed_stuff")!;

  const numericCount = features.filter((f) => f.dtype !== "str").length;
  const categoricalCount = features.filter((f) => f.dtype === "str").length;

  const topCorrelations = (() => {
    const pairs: { a: string; b: string; val: number }[] = [];
    for (let i = 0; i < corrFeatures.length; i++) {
      for (let j = i + 1; j < corrFeatures.length; j++) {
        const v = corrMatrix[i][j];
        if (Math.abs(v) > 0.5) {
          pairs.push({ a: corrFeatures[i], b: corrFeatures[j], val: v });
        }
      }
    }
    return pairs.sort((a, b) => Math.abs(b.val) - Math.abs(a.val)).slice(0, 8);
  })();

  const slides = [
    // 0: Title
    <div key="title" className="text-center space-y-6">
      <h1 className="text-5xl font-bold tracking-tight">
        Medical Insurance
        <br />
        <span className="text-accent">Risk Prediction</span>
      </h1>
      <p className="text-lg text-muted max-w-2xl mx-auto">
        Identifying patient-level features that predict financial risk and insurance quality
        using classification and regression models
      </p>
      <div className="flex justify-center gap-6 mt-8 text-sm text-muted">
        <span>ECS 171  - Machine Learning</span>
        <span>UC Davis</span>
      </div>
      <p className="text-xs text-muted/50 mt-12">
        Arrow keys to navigate &middot; Press <kbd className="px-1.5 py-0.5 rounded bg-card border border-border text-muted">N</kbd> for speaker notes
      </p>
    </div>,

    // 1: Problem & Applications
    <div key="problem" className="space-y-6">
      <h2 className="text-3xl font-bold">Problem & Motivation</h2>
      <div className="bg-card border border-border rounded-lg p-5">
        <p className="text-sm leading-relaxed">
          Insurers and healthcare providers must accurately determine a patient&apos;s financial risk
          and insurance quality tier  - yet the complexity of human health, lifestyle, and
          socioeconomic factors makes this inherently difficult to do manually or with simple heuristics.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-sm font-semibold text-accent mb-2">Insurance Underwriting</p>
          <p className="text-xs text-muted leading-relaxed">
            An underwriter inputs a new applicant&apos;s demographic and clinical data and the model
            automatically recommends a premium tier and risk rating  - replacing slow manual underwriting.
          </p>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-sm font-semibold text-success mb-2">Hospital Risk Stratification</p>
          <p className="text-xs text-muted leading-relaxed">
            A hospital administrator runs patient profiles through the model to flag high-risk
            patients before they require expensive intervention  - enabling proactive care management.
          </p>
        </div>
      </div>
      <div className="bg-card border border-border rounded-lg p-4">
        <p className="text-sm font-semibold text-warning mb-2">Key Challenges</p>
        <p className="text-xs text-muted leading-relaxed">
          High-dimensional, heterogeneous data combining continuous health metrics (BMI, blood pressure, HbA1c),
          binary chronic condition indicators, categorical socioeconomic variables, and insurance-specific features.
          Financial features are right-skewed with heavy outliers. ~30% missing values in alcohol_freq.
        </p>
      </div>
    </div>,

    // 2: Dataset
    <div key="dataset" className="space-y-6">
      <h2 className="text-3xl font-bold">The Dataset</h2>
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-card border border-border rounded-lg p-5 text-center">
          <p className="text-3xl font-bold text-accent">
            <AnimatedCounter target={100000} decimals={0} />
          </p>
          <p className="text-sm text-muted mt-1">Patient Records</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-5 text-center">
          <p className="text-3xl font-bold text-success">
            <AnimatedCounter target={numericCount} decimals={0} />
          </p>
          <p className="text-sm text-muted mt-1">Numeric Features</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-5 text-center">
          <p className="text-3xl font-bold text-warning">
            <AnimatedCounter target={categoricalCount} decimals={0} />
          </p>
          <p className="text-sm text-muted mt-1">Categorical Features</p>
        </div>
      </div>
      <div className="bg-card border border-border rounded-lg p-4 text-sm space-y-2">
        <p className="font-medium">Feature categories:</p>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs text-muted">
          <span><span className="text-foreground">Demographics:</span> age, sex, income, education, region</span>
          <span><span className="text-foreground">Health metrics:</span> BMI, blood pressure, LDL, HbA1c</span>
          <span><span className="text-foreground">Chronic conditions:</span> hypertension, diabetes, COPD, cancer, etc.</span>
          <span><span className="text-foreground">Insurance:</span> plan type, network tier, deductible, premiums</span>
          <span><span className="text-foreground">Utilization:</span> claims count, avg claim amount, total paid</span>
          <span><span className="text-foreground">Procedures:</span> imaging, surgery, physio, lab counts</span>
        </div>
      </div>
      <div className="bg-card border border-danger/30 rounded-lg p-4 text-sm">
        <p className="font-medium text-danger">Missing data</p>
        <p className="text-xs text-muted mt-1">
          Only <span className="text-warning font-medium">alcohol_freq</span> has missing values: 30,083 / 100,000 rows (30.1%).
          Handled via most-frequent imputation.
        </p>
      </div>
    </div>,

    // 3: EDA - Distributions
    <div key="eda-dist" className="space-y-6">
      <h2 className="text-3xl font-bold">EDA: Distributions</h2>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-sm font-semibold text-danger mb-2">Right-skewed (heavy tails)</p>
          <div className="space-y-2 text-xs text-muted">
            <p>Cost-related variables show extreme right skew:</p>
            <div className="grid grid-cols-2 gap-2 mt-2">
              {[
                { f: "annual_medical_cost", med: "$2,083", max: "$65,725" },
                { f: "total_claims_paid", med: "$643", max: "$72,518" },
                { f: "avg_claim_amount", med: "$318", max: "$30,011" },
                { f: "income", med: "$36,200", max: "$1,061,800" },
              ].map((d) => (
                <div key={d.f} className="bg-background rounded p-2">
                  <p className="text-foreground text-[10px] font-medium">{d.f}</p>
                  <p className="text-[10px]">median {d.med} &middot; max {d.max}</p>
                </div>
              ))}
            </div>
            <p className="mt-2">
              A small portion of patients account for disproportionately large healthcare costs.
              Outliers kept  - they represent real high-cost patients we want to identify.
            </p>
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-sm font-semibold text-success mb-2">Near-normal distributions</p>
          <div className="space-y-2 text-xs text-muted">
            <p>Physiological variables are approximately normal:</p>
            <div className="grid grid-cols-2 gap-2 mt-2">
              {[
                { f: "bmi", mean: "27.0", std: "5.0" },
                { f: "systolic_bp", mean: "117.8", std: "15.4" },
                { f: "diastolic_bp", mean: "73.6", std: "8.9" },
                { f: "ldl", mean: "120.0", std: "30.3" },
              ].map((d) => (
                <div key={d.f} className="bg-background rounded p-2">
                  <p className="text-foreground text-[10px] font-medium">{d.f}</p>
                  <p className="text-[10px]">&mu;={d.mean} &sigma;={d.std}</p>
                </div>
              ))}
            </div>
            <p className="mt-2">
              Consistent with real-world healthcare data where physiological measurements
              cluster around population norms.
            </p>
          </div>
        </div>
      </div>
    </div>,

    // 4: EDA - Correlations & Risk Groups
    <div key="eda-corr" className="space-y-6">
      <h2 className="text-3xl font-bold">EDA: Correlations & Risk Groups</h2>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-sm font-semibold mb-3">Top Feature Correlations</p>
          <div className="space-y-1.5">
            {topCorrelations.map((p) => (
              <div key={`${p.a}-${p.b}`} className="flex items-center gap-2 text-xs">
                <span className="truncate flex-1">{p.a}</span>
                <span className="text-muted">&times;</span>
                <span className="truncate flex-1">{p.b}</span>
                <span
                  className={`font-mono font-bold w-14 text-right ${
                    p.val > 0 ? "text-danger" : "text-accent"
                  }`}
                >
                  {p.val.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-muted mt-2">
            Strong correlations among cost features indicate potential redundancy.
          </p>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-sm font-semibold mb-3">High-Risk vs Low-Risk Patients</p>
          <div className="space-y-3 text-xs">
            <div>
              <p className="text-foreground font-medium">Strong separators</p>
              <p className="text-muted">Annual medical cost, claims count, chronic conditions  - clear median shifts between groups</p>
            </div>
            <div>
              <p className="text-foreground font-medium">Moderate separators</p>
              <p className="text-muted">Age, BMI, blood pressure, HbA1c  - overlapping distributions but higher medians in high-risk</p>
            </div>
            <div>
              <p className="text-foreground font-medium">Weak separators</p>
              <p className="text-muted">Region, marital status  - demographics alone are poor risk predictors</p>
            </div>
            <div>
              <p className="text-foreground font-medium">Behavioral</p>
              <p className="text-muted">Smoking status shows stronger association with risk than demographics</p>
            </div>
          </div>
        </div>
      </div>
    </div>,

    // 5: Pipeline
    <div key="pipeline" className="space-y-6">
      <h2 className="text-3xl font-bold">ML Pipeline</h2>
      <div className="flex items-center justify-center gap-2 flex-wrap">
        {[
          "Raw Data (100K)",
          "Impute Missing",
          "Encode + Scale",
          "60/20/20 Split",
          "GridSearchCV (3-fold)",
          "Evaluate",
        ].map((step, i) => (
          <div key={step} className="flex items-center gap-2">
            <div className="bg-card border border-border rounded-lg px-3 py-2 text-center">
              <p className="text-[10px] text-muted">Step {i + 1}</p>
              <p className="text-xs font-medium mt-0.5">{step}</p>
            </div>
            {i < 5 && <span className="text-muted">&rarr;</span>}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4 mt-2">
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-sm font-semibold mb-2">Models</p>
          <div className="space-y-1.5 text-xs">
            <p><span className="text-accent font-medium">Gradient Boosting</span>  - 300 rounds, max_depth=3, lr=0.1</p>
            <p><span className="text-success font-medium">Random Forest</span>  - 400 trees, unlimited depth</p>
            <p><span className="text-warning font-medium">Logistic Reg L1</span>  - saga solver, C=10, max_iter=5000</p>
            <p><span className="text-danger font-medium">Logistic Reg L2</span>  - saga solver, C=10, max_iter=5000</p>
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-sm font-semibold mb-2">Preprocessing</p>
          <div className="space-y-1.5 text-xs text-muted">
            <p><span className="text-foreground">Numeric:</span> Median imputation &rarr; StandardScaler</p>
            <p><span className="text-foreground">Categorical:</span> Mode imputation &rarr; OneHotEncoder</p>
            <p><span className="text-foreground">Splitting:</span> Stratified 60/20/20 (preserves class proportions)</p>
            <p><span className="text-foreground">Tuning:</span> GridSearchCV, scoring=f1_macro, cv=3</p>
          </div>
        </div>
      </div>
    </div>,

    // 6: Classification Results
    <div key="cls-results" className="space-y-6">
      <h2 className="text-3xl font-bold">Classification Results</h2>
      <p className="text-sm text-muted">
        Predicting <span className="text-accent">insurance_quality</span> (low / medium / high)  - test set
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-xs text-muted uppercase">
              <th className="text-left py-2 pr-4">Model</th>
              <th className="text-right py-2 px-4">Accuracy</th>
              <th className="text-right py-2 px-4">F1 (Macro)</th>
              <th className="text-right py-2 px-4">MCC</th>
            </tr>
          </thead>
          <tbody>
            {orig.clsMetrics.map((m) => (
              <tr
                key={m.model}
                className={`border-b border-border/50 ${
                  m.model === orig.summary.best_model ? "bg-accent/5" : ""
                }`}
              >
                <td className="py-3 pr-4 font-medium">
                  {modelLabel(m.model)}
                  {m.model === orig.summary.best_model && (
                    <span className="ml-2 text-xs text-accent">Best</span>
                  )}
                </td>
                <td className="text-right py-3 px-4 font-mono">{pct(m.test_accuracy)}</td>
                <td className="text-right py-3 px-4 font-mono">{fmt(m.test_f1_macro)}</td>
                <td className="text-right py-3 px-4 font-mono">{fmt(m.test_mcc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted">
        GBR and RF achieve &gt;99.9% accuracy. LogReg plateaus at ~88% due to linear decision boundaries.
        The near-perfect tree-based results raised a red flag  - investigated in the ablation study.
      </p>
    </div>,

    // 7: Confusion Matrices
    <div key="confusion" className="space-y-4">
      <h2 className="text-3xl font-bold">Confusion Matrices</h2>
      <p className="text-sm text-muted">Test set  - click cells for details</p>
      <div className="grid grid-cols-2 gap-3">
        {orig.confusions.slice(0, 4).map((cm) => (
          <ConfusionMatrix
            key={cm.model}
            model={cm.model}
            labels={cm.test.labels}
            matrix={cm.test.matrix}
          />
        ))}
      </div>
    </div>,

    // 8: Regression
    <div key="regression" className="space-y-6">
      <h2 className="text-3xl font-bold">Regression Results</h2>
      <p className="text-sm text-muted">
        Predicting <span className="text-accent">annual_medical_cost</span>
      </p>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="bg-card border border-border rounded-lg p-5 text-center">
          <p className="text-2xl font-bold text-success">
            <AnimatedCounter target={regMetrics[0].test_r2} decimals={4} />
          </p>
          <p className="text-sm text-muted mt-1">Best R²</p>
          <p className="text-xs text-muted">{modelLabel(regMetrics[0].model)}</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-5 text-center">
          <p className="text-2xl font-bold text-warning">
            <AnimatedCounter target={regMetrics[0].test_mae} decimals={1} prefix="$" />
          </p>
          <p className="text-sm text-muted mt-1">MAE</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-5 text-center">
          <p className="text-2xl font-bold text-danger">
            <AnimatedCounter target={regMetrics[0].test_rmse} decimals={1} prefix="$" />
          </p>
          <p className="text-sm text-muted mt-1">RMSE</p>
        </div>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-xs text-muted uppercase">
            <th className="text-left py-2 pr-4">Model</th>
            <th className="text-right py-2 px-4">MAE</th>
            <th className="text-right py-2 px-4">RMSE</th>
            <th className="text-right py-2 px-4">R²</th>
          </tr>
        </thead>
        <tbody>
          {regMetrics.map((m) => (
            <tr key={m.model} className="border-b border-border/50">
              <td className="py-3 pr-4 font-medium">{modelLabel(m.model)}</td>
              <td className="text-right py-3 px-4 font-mono">{fmt(m.test_mae, 2)}</td>
              <td className="text-right py-3 px-4 font-mono">{fmt(m.test_rmse, 2)}</td>
              <td className="text-right py-3 px-4 font-mono">{fmt(m.test_r2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-muted">
        Ridge and Linear achieve nearly identical R²=0.966. The model explains 96.6% of variance
        in medical costs, with average prediction error of ~$320.
      </p>
    </div>,

    // 9: Ablation
    <div key="ablation" className="space-y-6">
      <h2 className="text-3xl font-bold">Ablation Study: Data Leakage</h2>
      <p className="text-sm text-muted">
        What happens when we remove <span className="text-warning">is_high_risk</span> and{" "}
        <span className="text-warning">risk_score</span>?
      </p>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-card border border-success/30 rounded-lg p-4">
          <p className="text-xs text-success uppercase tracking-wider font-medium mb-2">With leaky features</p>
          {orig.clsMetrics.map((m) => (
            <div key={m.model} className="flex justify-between py-1.5 text-sm border-b border-border/30">
              <span>{modelLabel(m.model)}</span>
              <span className="font-mono text-success">{pct(m.test_accuracy)}</span>
            </div>
          ))}
        </div>
        <div className="bg-card border border-danger/30 rounded-lg p-4">
          <p className="text-xs text-danger uppercase tracking-wider font-medium mb-2">Without leaky features</p>
          {ablation.clsMetrics.map((m) => (
            <div key={m.model} className="flex justify-between py-1.5 text-sm border-b border-border/30">
              <span>{modelLabel(m.model)}</span>
              <span className="font-mono text-danger">{pct(m.test_accuracy)}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="bg-card border border-warning/30 rounded-lg p-4 text-sm">
        <p className="font-semibold text-warning">99.9% &rarr; ~40% accuracy</p>
        <p className="text-xs text-muted mt-1">
          risk_score and is_high_risk were encoding the target variable. The models weren&apos;t learning
          health patterns  - they were memorizing a direct mapping. This is a textbook case of
          data leakage, and the ablation proves it.
        </p>
      </div>
    </div>,

    // 10: High Risk experiment
    <div key="high-risk" className="space-y-6">
      <h2 className="text-3xl font-bold">Binary Risk Classification</h2>
      <p className="text-sm text-muted">
        Predicting <span className="text-accent">is_high_risk</span> (binary: 0/1)  - separate experiment
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-xs text-muted uppercase">
              <th className="text-left py-2 pr-4">Model</th>
              <th className="text-right py-2 px-4">Accuracy</th>
              <th className="text-right py-2 px-4">F1 (Macro)</th>
              <th className="text-right py-2 px-4">MCC</th>
              <th className="text-right py-2 px-4">ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
            {highRisk.clsMetrics.map((m) => {
              const roc = (m as unknown as Record<string, unknown>).test_roc_auc;
              return (
                <tr key={m.model} className={`border-b border-border/50 ${
                  m.model === highRisk.summary.best_model ? "bg-accent/5" : ""
                }`}>
                  <td className="py-3 pr-4 font-medium">
                    {modelLabel(m.model)}
                    {m.model === highRisk.summary.best_model && (
                      <span className="ml-2 text-xs text-accent">Best</span>
                    )}
                  </td>
                  <td className="text-right py-3 px-4 font-mono">{pct(m.test_accuracy)}</td>
                  <td className="text-right py-3 px-4 font-mono">{fmt(m.test_f1_macro)}</td>
                  <td className="text-right py-3 px-4 font-mono">{fmt(m.test_mcc)}</td>
                  <td className="text-right py-3 px-4 font-mono">{typeof roc === "number" ? fmt(roc) : " -"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted">
        GBR achieves perfect classification (MCC=1.0). All models perform well on this target
        because risk_score  - still in the feature set  - directly encodes risk. LogReg
        achieves 97%+ even with linear boundaries, confirming the signal is strong and nearly linear.
      </p>
    </div>,

    // 11: Takeaways
    <div key="takeaways" className="space-y-6">
      <h2 className="text-3xl font-bold">Key Takeaways</h2>
      <div className="space-y-3">
        {[
          {
            title: "Data Leakage Detection",
            desc: "Our ablation study revealed that risk_score and is_high_risk encode the target, inflating accuracy to 99.9%. Removing them crashes performance to ~40%, proving the leakage.",
            color: "text-danger",
          },
          {
            title: "Model Comparison",
            desc: "Tree-based models (GBR, RF) consistently outperform logistic regression. They handle non-linear interactions, heterogeneous features, and outliers naturally.",
            color: "text-accent",
          },
          {
            title: "Regression Strength",
            desc: "Ridge regression achieves R²=0.966 for annual medical cost prediction with MAE ~$320. Simple linear models work well when the relationship is genuinely linear.",
            color: "text-success",
          },
          {
            title: "Feature Engineering Matters",
            desc: "The derived insurance_quality target isn't predictable from health/demographic features alone. For real deployment, features must be available before the risk determination  - not derived from it.",
            color: "text-warning",
          },
        ].map((item) => (
          <div
            key={item.title}
            className="bg-card border border-border rounded-lg p-4 flex gap-4 items-start"
          >
            <div className={`text-lg ${item.color} mt-0.5`}>&#9654;</div>
            <div>
              <p className={`font-semibold ${item.color}`}>{item.title}</p>
              <p className="text-sm text-muted mt-0.5">{item.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>,

    // 12: Thank You
    <div key="thanks" className="text-center space-y-6">
      <h1 className="text-5xl font-bold">Thank You</h1>
      <p className="text-lg text-muted">Questions?</p>
      <div className="flex justify-center gap-6 mt-8">
        <div className="bg-card border border-border rounded-lg px-6 py-4 text-left">
          <p className="font-medium text-foreground text-sm">Explore the Dashboard</p>
          <p className="text-xs text-muted mt-1">All results are interactive in the sidebar</p>
        </div>
        <div className="bg-card border border-border rounded-lg px-6 py-4 text-left">
          <p className="font-medium text-foreground text-sm">3 Experiments</p>
          <p className="text-xs text-muted mt-1">Original, High Risk, Ablation  - toggle on any page</p>
        </div>
      </div>
    </div>,
  ];

  const totalSlides = slides.length;

  const goNext = useCallback(
    () => setSlide((s) => Math.min(s + 1, totalSlides - 1)),
    [totalSlides]
  );
  const goPrev = useCallback(() => setSlide((s) => Math.max(s - 1, 0)), []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " ") {
        e.preventDefault();
        goNext();
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        goPrev();
      }
      if (e.key === "n" || e.key === "N") {
        setShowNotes((v) => !v);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [goNext, goPrev]);

  return (
    <div className="fixed inset-0 ml-[220px] bg-background z-40 flex flex-col">
      {/* Main slide area */}
      <div className={`flex-1 overflow-hidden ${showNotes ? "h-[65%]" : "h-full"}`}>
        <div className="relative w-full h-full">
          {/* Left click area */}
          <button
            onClick={goPrev}
            className="absolute left-0 top-0 w-16 h-full z-50 cursor-w-resize opacity-0 hover:opacity-100 flex items-center justify-center"
            disabled={slide === 0}
          >
            <span className="text-muted text-2xl">&lsaquo;</span>
          </button>

          <div
            className="h-full transition-transform duration-500 ease-out flex"
            style={{
              width: `${totalSlides * 100}%`,
              transform: `translateX(-${(slide / totalSlides) * 100}%)`,
            }}
          >
            {slides.map((content, i) => (
              <div key={i} className="h-full" style={{ width: `${100 / totalSlides}%` }}>
                <SlideShell slideNum={i + 1} total={totalSlides}>
                  {content}
                </SlideShell>
              </div>
            ))}
          </div>

          {/* Right click area */}
          <button
            onClick={goNext}
            className="absolute right-0 top-0 w-16 h-full z-50 cursor-e-resize opacity-0 hover:opacity-100 flex items-center justify-center"
            disabled={slide === totalSlides - 1}
          >
            <span className="text-muted text-2xl">&rsaquo;</span>
          </button>
        </div>
      </div>

      {/* Speaker notes panel */}
      {showNotes && (
        <div className="h-[35%] border-t border-border bg-card overflow-y-auto px-6 py-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-semibold text-accent uppercase tracking-wider">
              Speaker Notes  - Slide {slide + 1}
            </p>
            <button
              onClick={() => setShowNotes(false)}
              className="text-xs text-muted hover:text-foreground"
            >
              Close (N)
            </button>
          </div>
          <div className="text-sm text-muted whitespace-pre-wrap leading-relaxed">
            {speakerNotes[slide] ?? "No notes for this slide."}
          </div>
        </div>
      )}

      {/* Progress dots */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-1.5 z-50">
        {slides.map((_, i) => (
          <button
            key={i}
            onClick={() => setSlide(i)}
            className={`w-2 h-2 rounded-full transition-all ${
              i === slide ? "bg-accent w-6" : "bg-border hover:bg-muted"
            }`}
          />
        ))}
      </div>
    </div>
  );
}

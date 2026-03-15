"use client";

import { useState, useEffect, useCallback } from "react";
import type { Experiment, RegressionMetric, FeatureStat } from "@/lib/types";
import { modelLabel, pct, fmt } from "@/lib/utils";
import AnimatedCounter from "./AnimatedCounter";

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
  `DATASET:
- Synthetic medical insurance dataset from Kaggle - 100,000 patient records, 54 columns
- Includes demographics, health metrics, chronic conditions, insurance info, claims, procedures
- Target variables: insurance_quality (derived, 3-class) and annual_medical_cost (continuous)
- This is supervised learning - labels (targets) are provided in the dataset

OUTLIERS & FEATURES:
- Cost features (income, medical cost, claims) are right-skewed with heavy outliers
- We kept outliers - they're real high-cost patients we want to identify
- StandardScaler reduces their leverage for LogReg; tree models handle them naturally
- Weak features (region, sex) kept because they may help in combination

MISSING VALUES:
- Only alcohol_freq has ~30% missing
- Imputed with most frequent value inside sklearn Pipeline (prevents data leakage)

CLASS IMBALANCE:
- 36.8% high-risk vs 63.2% low-risk
- F1-macro and MCC penalize majority-class guessing

WHY LOGREG OVER SIMPLER MODELS?
- Linear Regression can't classify - target is categorical, not continuous
- Single Decision Tree overfits without pruning
- LogReg gives interpretable coefficients + calibrated probabilities - important for healthcare
- RF and GBR added as more expressive baselines`,

  // 1: Pipeline
  `60/20/20 SPLIT:
- 60% training: model learns patterns from this data
- 20% validation: used to pick best hyperparameters (model never trains on this)
- 20% test: touched ONCE at the end - the "final exam"
- Stratified: each split has same class proportions

HYPERPARAMETERS EXPLAINED:
- C (LogReg): inverse regularization strength. Higher C = less penalty = more complex model
- alpha (Ridge): regularization strength. Lower alpha = less penalty
- n_estimators: how many trees in the ensemble (300 for GBR, 400 for RF)
- max_depth: how deep each tree can grow (deeper = more complex = risk of overfitting)
- learning_rate (GBR): how much each new tree corrects previous errors

GridSearchCV:
- Tries every combination of hyperparameters
- For each combo: splits training data 3 ways, trains on 2, validates on 1, rotates
- Picks the combo with best average F1-macro score
- Then retrains best combo on full training set

CHALLENGES & LIMITATIONS:
- High feature dimensionality (54 features) with correlated predictors (cost features are redundant)
- Insufficient regularization strength could let the model overfit to noise
- Imputed missing data (alcohol_freq at 30%) introduces noise
- Tuned regularization parameter C via cross-validation to find the right balance
- L1 regularization zeros out weakest features automatically (built-in feature selection)

GRADIENT DESCENT?
- LogReg uses SAGA - a stochastic gradient method (updates weights one sample at a time)
- GBR/RF don't use gradient descent at all - they greedily split features to build trees
- GBR is "gradient" because each new tree fits the gradient of the loss (residual errors), not SGD
- Ridge has a closed-form solution (linear algebra), no gradient descent needed
- We didn't choose between batch/mini-batch/SGD - sklearn abstracts that away`,

  // 2: Results
  `HYPERPARAMETER TUNING:
- GridSearchCV with 3-fold CV on training set
- LogReg: C=[0.1, 1.0, 10.0] via SAGA solver, tested balanced class weights
- GBR: 300 rounds, learning_rate=[0.05, 0.1], max_depth=[2, 3]
- RF: 400 trees, max_depth=[None, 12]
- Ridge: alpha=0.01 (best via 3-fold CV)

EPOCHS?
- These aren't neural networks - no epochs
- LogReg converges via SAGA optimizer (max_iter=5000, stops when loss plateaus)
- GBR: 300 boosting rounds; RF: 400 independent trees

OVERFITTING:
- 60/20/20 stratified split + 3-fold CV
- Caught when train accuracy >> val accuracy
- L1/L2 regularization directly prevents it

WHY THE MODEL PREDICTS WHAT IT DOES:
- Smoker coefficient pushes toward high-risk (73% vs 31% high-risk rate)
- Age: high-risk avg 59 vs low-risk avg 41
- Unexpected predictions = rare combos (young + multiple chronic conditions)

MOBILE DEPLOYMENT:
- L1 zeros out weak features automatically
- Drop sex, region, employment_status first
- 54 features down to ~15-20 with minimal loss`,

  // 2: Ablation + Key Findings
  `DATA LEAKAGE (THE KEY FINDING):
- risk_score and is_high_risk were encoding the target
- Removing them: 99.9% crashes to ~40% (near random for 3 classes)
- The model was memorizing, not learning

FAILURE CASE:
- Young non-smoker with kidney disease + cancer history
- Genuinely high-risk but looks low-risk on dominant features
- Shows model relies on population-level patterns, misses rare combos
- Exposes logistic regression's linearity limitation

BIAS:
- Income, region, education correlate with race/ethnicity
- Could flag low-income patients as high-risk due to socioeconomic disadvantage, not health
- Underserved populations with fewer claims (access barriers) get mislabeled as low-risk
- Fix: run F1 and false positive rate broken down by sex, region, income

VALIDATION:
- 60/20/20 split with stratification
- Val set for model selection, test set only touched at the end
- 3-fold CV ensures hyperparameters generalize`,
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

  void features; void corrFeatures; void corrMatrix;

  const slides = [
    // 0: Title + Problem + Dataset
    <div key="title" className="space-y-5">
      <div className="text-center">
        <h1 className="text-4xl font-bold tracking-tight">
          Medical Insurance <span className="text-accent">Risk Prediction</span>
        </h1>
        <p className="text-sm text-muted mt-2">ECS 171 - Machine Learning - UC Davis</p>
      </div>
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-card border border-border rounded-lg p-3 text-center">
          <p className="text-2xl font-bold text-accent"><AnimatedCounter target={100000} decimals={0} /></p>
          <p className="text-xs text-muted">Patients</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-3 text-center">
          <p className="text-2xl font-bold text-success"><AnimatedCounter target={54} decimals={0} /></p>
          <p className="text-xs text-muted">Features</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-3 text-center">
          <p className="text-2xl font-bold text-warning">4</p>
          <p className="text-xs text-muted">Models</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-3 text-center">
          <p className="text-2xl font-bold text-danger">3</p>
          <p className="text-xs text-muted">Experiments</p>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-lg p-3">
          <p className="text-xs font-semibold mb-1.5">Pipeline</p>
          <div className="flex items-center gap-1 flex-wrap text-[10px] text-muted">
            {["Impute", "Encode + Scale", "60/20/20 Split", "GridSearchCV (3-fold)", "Evaluate"].map((s, i) => (
              <span key={s} className="flex items-center gap-1">
                <span className="bg-background rounded px-1.5 py-0.5">{s}</span>
                {i < 4 && <span>&rarr;</span>}
              </span>
            ))}
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg p-3">
          <p className="text-xs font-semibold mb-1.5">Models</p>
          <div className="grid grid-cols-2 gap-1 text-[10px]">
            <span><span className="text-accent">GBR</span> - 300 rounds, depth=3</span>
            <span><span className="text-success">RF</span> - 400 trees</span>
            <span><span className="text-warning">LogReg L1</span> - SAGA, C=10</span>
            <span><span className="text-danger">LogReg L2</span> - SAGA, C=10</span>
          </div>
        </div>
      </div>
      <p className="text-[10px] text-muted/50 text-center">
        Arrow keys to navigate &middot; Press <kbd className="px-1 py-0.5 rounded bg-card border border-border text-muted">N</kbd> for speaker notes
      </p>
    </div>,

    // 1: Pipeline
    <div key="pipeline" className="space-y-5">
      <h2 className="text-3xl font-bold">ML Pipeline</h2>
      <div className="flex items-center justify-center gap-2 flex-wrap">
        {[
          { step: "Raw Data", detail: "100K rows, 54 cols" },
          { step: "Impute Missing", detail: "median / most frequent" },
          { step: "Encode + Scale", detail: "OneHot + StandardScaler" },
          { step: "Split", detail: "60 / 20 / 20" },
          { step: "GridSearchCV", detail: "3-fold CV" },
          { step: "Evaluate", detail: "test set" },
        ].map((s, i) => (
          <div key={s.step} className="flex items-center gap-2">
            <div className="bg-card border border-border rounded-lg px-3 py-2 text-center min-w-[100px]">
              <p className="text-xs font-medium">{s.step}</p>
              <p className="text-[10px] text-muted">{s.detail}</p>
            </div>
            {i < 5 && <span className="text-muted">&rarr;</span>}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-card border border-accent/30 rounded-lg p-3 text-center">
          <p className="text-2xl font-bold text-accent">60%</p>
          <p className="text-xs text-muted">Training</p>
          <p className="text-[10px] text-muted">Model learns here</p>
        </div>
        <div className="bg-card border border-warning/30 rounded-lg p-3 text-center">
          <p className="text-2xl font-bold text-warning">20%</p>
          <p className="text-xs text-muted">Validation</p>
          <p className="text-[10px] text-muted">Tune hyperparams</p>
        </div>
        <div className="bg-card border border-success/30 rounded-lg p-3 text-center">
          <p className="text-2xl font-bold text-success">20%</p>
          <p className="text-xs text-muted">Test</p>
          <p className="text-[10px] text-muted">Final evaluation only</p>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-lg p-3">
          <p className="text-xs font-semibold mb-2">Classification Models</p>
          <div className="space-y-1 text-[11px]">
            <p><span className="text-accent font-medium">Gradient Boosting</span> - n_estimators=300, lr=0.05, depth=3</p>
            <p><span className="text-success font-medium">Random Forest</span> - n_estimators=400, max_depth=None</p>
            <p><span className="text-warning font-medium">LogReg L1</span> - C=10, SAGA solver, max_iter=5000</p>
            <p><span className="text-danger font-medium">LogReg L2</span> - C=10, SAGA solver, max_iter=5000</p>
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg p-3">
          <p className="text-xs font-semibold mb-2">Hyperparameter Tuning</p>
          <div className="space-y-1 text-[11px] text-muted">
            <p><span className="text-foreground">LogReg C:</span> [0.1, 1.0, 10.0] - inverse regularization</p>
            <p><span className="text-foreground">GBR lr:</span> [0.05, 0.1] - learning rate per round</p>
            <p><span className="text-foreground">GBR depth:</span> [2, 3] - how deep each tree grows</p>
            <p><span className="text-foreground">RF depth:</span> [None, 12] - unlimited vs capped</p>
            <p><span className="text-foreground">Ridge alpha:</span> best=0.01 via 3-fold CV</p>
          </div>
        </div>
      </div>
    </div>,

    // 2: Results (Classification + Regression)
    <div key="results" className="space-y-4">
      <h2 className="text-3xl font-bold">Results</h2>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-muted mb-2">
            Classification: <span className="text-accent">insurance_quality</span> (low/med/high)
          </p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border text-[10px] text-muted uppercase">
                <th className="text-left py-1.5">Model</th>
                <th className="text-right py-1.5">Accuracy</th>
                <th className="text-right py-1.5">F1</th>
                <th className="text-right py-1.5">MCC</th>
              </tr>
            </thead>
            <tbody>
              {orig.clsMetrics.map((m) => (
                <tr key={m.model} className="border-b border-border/30">
                  <td className="py-1.5 font-medium">{modelLabel(m.model)}</td>
                  <td className="text-right py-1.5 font-mono">{pct(m.test_accuracy)}</td>
                  <td className="text-right py-1.5 font-mono">{fmt(m.test_f1_macro)}</td>
                  <td className="text-right py-1.5 font-mono">{fmt(m.test_mcc)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <p className="text-xs text-muted mb-2">
            Regression: <span className="text-accent">annual_medical_cost</span>
          </p>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border text-[10px] text-muted uppercase">
                <th className="text-left py-1.5">Model</th>
                <th className="text-right py-1.5">MAE</th>
                <th className="text-right py-1.5">RMSE</th>
                <th className="text-right py-1.5">R&sup2;</th>
              </tr>
            </thead>
            <tbody>
              {regMetrics.map((m) => (
                <tr key={m.model} className="border-b border-border/30">
                  <td className="py-1.5 font-medium">{modelLabel(m.model)}</td>
                  <td className="text-right py-1.5 font-mono">${fmt(m.test_mae, 0)}</td>
                  <td className="text-right py-1.5 font-mono">${fmt(m.test_rmse, 0)}</td>
                  <td className="text-right py-1.5 font-mono">{fmt(m.test_r2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-card border border-success/30 rounded-lg p-3 text-center">
          <p className="text-xl font-bold text-success font-mono">{pct(orig.clsMetrics[0]?.test_accuracy ?? 0)}</p>
          <p className="text-[10px] text-muted">Best Classification Acc</p>
        </div>
        <div className="bg-card border border-accent/30 rounded-lg p-3 text-center">
          <p className="text-xl font-bold text-accent font-mono">{fmt(regMetrics[0]?.test_r2 ?? 0)}</p>
          <p className="text-[10px] text-muted">Best R&sup2;</p>
        </div>
        <div className="bg-card border border-warning/30 rounded-lg p-3 text-center">
          <p className="text-xl font-bold text-warning font-mono">${fmt(regMetrics[0]?.test_mae ?? 0, 0)}</p>
          <p className="text-[10px] text-muted">Best MAE</p>
        </div>
      </div>
      <p className="text-xs text-muted">
        GBR/RF achieve &gt;99.9% classification accuracy - suspiciously perfect.
        Ridge regression explains 96.6% of cost variance with ~$320 average error.
      </p>
    </div>,

    // 2: Ablation + Key Findings
    <div key="ablation" className="space-y-4">
      <h2 className="text-3xl font-bold">Key Finding: Data Leakage</h2>
      <p className="text-sm text-muted">
        Removing <span className="text-warning font-medium">risk_score</span> and <span className="text-warning font-medium">is_high_risk</span> from features:
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
      <div className="bg-card border border-warning/30 rounded-lg p-4">
        <p className="font-semibold text-warning">99.9% &rarr; ~40% accuracy</p>
        <p className="text-xs text-muted mt-1">
          The models weren&apos;t learning health patterns - they were memorizing a direct mapping
          from risk_score to insurance_quality. This is textbook data leakage, and the ablation proves it.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-card border border-border rounded-lg p-3">
          <p className="text-xs font-semibold text-accent">What this means</p>
          <p className="text-[10px] text-muted mt-1">
            insurance_quality is NOT predictable from health/demographic features alone.
            For real deployment, features must exist before the risk determination - not be derived from it.
          </p>
        </div>
        <div className="bg-card border border-border rounded-lg p-3">
          <p className="text-xs font-semibold text-success">What works</p>
          <p className="text-[10px] text-muted mt-1">
            Ridge regression (R&sup2;=0.966) legitimately predicts annual medical cost.
            Binary risk classification achieves 97%+ with logistic regression alone.
          </p>
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
              Q&A Notes
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

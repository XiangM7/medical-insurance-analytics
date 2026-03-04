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
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-4xl">{children}</div>
      </div>
      <div className="flex items-center justify-between px-8 pb-4 text-xs text-muted">
        <span>ECS 171 &mdash; Medical Insurance ML Project</span>
        <span>
          {slideNum} / {total}
        </span>
      </div>
    </div>
  );
}

export default function PresentationClient({
  experiments,
  regMetrics,
  features,
  corrFeatures,
  corrMatrix,
}: Props) {
  const [slide, setSlide] = useState(0);

  const orig = experiments.find((e) => e.id === "original")!;
  const highRisk = experiments.find((e) => e.id === "high_risk")!;
  const ablation = experiments.find((e) => e.id === "removed_stuff")!;

  const numericCount = features.filter((f) => f.dtype !== "str").length;
  const categoricalCount = features.filter((f) => f.dtype === "str").length;

  // Mini heatmap renderer (top correlations)
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
        <span className="text-accent">ML Classification</span>
      </h1>
      <p className="text-lg text-muted">
        Predicting Insurance Quality & Risk from Patient Data
      </p>
      <div className="flex justify-center gap-6 mt-8 text-sm text-muted">
        <span>ECS 171 &mdash; Machine Learning</span>
        <span>UC Davis</span>
      </div>
      <p className="text-xs text-muted/50 mt-12">
        Press arrow keys or click edges to navigate
      </p>
    </div>,

    // 1: Dataset
    <div key="dataset" className="space-y-6">
      <h2 className="text-3xl font-bold">The Dataset</h2>
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-card border border-border rounded-lg p-5 text-center">
          <p className="text-3xl font-bold text-accent">
            <AnimatedCounter target={100000} decimals={0} />
          </p>
          <p className="text-sm text-muted mt-1">Samples</p>
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
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-muted text-xs">
          <span>Demographics (age, sex, income, education)</span>
          <span>Health (BMI, blood pressure, LDL, HbA1c)</span>
          <span>Conditions (hypertension, diabetes, COPD...)</span>
          <span>Insurance (plan type, deductible, premiums)</span>
          <span>Claims (count, avg amount, total paid)</span>
          <span>Procedures (imaging, surgery, lab counts)</span>
        </div>
      </div>
    </div>,

    // 2: EDA highlights
    <div key="eda" className="space-y-6">
      <h2 className="text-3xl font-bold">Exploratory Analysis</h2>
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
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-sm font-semibold mb-3">Data Quality</p>
          <div className="space-y-3 text-sm">
            <div>
              <p className="text-muted text-xs">Missing Values</p>
              <p className="font-medium">
                Only <span className="text-warning">alcohol_freq</span> has missing data
              </p>
              <p className="text-xs text-muted">30,083 / 100,000 rows (30%)</p>
            </div>
            <div>
              <p className="text-muted text-xs">Target Variable</p>
              <p className="font-medium">
                <span className="text-accent">insurance_quality</span> (low / medium / high)
              </p>
              <p className="text-xs text-muted">
                Derived from plan type, network tier, deductible, premium
              </p>
            </div>
            <div>
              <p className="text-muted text-xs">Class Balance</p>
              <p className="font-medium">
                3 balanced classes (~33% each via quantile split)
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>,

    // 3: Pipeline
    <div key="pipeline" className="space-y-6">
      <h2 className="text-3xl font-bold">ML Pipeline</h2>
      <div className="flex items-center justify-center gap-3">
        {[
          "Raw Data",
          "Impute + Encode",
          "Train/Val/Test Split",
          "GridSearchCV",
          "Evaluate",
        ].map((step, i) => (
          <div key={step} className="flex items-center gap-3">
            <div className="bg-card border border-border rounded-lg px-4 py-3 text-center">
              <p className="text-xs text-muted">Step {i + 1}</p>
              <p className="text-sm font-medium mt-0.5">{step}</p>
            </div>
            {i < 4 && <span className="text-muted text-lg">&rarr;</span>}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-sm font-semibold mb-2">Classification Models</p>
          <div className="space-y-1 text-xs">
            <p className="text-accent">Gradient Boosting (300 estimators)</p>
            <p className="text-success">Random Forest (400 estimators)</p>
            <p className="text-warning">Logistic Regression L1 (saga, C=10)</p>
            <p className="text-danger">Logistic Regression L2 (saga, C=10)</p>
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-sm font-semibold mb-2">Preprocessing</p>
          <div className="space-y-1 text-xs text-muted">
            <p>Numeric: Median imputation + StandardScaler</p>
            <p>Categorical: Mode imputation + OneHotEncoder</p>
            <p>Split: 60% train / 20% val / 20% test</p>
            <p>CV: 3-fold cross-validation</p>
          </div>
        </div>
      </div>
    </div>,

    // 4: Classification Results
    <div key="cls-results" className="space-y-6">
      <h2 className="text-3xl font-bold">Classification Results</h2>
      <p className="text-sm text-muted">
        Original experiment &mdash; predicting <span className="text-accent">insurance_quality</span> (3 classes)
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
        GBR and RF achieve near-perfect accuracy (&gt;99.9%) due to highly predictive features
        like <span className="text-warning">risk_score</span> and <span className="text-warning">is_high_risk</span>.
      </p>
    </div>,

    // 5: Confusion Matrices
    <div key="confusion" className="space-y-4">
      <h2 className="text-3xl font-bold">Confusion Matrices</h2>
      <p className="text-sm text-muted">Test set &mdash; Original experiment (click cells for details)</p>
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

    // 6: Regression Results
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
    </div>,

    // 7: Ablation Study
    <div key="ablation" className="space-y-6">
      <h2 className="text-3xl font-bold">Ablation Study</h2>
      <p className="text-sm text-muted">
        What happens when we remove <span className="text-warning">is_high_risk</span> and{" "}
        <span className="text-warning">risk_score</span>?
      </p>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-xs text-muted uppercase tracking-wider mb-2">With Features</p>
          {orig.clsMetrics.map((m) => (
            <div key={m.model} className="flex justify-between py-1.5 text-sm border-b border-border/30">
              <span>{modelLabel(m.model)}</span>
              <span className="font-mono text-success">{pct(m.test_accuracy)}</span>
            </div>
          ))}
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <p className="text-xs text-muted uppercase tracking-wider mb-2">Without Features</p>
          {ablation.clsMetrics.map((m) => (
            <div key={m.model} className="flex justify-between py-1.5 text-sm border-b border-border/30">
              <span>{modelLabel(m.model)}</span>
              <span className="font-mono text-danger">{pct(m.test_accuracy)}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="bg-card border border-border rounded-lg p-4 text-sm">
        <p className="font-medium text-danger">
          Accuracy drops from ~99.9% to ~40% (near random for 3 classes)
        </p>
        <p className="text-muted text-xs mt-1">
          This confirms risk_score and is_high_risk were driving nearly all predictive power
          &mdash; a clear case of data leakage in the original experiment.
        </p>
      </div>
    </div>,

    // 8: Key Takeaways
    <div key="takeaways" className="space-y-6">
      <h2 className="text-3xl font-bold">Key Takeaways</h2>
      <div className="space-y-4">
        {[
          {
            title: "Data Leakage Matters",
            desc: "risk_score and is_high_risk encode the target, inflating accuracy to 99.9%. Removing them drops performance to ~40%.",
            color: "text-danger",
          },
          {
            title: "Model Selection",
            desc: "GBR and RF outperform logistic regression across all experiments. Tree-based models capture non-linear feature interactions.",
            color: "text-accent",
          },
          {
            title: "Regression Performance",
            desc: "Ridge and Linear regression both achieve R² = 0.966 for predicting annual medical costs, with MAE ~$320.",
            color: "text-success",
          },
          {
            title: "Feature Engineering",
            desc: "The insurance_quality target is derived from plan/network/cost features. Without leakage features, these alone aren't enough for classification.",
            color: "text-warning",
          },
        ].map((item) => (
          <div
            key={item.title}
            className="bg-card border border-border rounded-lg p-4 flex gap-4 items-start"
          >
            <div className={`text-2xl ${item.color} mt-0.5`}>&#9654;</div>
            <div>
              <p className={`font-semibold ${item.color}`}>{item.title}</p>
              <p className="text-sm text-muted mt-0.5">{item.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>,

    // 9: Thank You
    <div key="thanks" className="text-center space-y-6">
      <h1 className="text-5xl font-bold">Thank You</h1>
      <p className="text-lg text-muted">Questions?</p>
      <div className="flex justify-center gap-8 mt-8 text-sm text-muted">
        <div className="bg-card border border-border rounded-lg px-6 py-4">
          <p className="font-medium text-foreground">Explore the Dashboard</p>
          <p className="text-xs mt-1">All results are interactive in the sidebar pages</p>
        </div>
      </div>
    </div>,
  ];

  const totalSlides = slides.length;

  const goNext = useCallback(() => setSlide((s) => Math.min(s + 1, totalSlides - 1)), [totalSlides]);
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
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [goNext, goPrev]);

  return (
    <div className="fixed inset-0 ml-[220px] bg-background z-40 flex">
      {/* Left click area */}
      <button
        onClick={goPrev}
        className="absolute left-0 top-0 w-16 h-full z-50 cursor-w-resize opacity-0 hover:opacity-100 flex items-center justify-center"
        disabled={slide === 0}
      >
        <span className="text-muted text-2xl">&lsaquo;</span>
      </button>

      {/* Slide content */}
      <div className="flex-1 overflow-hidden">
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
      </div>

      {/* Right click area */}
      <button
        onClick={goNext}
        className="absolute right-0 top-0 w-16 h-full z-50 cursor-e-resize opacity-0 hover:opacity-100 flex items-center justify-center"
        disabled={slide === totalSlides - 1}
      >
        <span className="text-muted text-2xl">&rsaquo;</span>
      </button>

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

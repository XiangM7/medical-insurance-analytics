"use client";

import { useState } from "react";
import KpiCard from "./KpiCard";
import ExperimentSelector from "./ExperimentSelector";
import { modelLabel, pct, fmt } from "@/lib/utils";
import type { Experiment, ExperimentId, RegressionMetric } from "@/lib/types";

interface Props {
  experiments: Experiment[];
  regMetrics: RegressionMetric[];
  totalFeatures: number;
  totalSamples: number;
}

export default function DashboardClient({
  experiments,
  regMetrics,
  totalFeatures,
  totalSamples,
}: Props) {
  const [expId, setExpId] = useState<ExperimentId>("original");
  const exp = experiments.find((e) => e.id === expId) ?? experiments[0];
  const { clsMetrics, summary } = exp;

  const best = clsMetrics.find((m) => m.model === summary.best_model) ?? clsMetrics[0];
  const bestReg = regMetrics[0];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-bold">Dashboard Overview</h1>
          <p className="text-sm text-muted mt-1">{exp.description}</p>
        </div>
        <ExperimentSelector value={expId} onChange={setExpId} />
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard
          title="Best Model Accuracy"
          value={best.test_accuracy * 100}
          suffix="%"
          subtitle={modelLabel(best.model)}
        />
        <KpiCard
          title="Best MCC Score"
          value={summary.best_test_mcc}
          decimals={4}
          subtitle={modelLabel(summary.best_model)}
        />
        <KpiCard
          title="Best R² Score"
          value={bestReg.test_r2}
          decimals={4}
          subtitle={modelLabel(bestReg.model)}
        />
        <KpiCard
          title="Dataset Size"
          value={totalSamples}
          decimals={0}
          subtitle={`${totalFeatures} features`}
        />
      </div>

      {/* Classification Comparison Table */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-3">Classification Model Comparison (Test Set)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-xs text-muted uppercase">
                <th className="text-left py-2 pr-4">Model</th>
                <th className="text-right py-2 px-3">Accuracy</th>
                <th className="text-right py-2 px-3">F1 (Macro)</th>
                <th className="text-right py-2 px-3">MCC</th>
              </tr>
            </thead>
            <tbody>
              {clsMetrics.map((m) => (
                <tr
                  key={m.model}
                  className={`border-b border-border/50 hover:bg-card-hover transition-colors ${
                    m.model === summary.best_model ? "bg-accent/5" : ""
                  }`}
                >
                  <td className="py-2 pr-4 font-medium">
                    {modelLabel(m.model)}
                    {m.model === summary.best_model && (
                      <span className="ml-2 text-xs text-accent">Best</span>
                    )}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-xs">
                    {pct(m.test_accuracy)}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-xs">
                    {fmt(m.test_f1_macro)}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-xs">
                    {fmt(m.test_mcc)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Regression Summary */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-3">Regression Model Comparison (Test Set)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-xs text-muted uppercase">
                <th className="text-left py-2 pr-4">Model</th>
                <th className="text-right py-2 px-3">MAE</th>
                <th className="text-right py-2 px-3">RMSE</th>
                <th className="text-right py-2 px-3">R²</th>
              </tr>
            </thead>
            <tbody>
              {regMetrics.map((m) => (
                <tr
                  key={m.model}
                  className="border-b border-border/50 hover:bg-card-hover transition-colors"
                >
                  <td className="py-2 pr-4 font-medium">{modelLabel(m.model)}</td>
                  <td className="text-right py-2 px-3 font-mono text-xs">
                    {fmt(m.test_mae, 2)}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-xs">
                    {fmt(m.test_rmse, 2)}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-xs">
                    {fmt(m.test_r2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

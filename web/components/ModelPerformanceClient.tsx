"use client";

import { useState } from "react";
import type { Experiment, ExperimentId, RegressionMetric, Split } from "@/lib/types";
import MetricToggle from "./MetricToggle";
import ExperimentSelector from "./ExperimentSelector";
import ClassificationChart from "./ClassificationChart";
import RegressionChart from "./RegressionChart";

interface Props {
  experiments: Experiment[];
  regMetrics: RegressionMetric[];
}

export default function ModelPerformanceClient({ experiments, regMetrics }: Props) {
  const [split, setSplit] = useState<Split>("test");
  const [expId, setExpId] = useState<ExperimentId>("original");

  const exp = experiments.find((e) => e.id === expId) ?? experiments[0];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-bold">Model Performance</h1>
          <p className="text-sm text-muted mt-1">{exp.description}</p>
        </div>
        <div className="flex items-center gap-2">
          <ExperimentSelector value={expId} onChange={setExpId} />
          <MetricToggle value={split} onChange={setSplit} />
        </div>
      </div>

      <div className="bg-card border border-border rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-3">Classification Metrics</h2>
        <ClassificationChart metrics={exp.clsMetrics} split={split} />
      </div>

      <div className="bg-card border border-border rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-3">Regression Metrics</h2>
        <RegressionChart metrics={regMetrics} split={split} />
      </div>
    </div>
  );
}

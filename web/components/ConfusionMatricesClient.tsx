"use client";

import { useState } from "react";
import type { Experiment, ExperimentId, Split } from "@/lib/types";
import MetricToggle from "./MetricToggle";
import ExperimentSelector from "./ExperimentSelector";
import ConfusionMatrix from "./ConfusionMatrix";

interface Props {
  experiments: Experiment[];
}

export default function ConfusionMatricesClient({ experiments }: Props) {
  const [split, setSplit] = useState<Split>("test");
  const [expId, setExpId] = useState<ExperimentId>("original");

  const exp = experiments.find((e) => e.id === expId) ?? experiments[0];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-bold">Confusion Matrices</h1>
          <p className="text-sm text-muted mt-1">{exp.description}</p>
        </div>
        <div className="flex items-center gap-2">
          <ExperimentSelector value={expId} onChange={setExpId} />
          <MetricToggle value={split} onChange={setSplit} />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {exp.confusions.map((cm) => (
          <ConfusionMatrix
            key={`${expId}-${cm.model}`}
            model={cm.model}
            labels={cm[split].labels}
            matrix={cm[split].matrix}
          />
        ))}
      </div>
    </div>
  );
}

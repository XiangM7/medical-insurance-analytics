"use client";

import CorrelationHeatmap from "./CorrelationHeatmap";
import FeatureTable from "./FeatureTable";
import type { FeatureStat } from "@/lib/types";

interface Props {
  features: FeatureStat[];
  corrFeatures: string[];
  corrMatrix: number[][];
}

export default function EDAClient({ features, corrFeatures, corrMatrix }: Props) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">EDA Explorer</h1>
        <p className="text-sm text-muted mt-1">
          Correlation heatmap and feature statistics
        </p>
      </div>

      <div className="bg-card border border-border rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-3">
          Correlation Heatmap ({corrFeatures.length}&times;{corrFeatures.length})
        </h2>
        <CorrelationHeatmap features={corrFeatures} matrix={corrMatrix} />
      </div>

      <div className="bg-card border border-border rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-3">Feature Statistics</h2>
        <FeatureTable features={features} />
      </div>
    </div>
  );
}

"use client";

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { ClassificationMetric, Split } from "@/lib/types";
import { modelLabel } from "@/lib/utils";

const COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"];

interface Props {
  metrics: ClassificationMetric[];
  split: Split;
}

export default function ModelRadarChart({ metrics, split }: Props) {
  const subjects = ["Accuracy", "F1 (Macro)", "MCC"];

  const data = subjects.map((subject) => {
    const entry: Record<string, string | number> = { subject };
    metrics.forEach((m) => {
      const label = modelLabel(m.model);
      if (subject === "Accuracy") entry[label] = split === "val" ? m.val_accuracy : m.test_accuracy;
      else if (subject === "F1 (Macro)")
        entry[label] = split === "val" ? m.val_f1_macro : m.test_f1_macro;
      else entry[label] = split === "val" ? m.val_mcc : m.test_mcc;
    });
    return entry;
  });

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadarChart data={data} cx="50%" cy="50%" outerRadius="70%">
        <PolarGrid stroke="#262626" />
        <PolarAngleAxis dataKey="subject" tick={{ fill: "#737373", fontSize: 11 }} />
        <PolarRadiusAxis
          domain={[0, 1]}
          tick={{ fill: "#525252", fontSize: 10 }}
          axisLine={false}
        />
        {metrics.map((m, i) => (
          <Radar
            key={m.model}
            name={modelLabel(m.model)}
            dataKey={modelLabel(m.model)}
            stroke={COLORS[i]}
            fill={COLORS[i]}
            fillOpacity={0.15}
          />
        ))}
        <Legend wrapperStyle={{ fontSize: 11 }} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

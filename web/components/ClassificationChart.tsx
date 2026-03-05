"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { ClassificationMetric, Split } from "@/lib/types";
import { modelLabel } from "@/lib/utils";

interface Props {
  metrics: ClassificationMetric[];
  split: Split;
}

export default function ClassificationChart({ metrics, split }: Props) {
  const data = metrics.map((m) => ({
    name: modelLabel(m.model),
    Accuracy: split === "val" ? m.val_accuracy : m.test_accuracy,
    "F1 (Macro)": split === "val" ? m.val_f1_macro : m.test_f1_macro,
    MCC: split === "val" ? m.val_mcc : m.test_mcc,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
        <XAxis dataKey="name" tick={{ fill: "#737373", fontSize: 11 }} />
        <YAxis domain={[0, 1]} tick={{ fill: "#737373", fontSize: 11 }} />
        <Tooltip
          contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #262626", borderRadius: 6, fontSize: 12 }}
          itemStyle={{ color: "#e5e5e5" }}
          labelStyle={{ color: "#737373" }}
          cursor={{ fill: "rgba(255,255,255,0.05)" }}
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          formatter={(val: any) => typeof val === "number" ? val.toFixed(4) : String(val ?? "")}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="Accuracy" fill="#3b82f6" radius={[2, 2, 0, 0]} />
        <Bar dataKey="F1 (Macro)" fill="#22c55e" radius={[2, 2, 0, 0]} />
        <Bar dataKey="MCC" fill="#f59e0b" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

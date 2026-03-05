"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { RegressionMetric, Split } from "@/lib/types";
import { modelLabel } from "@/lib/utils";

interface Props {
  metrics: RegressionMetric[];
  split: Split;
}

export default function RegressionChart({ metrics, split }: Props) {
  const maeData = metrics.map((m) => ({
    name: modelLabel(m.model),
    MAE: split === "val" ? m.val_mae : m.test_mae,
  }));

  const rmseData = metrics.map((m) => ({
    name: modelLabel(m.model),
    RMSE: split === "val" ? m.val_rmse : m.test_rmse,
  }));

  const r2Data = metrics.map((m) => ({
    name: modelLabel(m.model),
    R2: split === "val" ? m.val_r2 : m.test_r2,
  }));

  const chartProps = {
    margin: { top: 5, right: 20, left: 0, bottom: 5 },
  };

  const tooltipStyle = {
    backgroundColor: "#1a1a1a",
    border: "1px solid #262626",
    borderRadius: 6,
    fontSize: 12,
  };
  const tooltipExtra = {
    itemStyle: { color: "#e5e5e5" },
    labelStyle: { color: "#737373" },
    cursor: { fill: "rgba(255,255,255,0.05)" },
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div>
        <p className="text-xs text-muted mb-2 font-medium">MAE (lower is better)</p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={maeData} {...chartProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
            <XAxis dataKey="name" tick={{ fill: "#737373", fontSize: 10 }} />
            <YAxis tick={{ fill: "#737373", fontSize: 10 }} />
            <Tooltip contentStyle={tooltipStyle} {...tooltipExtra} formatter={(v: unknown) => typeof v === "number" ? v.toFixed(2) : String(v ?? "")} />
            <Bar dataKey="MAE" fill="#ef4444" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div>
        <p className="text-xs text-muted mb-2 font-medium">RMSE (lower is better)</p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={rmseData} {...chartProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
            <XAxis dataKey="name" tick={{ fill: "#737373", fontSize: 10 }} />
            <YAxis tick={{ fill: "#737373", fontSize: 10 }} />
            <Tooltip contentStyle={tooltipStyle} {...tooltipExtra} formatter={(v: unknown) => typeof v === "number" ? v.toFixed(2) : String(v ?? "")} />
            <Bar dataKey="RMSE" fill="#f59e0b" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div>
        <p className="text-xs text-muted mb-2 font-medium">R² (higher is better)</p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={r2Data} {...chartProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
            <XAxis dataKey="name" tick={{ fill: "#737373", fontSize: 10 }} />
            <YAxis domain={[0, 1]} tick={{ fill: "#737373", fontSize: 10 }} />
            <Tooltip contentStyle={tooltipStyle} {...tooltipExtra} formatter={(v: unknown) => typeof v === "number" ? v.toFixed(4) : String(v ?? "")} />
            <Bar dataKey="R2" fill="#22c55e" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

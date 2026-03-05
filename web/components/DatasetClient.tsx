"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import KpiCard from "./KpiCard";
import FeatureTable from "./FeatureTable";
import type { FeatureStat } from "@/lib/types";

interface Props {
  features: FeatureStat[];
}

const tooltipStyle = {
  backgroundColor: "#1a1a1a",
  border: "1px solid #262626",
  borderRadius: 6,
  fontSize: 12,
};

export default function DatasetClient({ features }: Props) {
  const totalSamples = features[0]?.count ?? 0;
  const numericCount = features.filter((f) => f.dtype !== "str").length;
  const categoricalCount = features.filter((f) => f.dtype === "str").length;
  const totalMissing = features.reduce((acc, f) => acc + f.missing, 0);

  // Data type breakdown
  const dtypeCounts: Record<string, number> = {};
  features.forEach((f) => {
    dtypeCounts[f.dtype] = (dtypeCounts[f.dtype] ?? 0) + 1;
  });
  const dtypeData = Object.entries(dtypeCounts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count);

  const tooltipExtra = {
    itemStyle: { color: "#e5e5e5" } as React.CSSProperties,
    labelStyle: { color: "#737373" } as React.CSSProperties,
    cursor: { fill: "rgba(255,255,255,0.05)" },
  };

  const dtypeColors: Record<string, string> = {
    int64: "#f59e0b",
    float64: "#22c55e",
    str: "#3b82f6",
  };

  // Missing values (only show features with missing > 0)
  const missingData = features
    .filter((f) => f.missing > 0)
    .map((f) => ({ name: f.feature, missing: f.missing }))
    .sort((a, b) => b.missing - a.missing);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">Dataset Overview</h1>
        <p className="text-sm text-muted mt-1">Medical insurance dataset statistics</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard title="Total Samples" value={totalSamples} decimals={0} />
        <KpiCard title="Total Features" value={features.length} decimals={0} />
        <KpiCard title="Numeric Features" value={numericCount} decimals={0} />
        <KpiCard title="Categorical Features" value={categoricalCount} decimals={0} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Data Type Breakdown */}
        <div className="bg-card border border-border rounded-lg p-4">
          <h2 className="text-sm font-semibold mb-3">Data Type Breakdown</h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={dtypeData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
              <XAxis dataKey="name" tick={{ fill: "#737373", fontSize: 11 }} />
              <YAxis tick={{ fill: "#737373", fontSize: 11 }} />
              <Tooltip contentStyle={tooltipStyle} {...tooltipExtra} />
              <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                {dtypeData.map((entry) => (
                  <Cell key={entry.name} fill={dtypeColors[entry.name] ?? "#737373"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Missing Values */}
        <div className="bg-card border border-border rounded-lg p-4">
          <h2 className="text-sm font-semibold mb-3">Missing Values</h2>
          {missingData.length === 0 ? (
            <p className="text-sm text-muted">No missing values in any feature.</p>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={missingData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }} style={{ backgroundColor: "transparent" }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" fill="transparent" />
                <XAxis dataKey="name" tick={{ fill: "#737373", fontSize: 11 }} />
                <YAxis tick={{ fill: "#737373", fontSize: 11 }} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  {...tooltipExtra}
                  formatter={(v: unknown) => typeof v === "number" ? v.toLocaleString() : String(v ?? "")}
                />
                <Bar dataKey="missing" fill="#ef4444" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
          <p className="text-xs text-muted mt-2">
            Total missing cells: {totalMissing.toLocaleString()}
          </p>
        </div>
      </div>

      {/* Full Feature Table */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h2 className="text-sm font-semibold mb-3">All Features</h2>
        <FeatureTable features={features} />
      </div>
    </div>
  );
}

"use client";

import { useState, useMemo } from "react";
import type { FeatureStat } from "@/lib/types";
import { fmt } from "@/lib/utils";

type SortKey = "feature" | "dtype" | "count" | "mean" | "std" | "min" | "max" | "missing";

interface Props {
  features: FeatureStat[];
}

export default function FeatureTable({ features }: Props) {
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("feature");
  const [sortAsc, setSortAsc] = useState(true);

  const filtered = useMemo(() => {
    let result = features.filter((f) =>
      f.feature.toLowerCase().includes(search.toLowerCase())
    );

    result.sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;
      if (typeof aVal === "string" && typeof bVal === "string")
        return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      return sortAsc ? Number(aVal) - Number(bVal) : Number(bVal) - Number(aVal);
    });

    return result;
  }, [features, search, sortKey, sortAsc]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const SortHeader = ({ k, label }: { k: SortKey; label: string }) => (
    <th
      className="py-2 px-2 text-left cursor-pointer hover:text-foreground transition-colors select-none"
      onClick={() => handleSort(k)}
    >
      {label}
      {sortKey === k && <span className="ml-1">{sortAsc ? "↑" : "↓"}</span>}
    </th>
  );

  const cell = (val: number | null) =>
    val != null ? (
      <span className="font-mono">{fmt(val, 2)}</span>
    ) : (
      <span className="text-muted/50">-</span>
    );

  return (
    <div>
      <input
        type="text"
        placeholder="Filter features..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full max-w-xs bg-card border border-border rounded px-3 py-1.5 text-sm text-foreground placeholder:text-muted mb-3 focus:outline-none focus:ring-1 focus:ring-accent"
      />
      <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-card z-10">
            <tr className="border-b border-border text-muted uppercase">
              <SortHeader k="feature" label="Feature" />
              <SortHeader k="dtype" label="Type" />
              <SortHeader k="count" label="Count" />
              <SortHeader k="mean" label="Mean" />
              <SortHeader k="std" label="Std" />
              <SortHeader k="min" label="Min" />
              <SortHeader k="max" label="Max" />
              <SortHeader k="missing" label="Missing" />
            </tr>
          </thead>
          <tbody>
            {filtered.map((f) => (
              <tr
                key={f.feature}
                className="border-b border-border/30 hover:bg-card-hover transition-colors"
              >
                <td className="py-1.5 px-2 font-medium">{f.feature}</td>
                <td className="py-1.5 px-2">
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] ${
                      f.dtype === "str"
                        ? "bg-accent/15 text-accent"
                        : f.dtype === "float64"
                        ? "bg-success/15 text-success"
                        : "bg-warning/15 text-warning"
                    }`}
                  >
                    {f.dtype}
                  </span>
                </td>
                <td className="py-1.5 px-2 font-mono">{f.count.toLocaleString()}</td>
                <td className="py-1.5 px-2">{cell(f.mean)}</td>
                <td className="py-1.5 px-2">{cell(f.std)}</td>
                <td className="py-1.5 px-2">{cell(f.min)}</td>
                <td className="py-1.5 px-2">{cell(f.max)}</td>
                <td className="py-1.5 px-2">
                  <span className={f.missing > 0 ? "text-danger font-medium" : "text-muted"}>
                    {f.missing.toLocaleString()}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted mt-2">
        Showing {filtered.length} of {features.length} features
      </p>
    </div>
  );
}

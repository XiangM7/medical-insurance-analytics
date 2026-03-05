"use client";

import type { Split } from "@/lib/types";

interface Props {
  value: Split;
  onChange: (v: Split) => void;
}

export default function MetricToggle({ value, onChange }: Props) {
  return (
    <div className="inline-flex bg-card border border-border rounded-md p-0.5 text-xs">
      {(["val", "test"] as Split[]).map((s) => (
        <button
          key={s}
          onClick={() => onChange(s)}
          className={`px-3 py-1.5 rounded transition-colors capitalize ${
            value === s
              ? "bg-accent text-white font-medium"
              : "text-muted hover:text-foreground"
          }`}
        >
          {s === "val" ? "Validation" : "Test"}
        </button>
      ))}
    </div>
  );
}

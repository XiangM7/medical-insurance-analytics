"use client";

import type { ExperimentId } from "@/lib/types";

interface Props {
  value: ExperimentId;
  onChange: (v: ExperimentId) => void;
}

const options: { id: ExperimentId; label: string; short: string }[] = [
  { id: "original", label: "Original (Insurance Quality)", short: "Original" },
  { id: "high_risk", label: "High Risk (Binary)", short: "High Risk" },
  { id: "removed_stuff", label: "Removed Features (Ablation)", short: "Ablation" },
];

export default function ExperimentSelector({ value, onChange }: Props) {
  return (
    <div className="inline-flex bg-card border border-border rounded-md p-0.5 text-xs">
      {options.map((o) => (
        <button
          key={o.id}
          onClick={() => onChange(o.id)}
          title={o.label}
          className={`px-3 py-1.5 rounded transition-colors ${
            value === o.id
              ? "bg-accent text-white font-medium"
              : "text-muted hover:text-foreground"
          }`}
        >
          {o.short}
        </button>
      ))}
    </div>
  );
}

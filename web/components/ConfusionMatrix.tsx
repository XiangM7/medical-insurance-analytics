"use client";

import toast from "react-hot-toast";
import { confusionCellColor, modelLabel } from "@/lib/utils";

interface Props {
  model: string;
  labels: string[];
  matrix: number[][];
}

export default function ConfusionMatrix({ model, labels, matrix }: Props) {
  const maxVal = Math.max(...matrix.flat());

  const handleClick = (row: number, col: number) => {
    const actual = labels[row];
    const predicted = labels[col];
    const count = matrix[row][col];
    const total = matrix[row].reduce((a, b) => a + b, 0);
    const pct = ((count / total) * 100).toFixed(1);

    toast(
      `${modelLabel(model)}: ${count} samples (${pct}%)\nActual: ${actual} → Predicted: ${predicted}`,
      { duration: 3000 }
    );
  };

  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <h3 className="text-sm font-semibold mb-3">{modelLabel(model)}</h3>
      <div className="inline-block">
        {/* Header row */}
        <div className="flex">
          <div className="w-16" />
          {labels.map((l) => (
            <div key={l} className="w-16 text-center text-xs text-muted capitalize pb-1">
              {l}
            </div>
          ))}
        </div>
        {/* Matrix rows */}
        {matrix.map((row, i) => (
          <div key={i} className="flex">
            <div className="w-16 text-xs text-muted capitalize flex items-center pr-2 justify-end">
              {labels[i]}
            </div>
            {row.map((val, j) => (
              <button
                key={j}
                onClick={() => handleClick(i, j)}
                className="w-16 h-14 flex items-center justify-center text-xs font-mono border border-border/30 cursor-pointer hover:ring-1 hover:ring-accent transition-all"
                style={{ backgroundColor: confusionCellColor(val, maxVal, i === j) }}
                title={`Actual: ${labels[i]}, Predicted: ${labels[j]}: ${val}`}
              >
                {val.toLocaleString()}
              </button>
            ))}
          </div>
        ))}
        <div className="mt-2 flex gap-4 text-xs text-muted">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded" style={{ backgroundColor: "rgba(34,197,94,0.5)" }} />
            Correct
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded" style={{ backgroundColor: "rgba(239,68,68,0.3)" }} />
            Misclassified
          </span>
        </div>
      </div>
    </div>
  );
}

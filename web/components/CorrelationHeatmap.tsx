"use client";

import React, { useState, useRef } from "react";
import { heatColor } from "@/lib/utils";

interface Props {
  features: string[];
  matrix: number[][];
}

export default function CorrelationHeatmap({ features, matrix }: Props) {
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    row: string;
    col: string;
    val: number;
  } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const n = features.length;
  const cellSize = 14;

  const handleMouseMove = (
    e: React.MouseEvent,
    row: number,
    col: number
  ) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    setTooltip({
      x: e.clientX - rect.left + 10,
      y: e.clientY - rect.top - 30,
      row: features[row],
      col: features[col],
      val: matrix[row][col],
    });
  };

  return (
    <div className="relative" ref={containerRef}>
      <div
        className="overflow-auto max-h-[700px] max-w-full"
        style={{ position: "relative" }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: `80px repeat(${n}, ${cellSize}px)`,
            gridTemplateRows: `20px repeat(${n}, ${cellSize}px)`,
            gap: 0,
            width: "fit-content",
          }}
        >
          {/* Top-left empty cell */}
          <div />
          {/* Column headers */}
          {features.map((f, i) => (
            <div
              key={`h-${i}`}
              className="text-[7px] text-muted overflow-hidden"
              style={{
                writingMode: "vertical-rl",
                transform: "rotate(180deg)",
                height: 20,
                lineHeight: `${cellSize}px`,
              }}
              title={f}
            />
          ))}
          {/* Rows */}
          {matrix.map((row, i) => (
            <React.Fragment key={`row-${i}`}>
              <div
                className="text-[8px] text-muted truncate pr-1 flex items-center justify-end"
                style={{ height: cellSize }}
                title={features[i]}
              >
                {features[i]}
              </div>
              {row.map((val, j) => (
                <div
                  key={`${i}-${j}`}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    backgroundColor: heatColor(val, -1, 1),
                  }}
                  onMouseMove={(e) => handleMouseMove(e, i, j)}
                  onMouseLeave={() => setTooltip(null)}
                />
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute pointer-events-none bg-[#1a1a1a] border border-border rounded px-2 py-1 text-xs z-50 shadow-lg"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          <div className="font-medium text-foreground">{tooltip.val.toFixed(4)}</div>
          <div className="text-muted">
            {tooltip.row} × {tooltip.col}
          </div>
        </div>
      )}

      {/* Color scale legend */}
      <div className="flex items-center gap-2 mt-3 text-xs text-muted">
        <span>-1</span>
        <div
          className="h-3 w-40 rounded"
          style={{
            background: "linear-gradient(to right, #3b82f6, #ffffff, #ef4444)",
          }}
        />
        <span>+1</span>
      </div>
    </div>
  );
}

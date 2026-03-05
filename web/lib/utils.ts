export function fmt(n: number, decimals = 4): string {
  if (Math.abs(n) >= 1000) return n.toLocaleString("en-US", { maximumFractionDigits: 1 });
  if (Math.abs(n) < 0.01) return n.toExponential(2);
  return n.toFixed(decimals);
}

export function pct(n: number): string {
  return (n * 100).toFixed(2) + "%";
}

export function modelLabel(model: string): string {
  const labels: Record<string, string> = {
    gbr: "Gradient Boosting",
    rf: "Random Forest",
    logreg_l1: "Logistic Reg (L1)",
    logreg_l2: "Logistic Reg (L2)",
    ridge: "Ridge Regression",
    linear: "Linear Regression",
  };
  return labels[model] ?? model;
}

export function heatColor(val: number, min: number, max: number): string {
  const t = max === min ? 0.5 : (val - min) / (max - min);
  // blue (#3b82f6) -> white -> red (#ef4444)
  if (t < 0.5) {
    const s = t * 2;
    const r = Math.round(59 + (255 - 59) * s);
    const g = Math.round(130 + (255 - 130) * s);
    const b = Math.round(246 + (255 - 246) * s);
    return `rgb(${r},${g},${b})`;
  } else {
    const s = (t - 0.5) * 2;
    const r = Math.round(255 + (239 - 255) * s);
    const g = Math.round(255 + (68 - 255) * s);
    const b = Math.round(255 + (68 - 255) * s);
    return `rgb(${r},${g},${b})`;
  }
}

export function confusionCellColor(val: number, maxVal: number, isDiag: boolean): string {
  const intensity = maxVal === 0 ? 0 : val / maxVal;
  if (isDiag) {
    // green scale
    return `rgba(34, 197, 94, ${0.15 + intensity * 0.7})`;
  }
  // red scale
  const clampedIntensity = Math.min(intensity * 5, 1); // amplify off-diagonal
  return `rgba(239, 68, 68, ${0.05 + clampedIntensity * 0.6})`;
}

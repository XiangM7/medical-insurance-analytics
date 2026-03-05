import fs from "fs";
import path from "path";
import Papa from "papaparse";
import type {
  ClassificationMetric,
  RegressionMetric,
  ConfusionData,
  RunSummary,
  FeatureStat,
  Experiment,
  LearningCurveData,
} from "./types";

const dataDir = path.join(process.cwd(), "public", "data");

function readCSV<T>(filepath: string): T[] {
  const raw = fs.readFileSync(path.join(dataDir, filepath), "utf-8");
  const { data } = Papa.parse<T>(raw, { header: true, skipEmptyLines: true, dynamicTyping: true });
  return data;
}

function readJSON<T>(filepath: string): T {
  const raw = fs.readFileSync(path.join(dataDir, filepath), "utf-8");
  return JSON.parse(raw);
}

// --- Per-experiment loaders ---

function loadExperimentCls(subdir: string): ClassificationMetric[] {
  const file = subdir ? `${subdir}/metrics_cls.csv` : "metrics_cls.csv";
  return readCSV<ClassificationMetric>(file);
}

function loadExperimentConfusions(subdir: string): ConfusionData[] {
  const prefix = subdir ? `${subdir}/` : "";
  return ["gbr", "rf", "logreg_l1", "logreg_l2"].map((m) =>
    readJSON<ConfusionData>(`${prefix}confusion_${m}.json`)
  );
}

function loadExperimentSummary(subdir: string): RunSummary {
  const file = subdir ? `${subdir}/run_cls_summary.json` : "run_cls_summary.json";
  return readJSON<RunSummary>(file);
}

export function getAllExperiments(): Experiment[] {
  return [
    {
      id: "original",
      label: "Original (Insurance Quality)",
      description: "3-class classification on insurance_quality with all features",
      clsMetrics: loadExperimentCls(""),
      confusions: loadExperimentConfusions(""),
      summary: loadExperimentSummary(""),
    },
    {
      id: "high_risk",
      label: "High Risk (Binary)",
      description: "Binary classification on is_high_risk",
      clsMetrics: loadExperimentCls("high_risk"),
      confusions: loadExperimentConfusions("high_risk"),
      summary: loadExperimentSummary("high_risk"),
    },
    {
      id: "removed_stuff",
      label: "Removed Features (Ablation)",
      description: "3-class insurance_quality with key features removed",
      clsMetrics: loadExperimentCls("removed_stuff"),
      confusions: loadExperimentConfusions("removed_stuff"),
      summary: loadExperimentSummary("removed_stuff"),
    },
  ];
}

// --- Original single-experiment loaders (still used by dashboard/regression) ---

export function getClassificationMetrics(): ClassificationMetric[] {
  return loadExperimentCls("");
}

export function getRegressionMetrics(): RegressionMetric[] {
  return readCSV<RegressionMetric>("metrics_reg.csv");
}

export function getAllConfusionMatrices(): ConfusionData[] {
  return loadExperimentConfusions("");
}

export function getRunSummary(): RunSummary {
  return loadExperimentSummary("");
}

export function getFeatureStats(): FeatureStat[] {
  const summary = readCSV<Record<string, string | number | null>>("eda_summary.csv");
  const dtypes = readCSV<Record<string, string>>("eda_dtypes.csv");
  const missing = readCSV<Record<string, string | number>>("eda_missing.csv");

  const dtypeMap = new Map<string, string>();
  for (const row of dtypes) {
    const name = row[""] ?? row["Unnamed: 0"];
    if (name) dtypeMap.set(String(name), String(row["0"] ?? "unknown"));
  }

  const missingMap = new Map<string, number>();
  for (const row of missing) {
    const name = row[""] ?? row["Unnamed: 0"];
    if (name) missingMap.set(String(name), Number(row["0"] ?? 0));
  }

  return summary.map((row) => {
    const feature = String(row[""] ?? row["Unnamed: 0"] ?? "");
    return {
      feature,
      count: Number(row.count ?? 0),
      mean: row.mean != null && row.mean !== "" ? Number(row.mean) : null,
      std: row.std != null && row.std !== "" ? Number(row.std) : null,
      min: row.min != null && row.min !== "" ? Number(row.min) : null,
      max: row.max != null && row.max !== "" ? Number(row.max) : null,
      unique: row.unique != null && row.unique !== "" ? Number(row.unique) : null,
      top: row.top != null && row.top !== "" ? String(row.top) : null,
      dtype: dtypeMap.get(feature) ?? "unknown",
      missing: missingMap.get(feature) ?? 0,
    };
  });
}

export function getCorrelationMatrix(): { features: string[]; matrix: number[][] } {
  const raw = fs.readFileSync(path.join(dataDir, "eda_corr.csv"), "utf-8");
  const { data } = Papa.parse<string[]>(raw, { header: false, skipEmptyLines: true });

  const header = data[0].slice(1);
  const matrix = data.slice(1).map((row) => row.slice(1).map(Number));

  return { features: header, matrix };
}

export function getLearningCurves(): LearningCurveData {
  return readJSON<LearningCurveData>("learning_curves.json");
}

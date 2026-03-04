export interface ClassificationMetric {
  model: string;
  val_accuracy: number;
  val_f1_macro: number;
  val_mcc: number;
  test_accuracy: number;
  test_f1_macro: number;
  test_mcc: number;
  best_params: string;
}

export interface RegressionMetric {
  model: string;
  val_mae: number;
  val_rmse: number;
  val_r2: number;
  test_mae: number;
  test_rmse: number;
  test_r2: number;
  best_params: string;
}

export interface ConfusionData {
  model: string;
  val: { labels: string[]; matrix: number[][] };
  test: { labels: string[]; matrix: number[][] };
}

export interface RunSummary {
  best_model: string;
  best_test_mcc: number;
  target: string;
  derived_target: boolean;
}

export interface FeatureStat {
  feature: string;
  count: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  unique: number | null;
  top: string | null;
  dtype: string;
  missing: number;
}

export type Split = "val" | "test";

export type ExperimentId = "original" | "high_risk" | "removed_stuff";

export interface LearningCurveModel {
  train_sizes: number[];
  train_mean: number[];
  train_std: number[];
  val_mean: number[];
  val_std: number[];
}

export type LearningCurveData = Record<string, Record<string, LearningCurveModel>>;

export interface Experiment {
  id: ExperimentId;
  label: string;
  description: string;
  clsMetrics: ClassificationMetric[];
  confusions: ConfusionData[];
  summary: RunSummary;
}

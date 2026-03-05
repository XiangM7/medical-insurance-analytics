import { getAllExperiments, getRegressionMetrics } from "@/lib/data";
import ModelPerformanceClient from "@/components/ModelPerformanceClient";

export default function ModelPerformancePage() {
  const experiments = getAllExperiments();
  const regMetrics = getRegressionMetrics();

  return <ModelPerformanceClient experiments={experiments} regMetrics={regMetrics} />;
}

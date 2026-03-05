import {
  getAllExperiments,
  getRegressionMetrics,
  getFeatureStats,
} from "@/lib/data";
import DashboardClient from "@/components/DashboardClient";

export default function Home() {
  const experiments = getAllExperiments();
  const regMetrics = getRegressionMetrics();
  const features = getFeatureStats();

  return (
    <DashboardClient
      experiments={experiments}
      regMetrics={regMetrics}
      totalFeatures={features.length}
      totalSamples={features[0]?.count ?? 100000}
    />
  );
}

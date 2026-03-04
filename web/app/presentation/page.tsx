import {
  getAllExperiments,
  getRegressionMetrics,
  getFeatureStats,
  getCorrelationMatrix,
} from "@/lib/data";
import PresentationClient from "@/components/PresentationClient";

export default function PresentationPage() {
  const experiments = getAllExperiments();
  const regMetrics = getRegressionMetrics();
  const features = getFeatureStats();
  const { features: corrFeatures, matrix: corrMatrix } = getCorrelationMatrix();

  return (
    <PresentationClient
      experiments={experiments}
      regMetrics={regMetrics}
      features={features}
      corrFeatures={corrFeatures}
      corrMatrix={corrMatrix}
    />
  );
}

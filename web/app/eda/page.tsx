import { getFeatureStats, getCorrelationMatrix } from "@/lib/data";
import EDAClient from "@/components/EDAClient";

export default function EDAPage() {
  const features = getFeatureStats();
  const { features: corrFeatures, matrix: corrMatrix } = getCorrelationMatrix();

  return (
    <EDAClient
      features={features}
      corrFeatures={corrFeatures}
      corrMatrix={corrMatrix}
    />
  );
}

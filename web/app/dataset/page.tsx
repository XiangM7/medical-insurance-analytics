import { getFeatureStats } from "@/lib/data";
import DatasetClient from "@/components/DatasetClient";

export default function DatasetPage() {
  const features = getFeatureStats();
  return <DatasetClient features={features} />;
}

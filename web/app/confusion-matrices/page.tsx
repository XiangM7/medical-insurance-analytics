import { getAllExperiments } from "@/lib/data";
import ConfusionMatricesClient from "@/components/ConfusionMatricesClient";

export default function ConfusionMatricesPage() {
  const experiments = getAllExperiments();
  return <ConfusionMatricesClient experiments={experiments} />;
}

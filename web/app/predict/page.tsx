export const dynamic = "force-dynamic";

export default function PredictPage() {
  return (
    <div>
      <script
        dangerouslySetInnerHTML={{
          __html: `window.__PREDICT_API__ = "http://localhost:5050";`,
        }}
      />
      <PredictClientWrapper />
    </div>
  );
}

import PredictClient from "@/components/PredictClient";

function PredictClientWrapper() {
  return <PredictClient />;
}

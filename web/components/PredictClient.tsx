"use client";

import { useState } from "react";

const API = "http://localhost:5050";

interface PredictionResult {
  regression: {
    predicted_cost: number;
    actual_cost: number;
    error: number;
  };
  classification: {
    predicted_risk: string;
    actual_risk: string;
    probabilities: Record<string, number>;
    correct: boolean;
  };
  top_features: { name: string; importance: number }[];
}

function buildReasoning(
  patient: Record<string, unknown>,
  result: PredictionResult
): string {
  const isHighRisk = result.classification.predicted_risk === "1";
  const conf = Math.max(
    ...Object.values(result.classification.probabilities)
  );
  const top3 = result.top_features.filter((f) => f.importance > 0).slice(0, 3);

  const age = Number(patient.age ?? 0);
  const smoker = String(patient.smoker ?? "");
  const chronic = Number(patient.chronic_count ?? 0);
  const bmi = Number(patient.bmi ?? 0);
  const cost = result.regression.predicted_cost;

  const reasons: string[] = [];
  if (age >= 55) reasons.push(`age of ${age} (older patients carry higher risk)`);
  else if (age <= 30) reasons.push(`young age of ${age} (lower baseline risk)`);
  if (smoker === "Current") reasons.push("active smoking status");
  else if (smoker === "Never") reasons.push("non-smoker status");
  if (chronic >= 2) reasons.push(`${chronic} chronic conditions`);
  else if (chronic === 0) reasons.push("no chronic conditions");
  if (bmi >= 30) reasons.push(`elevated BMI of ${bmi.toFixed(1)}`);

  const reasonStr = reasons.length > 0 ? reasons.join(", ") : "overall feature profile";
  const topStr = top3.map((f) => f.name).join(", ");

  return `The model predicts this patient is ${isHighRisk ? "HIGH RISK" : "LOW RISK"} with ${(conf * 100).toFixed(1)}% confidence. Key drivers: ${topStr}. This is primarily due to ${reasonStr}. Predicted annual cost: $${cost.toLocaleString()}.`;
}

// Key features to display/edit in the form
const DISPLAY_FIELDS: {
  key: string;
  label: string;
  type: "number" | "text" | "select";
  options?: string[];
}[] = [
  { key: "age", label: "Age", type: "number" },
  { key: "sex", label: "Sex", type: "select", options: ["male", "female"] },
  { key: "bmi", label: "BMI", type: "number" },
  { key: "income", label: "Income", type: "number" },
  {
    key: "smoker",
    label: "Smoker",
    type: "select",
    options: ["never", "former", "current"],
  },
  { key: "chronic_count", label: "Chronic Conditions", type: "number" },
  { key: "systolic_bp", label: "Systolic BP", type: "number" },
  { key: "diastolic_bp", label: "Diastolic BP", type: "number" },
  { key: "hba1c", label: "HbA1c", type: "number" },
  { key: "ldl", label: "LDL Cholesterol", type: "number" },
  { key: "medication_count", label: "Medications", type: "number" },
  { key: "claims_count", label: "Claims Count", type: "number" },
  { key: "annual_medical_cost", label: "Annual Medical Cost", type: "number" },
  { key: "deductible", label: "Deductible", type: "number" },
  { key: "hypertension", label: "Hypertension", type: "number" },
  { key: "diabetes", label: "Diabetes", type: "number" },
  { key: "cardiovascular_disease", label: "Cardiovascular Disease", type: "number" },
  { key: "cancer_history", label: "Cancer History", type: "number" },
];

export default function PredictClient() {
  const [patient, setPatient] = useState<Record<string, unknown> | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRandomPatient = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/api/random-patient`);
      if (!res.ok) throw new Error("Failed to fetch patient");
      const data = await res.json();
      setPatient(data);
    } catch (e) {
      setError(
        `Cannot connect to prediction API. Make sure the Flask server is running:\n  source .venv/bin/activate && python src/predict_api.py`
      );
    } finally {
      setLoading(false);
    }
  };

  const runPrediction = async () => {
    if (!patient) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient),
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError("Prediction request failed. Check the Flask API.");
    } finally {
      setLoading(false);
    }
  };

  const updateField = (key: string, value: unknown) => {
    if (!patient) return;
    setPatient({ ...patient, [key]: value });
    setResult(null);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">Live Model Demo</h1>
        <p className="text-sm text-muted mt-1">
          Feed unseen patient data to the trained models and see predictions in
          real-time
        </p>
      </div>

      {/* Controls */}
      <div className="flex gap-3">
        <button
          onClick={fetchRandomPatient}
          disabled={loading}
          className="bg-accent hover:bg-accent/80 text-white px-4 py-2 rounded text-sm font-medium transition-colors disabled:opacity-50"
        >
          {loading && !patient ? "Loading..." : "Random Test Patient"}
        </button>
        {patient && (
          <button
            onClick={runPrediction}
            disabled={loading}
            className="bg-success hover:bg-success/80 text-white px-4 py-2 rounded text-sm font-medium transition-colors disabled:opacity-50"
          >
            {loading && patient ? "Predicting..." : "Run Prediction"}
          </button>
        )}
      </div>

      {error && (
        <div className="bg-danger/10 border border-danger/30 rounded-lg p-4 text-sm text-danger whitespace-pre-wrap">
          {error}
        </div>
      )}

      {patient && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Patient Input Card */}
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold">Patient Features</h2>
              <span className="text-xs text-muted">
                Test row #{String(patient._index ?? "")}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {DISPLAY_FIELDS.map((field) => {
                const val = patient[field.key];
                return (
                  <div key={field.key} className="flex flex-col gap-0.5">
                    <label className="text-[10px] text-muted uppercase tracking-wider">
                      {field.label}
                    </label>
                    {field.type === "select" ? (
                      <select
                        value={String(val ?? "")}
                        onChange={(e) => updateField(field.key, e.target.value)}
                        className="bg-background border border-border rounded px-2 py-1 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
                      >
                        {field.options?.map((opt) => (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        value={val != null ? String(val) : ""}
                        onChange={(e) => {
                          const v = e.target.value;
                          updateField(
                            field.key,
                            v === "" ? null : parseFloat(v)
                          );
                        }}
                        className="bg-background border border-border rounded px-2 py-1 text-xs font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
                      />
                    )}
                  </div>
                );
              })}
            </div>
            <p className="text-[10px] text-muted mt-3">
              Edit any field above, then click &quot;Run Prediction&quot; to see
              how it affects the output.
            </p>
          </div>

          {/* Results Card */}
          <div className="space-y-4">
            {result ? (
              <>
                {/* Regression Result */}
                <div className="bg-card border border-border rounded-lg p-4">
                  <h2 className="text-sm font-semibold mb-3">
                    Cost Prediction
                    <span className="text-xs text-muted font-normal ml-2">
                      Ridge Regression (R2=0.966)
                    </span>
                  </h2>
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div>
                      <p className="text-2xl font-bold text-accent font-mono">
                        ${result.regression.predicted_cost.toLocaleString()}
                      </p>
                      <p className="text-xs text-muted mt-1">Predicted</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-muted font-mono">
                        ${result.regression.actual_cost.toLocaleString()}
                      </p>
                      <p className="text-xs text-muted mt-1">Actual</p>
                    </div>
                    <div>
                      <p
                        className={`text-2xl font-bold font-mono ${
                          result.regression.error < 500
                            ? "text-success"
                            : result.regression.error < 1000
                            ? "text-warning"
                            : "text-danger"
                        }`}
                      >
                        ${result.regression.error.toLocaleString()}
                      </p>
                      <p className="text-xs text-muted mt-1">Error</p>
                    </div>
                  </div>
                </div>

                {/* Classification Result */}
                <div className="bg-card border border-border rounded-lg p-4">
                  <h2 className="text-sm font-semibold mb-3">
                    Risk Classification
                    <span className="text-xs text-muted font-normal ml-2">
                      Gradient Boosting (MCC=1.0)
                    </span>
                  </h2>
                  <div className="flex items-center gap-4">
                    <div
                      className={`text-center px-6 py-3 rounded-lg border ${
                        result.classification.predicted_risk === "1"
                          ? "bg-danger/10 border-danger/30"
                          : "bg-success/10 border-success/30"
                      }`}
                    >
                      <p
                        className={`text-lg font-bold ${
                          result.classification.predicted_risk === "1"
                            ? "text-danger"
                            : "text-success"
                        }`}
                      >
                        {result.classification.predicted_risk === "1"
                          ? "HIGH RISK"
                          : "LOW RISK"}
                      </p>
                      <p className="text-xs text-muted mt-1">Predicted</p>
                    </div>
                    <div className="text-center">
                      <p
                        className={`text-sm font-medium ${
                          result.classification.correct
                            ? "text-success"
                            : "text-danger"
                        }`}
                      >
                        {result.classification.correct
                          ? "Correct"
                          : "Incorrect"}
                      </p>
                      <p className="text-xs text-muted mt-0.5">
                        Actual:{" "}
                        {result.classification.actual_risk === "1"
                          ? "High Risk"
                          : "Low Risk"}
                      </p>
                    </div>
                    <div className="flex-1">
                      <p className="text-[10px] text-muted mb-1">
                        Confidence
                      </p>
                      {Object.entries(result.classification.probabilities).map(
                        ([cls, prob]) => (
                          <div key={cls} className="flex items-center gap-2 mb-1">
                            <span className="text-xs w-16">
                              {cls === "1" ? "High" : "Low"}
                            </span>
                            <div className="flex-1 bg-background rounded-full h-2 overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  cls === "1" ? "bg-danger" : "bg-success"
                                }`}
                                style={{ width: `${prob * 100}%` }}
                              />
                            </div>
                            <span className="text-xs font-mono w-14 text-right">
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                </div>

                {/* Feature Importance */}
                <div className="bg-card border border-border rounded-lg p-4">
                  <h2 className="text-sm font-semibold mb-3">
                    Top Contributing Features
                    <span className="text-xs text-muted font-normal ml-2">
                      GBR feature importance
                    </span>
                  </h2>
                  <div className="space-y-1.5">
                    {result.top_features.map((f) => {
                      const maxImp = result.top_features[0].importance;
                      const pct = (f.importance / maxImp) * 100;
                      return (
                        <div key={f.name} className="flex items-center gap-2">
                          <span className="text-xs w-40 truncate">
                            {f.name}
                          </span>
                          <div className="flex-1 bg-background rounded-full h-2 overflow-hidden">
                            <div
                              className="h-full bg-accent rounded-full"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                          <span className="text-xs font-mono w-14 text-right text-muted">
                            {(f.importance * 100).toFixed(1)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
                {/* Reasoning */}
                <div className="bg-card border border-accent/30 rounded-lg p-4">
                  <h2 className="text-sm font-semibold mb-2 text-accent">
                    Model Reasoning
                  </h2>
                  <p className="text-sm text-muted leading-relaxed">
                    {buildReasoning(patient!, result)}
                  </p>
                </div>
              </>
            ) : (
              <div className="bg-card border border-border rounded-lg p-8 flex items-center justify-center">
                <p className="text-sm text-muted">
                  Click &quot;Run Prediction&quot; to see model output
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {!patient && !error && (
        <div className="bg-card border border-border rounded-lg p-12 text-center">
          <p className="text-muted text-sm">
            Click &quot;Random Test Patient&quot; to load a patient from the
            held-out test set (data the model has never seen during training)
          </p>
        </div>
      )}
    </div>
  );
}

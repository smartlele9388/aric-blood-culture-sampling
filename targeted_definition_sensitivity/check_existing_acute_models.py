"""Check all 24 existing Figure 5 estimates against aggregate model sources.

This verifies carried-forward results and provenance; it does not refit a model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-source-dir", type=Path, default=ROOT / "acute_models_audit/model_sources")
parser.add_argument("--source-data", type=Path, default=ROOT / "acute_models_audit/all_24_existing_acute_estimates.csv")
parser.add_argument("--output-root", type=Path, default=ROOT / "acute_models_audit")
args = parser.parse_args()
MODEL = args.model_source_dir
sources = [
    {"model": "Retrospective component model",
     "sampling": MODEL / "sampling_components_specimens_final_coefficients.csv",
     "bsi": MODEL / "bsi_components_final_r_coefficients.csv"},
    {"model": "Diagnosis-free landmark-fixed model",
     "sampling": MODEL / "sampling_acute_landmark_only_specimens_coefficients.csv",
     "bsi": MODEL / "bsi_acute_landmark_only_r_coefficients.csv"},
    {"model": "Diagnosis-free prior-window updated model",
     "sampling": MODEL / "sampling_acute_daily_prior_window_specimens_coefficients.csv",
     "bsi": MODEL / "bsi_acute_daily_prior_window_r_coefficients.csv"},
]
data = pd.read_csv(args.source_data)
results = []
for source in sources:
    for endpoint, key, ratio in (("Sampling", "sampling", "rate_ratio"),
                                  ("Detected line-timed BSI", "bsi", "hazard_ratio")):
        path = source[key]
        original = pd.read_csv(path)
        rows = data.loc[data.model.eq(source["model"]) & data.endpoint.eq(endpoint)]
        assert len(rows) == 4
        for row in rows.itertuples():
            hit = original.loc[original.term.eq(row.source_term)]
            assert len(hit) == 1
            values = [float(hit.iloc[0][v]) for v in (ratio, "ci_lower", "ci_upper", "p_value")]
            assert np.allclose(values, [row.estimate, row.ci_lower, row.ci_upper, row.p_value],
                               rtol=1e-13, atol=1e-14)
        results.append({"model": source["model"], "endpoint": endpoint,
                        "source_file": path.name, "checked_estimates": len(rows),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
assert len(data) == 24
assert len(data.model.unique()) == 3
output = args.output_root
output.mkdir(exist_ok=True)
data.to_csv(output / "all_24_existing_acute_estimates.csv", index=False)
report = {"status": "executed_source_and_numeric_audit_no_refit",
          "rows_checked": len(data),
          "intervals_crossing_one_shown": int(((data.ci_lower <= 1) & (data.ci_upper >= 1)).sum()),
          "model_source_checks": results,
          "scope": "S0 carried-forward acute/component models; S1 primary fits are separate",
          "boundary": "Diagnosis-code exclusion answers a different question and does not restore the original score's diagnosis-availability time"}
(output / "EXISTING_ACUTE_MODEL_CHECK.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({k: report[k] for k in ("status", "rows_checked", "intervals_crossing_one_shown")}))

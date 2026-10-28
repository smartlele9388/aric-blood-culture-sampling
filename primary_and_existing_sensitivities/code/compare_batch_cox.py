"""Aggregate coefficient and robust-SE comparison for R Cox batch vs 10/04 outputs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / "ARIC_revision_20260920" / "association_audit_20261003" / "corrected_models_final"
BATCH = HERE / "r_batch_results"


def r_to_python(name: str) -> str:
    pieces = name.split(":")
    converted = []
    for part in pieces:
        if part.startswith("exposure_vulnerability_category"):
            converted.append("C(exposure_vulnerability_category, Treatment(reference='low'))[T." +
                             part.removeprefix("exposure_vulnerability_category") + "]")
        elif part.startswith(("icu_group", "admission_group", "cvc_type_group")):
            variable = next(v for v in ("icu_group", "admission_group", "cvc_type_group") if part.startswith(v))
            converted.append(f"C({variable})[T.{part.removeprefix(variable)}]")
        else:
            converted.append(part)
    return ":".join(converted)


def canonical(term: str) -> str:
    return ":".join(sorted(term.split(":")))


def main():
    comparisons = []
    report = []
    for path in sorted(BATCH.glob("*_coefficients.csv")):
        if path.name == "batch_coefficients.csv":
            continue
        model = path.name.removesuffix("_coefficients.csv")
        py = OLD / (model.removesuffix("_r") + "_coefficients.csv")
        if not py.exists():
            report.append({"model": model, "status": "no prior Python counterpart", "r_file": str(path)})
            continue
        r = pd.read_csv(path)
        p = pd.read_csv(py)
        r["python_term"] = r.term.map(r_to_python)
        r["canonical_term"] = r.python_term.map(canonical)
        p["canonical_term"] = p.term.map(canonical)
        joined = r.merge(p[["canonical_term", "term", "coefficient", "robust_se", "hazard_ratio", "ci_lower", "ci_upper", "p_value"]],
                         on="canonical_term", suffixes=("_r", "_python"), validate="one_to_one")
        if len(joined) != len(r) or len(joined) != len(p):
            raise AssertionError(f"Term mismatch in {model}: R={len(r)} Python={len(p)} matched={len(joined)}")
        joined["r_model"] = model
        joined["coefficient_difference"] = joined.coefficient_r - joined.coefficient_python
        joined["robust_se_difference"] = joined.robust_se_r - joined.robust_se_python
        comparisons.append(joined)
        report.append({"model": model, "status": "matched", "terms": len(joined),
                       "max_abs_coefficient_difference": float(joined.coefficient_difference.abs().max()),
                       "max_abs_robust_se_difference": float(joined.robust_se_difference.abs().max()),
                       "terms_crossing_p_0_05": int(((joined.p_value_r < .05) != (joined.p_value_python < .05)).sum())})
    pd.concat(comparisons, ignore_index=True).to_csv(HERE / "r_vs_python_batch_all_terms.csv", index=False)
    pd.DataFrame(report).to_csv(HERE / "r_vs_python_batch_model_summary.csv", index=False)
    summary = {"models_r_fitted": len(report), "models_matched": sum(x["status"] == "matched" for x in report),
               "max_abs_coefficient_difference_across_models": max(x["max_abs_coefficient_difference"] for x in report if x["status"] == "matched"),
               "models_with_coefficient_difference_gt_1e-6": [x["model"] for x in report if x["status"] == "matched" and x["max_abs_coefficient_difference"] > 1e-6],
               "models_with_p_0_05_crossings": [x["model"] for x in report if x["status"] == "matched" and x["terms_crossing_p_0_05"] > 0]}
    (HERE / "r_vs_python_batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

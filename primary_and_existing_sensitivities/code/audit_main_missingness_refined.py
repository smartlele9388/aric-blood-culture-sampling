"""Type-aware aggregate missingness table for corrected main Cox risk intervals."""
from pathlib import Path
import json
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "ARIC_revision_20260920"
SEC = ROOT / "ARIC_revision_20261009/secure_derived_inputs"
sys.path.insert(0, str(BASE / "association_audit_20261003/code"))
from coding_rules_final import correct_frame  # noqa: E402

KINDS = {
    "exposure_vulnerability_category": "category",
    "exposure_indication_present": "documented_binary",
    "age_per_10y": "continuous",
    "female": "binary_demographic",
    "icu_group": "category",
    "admission_group": "category",
    "multiple_line": "documented_binary",
    "cvc_type_group": "category",
    "exposure_antibiotic_exposure": "documented_binary",
    "exposure_lactate_available": "measurement_status",
    "exposure_aki_assessable": "measurement_status",
    "lag_lactate_high": "source_measurement_with_NA",
    "lag_aki": "source_measurement_with_NA",
    "chf": "admission_diagnosis_flag", "copd": "admission_diagnosis_flag",
    "ckd": "admission_diagnosis_flag", "diabetes": "admission_diagnosis_flag",
    "cancer": "admission_diagnosis_flag", "metastatic_cancer": "admission_diagnosis_flag",
    "liver_disease": "admission_diagnosis_flag", "cerebrovascular_disease": "admission_diagnosis_flag",
    "peripheral_vascular_disease": "admission_diagnosis_flag",
    "immunosuppression": "admission_diagnosis_flag",
}


def meaning(name: str, kind: str) -> str:
    if kind == "source_measurement_with_NA":
        return "NA is unavailable/unevaluable source measurement; score uses zero documented-positive point, not imputed normal."
    if kind == "measurement_status":
        return "0 explicitly denotes no eligible measurement/comparison, retained as adjustment indicator."
    if kind == "admission_diagnosis_flag":
        return "0 means no matching discharge/admission diagnosis record; timing before landmark unverified."
    if kind == "documented_binary":
        return "0 means no documented positive under source rule; not proof of true absence."
    if name == "cvc_type_group":
        return "Unknown device type is an explicit category."
    if kind == "continuous":
        return "Native missingness reported; numeric zero is a valid value, not a missingness code."
    return "Native missingness reported separately from retained categories."


def main():
    original = pd.read_csv(BASE / "secure_derived_inputs/revised_risk_day3.csv.gz")
    delta = pd.read_csv(SEC / "final_score_deltas_stay.csv.gz")
    d = correct_frame(original, delta)
    assert (len(d), int(d.event.sum()), d.stay_id.nunique()) == (60272, 268, 13863)
    rows = []
    for name, kind in KINDS.items():
        s = d[name]
        missing = s.isna()
        is_binary = kind in {"documented_binary", "binary_demographic", "measurement_status", "admission_diagnosis_flag", "source_measurement_with_NA"}
        rows.append({
            "variable": name, "variable_type": kind,
            "intervals": len(d), "events": int(d.event.sum()),
            "native_missing_intervals": int(missing.sum()),
            "native_missing_stays": int(d.loc[missing, "stay_id"].nunique()),
            "events_with_native_missing": int(d.loc[missing, "event"].sum()),
            "coded_zero_intervals": int(s.eq(0).sum()) if is_binary else None,
            "coded_zero_events": int(d.loc[s.eq(0), "event"].sum()) if is_binary else None,
            "coded_one_intervals": int(s.eq(1).sum()) if is_binary else None,
            "coded_one_events": int(d.loc[s.eq(1), "event"].sum()) if is_binary else None,
            "explicit_unknown_intervals": int(s.astype(str).str.lower().eq("unknown").sum()) if kind == "category" else None,
            "number_of_observed_categories": int(s.nunique(dropna=True)) if kind == "category" else None,
            "interpretation": meaning(name, kind),
        })
    pd.DataFrame(rows).to_csv(HERE / "corrected_main_variable_level_missingness.csv", index=False)
    info = {"intervals": len(d), "stays": int(d.stay_id.nunique()), "patients": int(d.subject_id.nunique()),
            "events": int(d.event.sum()),
            "all_main_model_covariates_nonmissing_after_documented_positive_encoding": bool(
                d[[x for x in KINDS if x not in ("lag_lactate_high", "lag_aki")]].notna().all().all()),
            "source_lactate_unmeasured_intervals": int(d.lag_lactate_high.isna().sum()),
            "source_creatinine_unevaluable_intervals": int(d.lag_aki.isna().sum()),
            "diagnosis_code_timing_proven_prior_to_landmark": False}
    (HERE / "corrected_main_missingness_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

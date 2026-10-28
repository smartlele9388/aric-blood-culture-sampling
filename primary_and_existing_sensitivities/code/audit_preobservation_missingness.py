"""Aggregate variable-level missingness for diagnosis-free acute Cox models."""
from pathlib import Path
import json

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEC = ROOT / "ARIC_revision_20261009" / "secure_derived_inputs"
FRAMES = {
    "landmark_only": SEC / "preobservation_landmark_day3_risk.csv.gz",
    "daily_prior_window": SEC / "preobservation_daily_prior_window_day3_risk.csv.gz",
}
VARIABLES = {
    "exposure_mechanical_ventilation": ("documented_binary", "0 = no recorded prior-window ventilation; not a verified clinical negative"),
    "exposure_creatinine_rise_documented": ("documented_binary", "0 combines measured non-rise and unassessable comparison; use exposure_aki_assessable"),
    "exposure_lactate_high_documented": ("documented_binary", "0 combines measured value below threshold and no measured lactate; use exposure_lactate_available"),
    "exposure_indication_present": ("documented_binary", "0 = no recorded selected treatment marker; not proof of no line indication"),
    "age_per_10y": ("continuous", "age centered at 60 years and divided by 10"),
    "female": ("documented_binary", "recorded sex covariate"),
    "icu_group": ("categorical", "constructed ICU grouping; unknown must remain explicit"),
    "admission_group": ("categorical", "constructed admission grouping"),
    "multiple_line": ("documented_binary", "0 = no documented multiple-line overlap in source window"),
    "cvc_type_group": ("categorical", "unknown is an explicit retained category, not native missingness"),
    "exposure_antibiotic_exposure": ("documented_binary", "0 = no documented antibiotic exposure in source window"),
    "exposure_lactate_available": ("assessment_status", "0 = no eligible lactate measurement; retain separately from documented high lactate"),
    "exposure_aki_assessable": ("assessment_status", "0 = unable to evaluate creatinine rise; retain separately from documented rise"),
}


def main():
    rows = []
    summary = {}
    for model, path in FRAMES.items():
        d = pd.read_csv(path)
        assert (len(d), d.stay_id.nunique(), d.subject_id.nunique(), int(d.event.sum())) == (60272, 13863, 12757, 268)
        assert not d[list(VARIABLES)].isna().any().any()
        summary[model] = {"intervals": len(d), "events": int(d.event.sum()),
                          "stays": int(d.stay_id.nunique()), "patients": int(d.subject_id.nunique()),
                          "any_native_missing_in_cox_covariates": False,
                          "lactate_unmeasured_intervals": int(d.exposure_lactate_available.eq(0).sum()),
                          "stays_with_any_lactate_unmeasured_interval": int(d.loc[d.exposure_lactate_available.eq(0), "stay_id"].nunique()),
                          "creatinine_unevaluable_intervals": int(d.exposure_aki_assessable.eq(0).sum()),
                          "stays_with_any_creatinine_unevaluable_interval": int(d.loc[d.exposure_aki_assessable.eq(0), "stay_id"].nunique()),
                          "lactate_high_zero_among_unmeasured": int((d.exposure_lactate_high_documented.eq(0) & d.exposure_lactate_available.eq(0)).sum()),
                          "creatinine_rise_zero_among_unevaluable": int((d.exposure_creatinine_rise_documented.eq(0) & d.exposure_aki_assessable.eq(0)).sum())}
        for name, (kind, meaning) in VARIABLES.items():
            s = d[name]
            row = {"analysis": model, "variable": name, "type": kind, "intervals": len(d),
                   "native_missing_intervals": int(s.isna().sum()),
                   "BSI_events_in_native_missing_intervals": int(d.loc[s.isna(), "event"].sum()),
                   "explicit_unknown_intervals": int(s.astype(str).str.lower().eq("unknown").sum()) if kind == "categorical" else None,
                   "zero_intervals": int(s.eq(0).sum()) if kind in ("documented_binary", "assessment_status") else None,
                   "zero_BSI_events": int(d.loc[s.eq(0), "event"].sum()) if kind in ("documented_binary", "assessment_status") else None,
                   "one_intervals": int(s.eq(1).sum()) if kind in ("documented_binary", "assessment_status") else None,
                   "one_BSI_events": int(d.loc[s.eq(1), "event"].sum()) if kind in ("documented_binary", "assessment_status") else None,
                   "interpretation": meaning}
            rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / "preobservation_variable_level_missingness.csv", index=False)
    (HERE / "preobservation_missingness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

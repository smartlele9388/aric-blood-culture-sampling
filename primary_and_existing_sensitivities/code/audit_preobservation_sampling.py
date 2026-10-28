"""Independent arithmetic check of four new diagnosis-free sampling Poisson fits."""
from pathlib import Path
import json

import pandas as pd

from audit_saved_primary_models import BASE, ROOT, SAMPLE, check_poisson

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "ARIC_revision_20261009" / "analysis_core" / "results"
RHS = (
    "prior_ventilation + prior_creatinine_rise + prior_high_lactate + "
    "exposure_indication_present + age_per_10y + female + C(icu_group) + "
    "C(admission_group) + multiple_line + C(cvc_type_group) + "
    "exposure_antibiotic_exposure + exposure_lactate_available + exposure_aki_assessable + "
    "C(time_band, Treatment(reference='0 to <2'))"
)
FIELDS = [
    "exposure_mechanical_ventilation", "exposure_creatinine_rise_documented",
    "exposure_lactate_high_documented", "exposure_indication_present", "multiple_line",
    "exposure_antibiotic_exposure", "exposure_lactate_available", "exposure_aki_assessable",
]


def frame(mode: str, original: pd.DataFrame) -> pd.DataFrame:
    d = original.copy()
    d["exposure_mechanical_ventilation"] = d.prior_ventilation.astype(int)
    d["exposure_creatinine_rise_documented"] = d.lag_aki.eq(1).astype(int)
    d["exposure_lactate_high_documented"] = d.lag_lactate_high.eq(1).astype(int)
    if mode == "landmark_only":
        first = d.sort_values(["stay_id", "t_start", "t_stop"]).drop_duplicates("stay_id").set_index("stay_id")
        assert first.t_start.eq(0).all()
        for name in FIELDS:
            d[name] = d.stay_id.map(first[name])
            assert d[name].notna().all()
    d["prior_ventilation"] = d.exposure_mechanical_ventilation
    d["prior_creatinine_rise"] = d.exposure_creatinine_rise_documented
    d["prior_high_lactate"] = d.exposure_lactate_high_documented
    return d


def main():
    original = pd.read_csv(SAMPLE)
    output = {}
    for mode in ("landmark_only", "daily_prior_window"):
        d = frame(mode, original)
        for kind, outcome in (("specimens", "collection_count"), ("occasions", "occasion_count")):
            stem = f"sampling_acute_{mode}_{kind}"
            output[stem] = check_poisson(d, stem, RESULTS / f"{stem}_coefficients.csv",
                                         RESULTS / f"{stem}_covariance.npz", RHS, outcome)
    (HERE / "preobservation_sampling_arithmetic_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({name: {"outcome_count": result["outcome_count"],
                             "max_abs_saved_vs_rebuilt_covariance": result["max_abs_saved_vs_rebuilt_covariance"]}
                      for name, result in output.items()}, indent=2))


if __name__ == "__main__":
    main()

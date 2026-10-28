"""Rebuild the baseline table after final coding correction, aggregate only."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path((__import__("os").environ.get("ARIC_PROJECT_ROOT", "ARIC_PROJECT_ROOT") + ''))
RISK = ROOT / "ARIC_revision_20260920" / "secure_derived_inputs" / "revised_risk_day3.csv.gz"
DELTA = ROOT / "ARIC_revision_20261009" / "secure_derived_inputs" / "final_score_deltas_stay.csv.gz"
TREATMENTS = ROOT / "ARIC_revision_20260920" / "secure_derived_inputs" / "corrected_treatments_day3.csv.gz"
OUT = ROOT / "ARIC_revision_20261009" / "documents" / "baseline_characteristics_corrected_full.csv"

FLAG_LABELS = [
    ("Heart failure, n (%)", "chf"),
    ("Chronic obstructive pulmonary disease, n (%)", "copd"),
    ("Chronic kidney disease, n (%)", "ckd"),
    ("Diabetes, n (%)", "diabetes"),
    ("Cancer, n (%)", "corrected_cancer"),
    ("Metastatic cancer, n (%)", "corrected_meta"),
    ("Liver disease, n (%)", "liver_disease"),
    ("Cerebrovascular disease, n (%)", "cerebrovascular_disease"),
    ("Peripheral vascular disease, n (%)", "peripheral_vascular_disease"),
    ("Immunosuppression-related flag, n (%)", "immunosuppression"),
    ("Treatment marker, n (%)", "exposure_indication_present"),
    ("Invasive ventilation, n (%)", "exposure_mechanical_ventilation"),
    ("Vasoactive therapy, n (%)", "exposure_vasopressor"),
    ("Renal replacement treatment, n (%)", "exposure_renal_replacement_therapy"),
    ("Total parenteral nutrition, n (%)", "baseline_tpn"),
    ("Antibiotic prescription, n (%)", "exposure_antibiotic_exposure"),
    ("Lactate measured, n (%)", "exposure_lactate_available"),
    ("Creatinine rise assessable, n (%)", "exposure_aki_assessable"),
    ("Multiple qualifying CVC records, n (%)", "multiple_line"),
]


def n_pct(frame: pd.DataFrame, key: str, denominator: str | None = None) -> str:
    numerator = int(frame[key].fillna(0).astype(bool).sum())
    denom = int(frame[denominator].fillna(0).astype(bool).sum()) if denominator else len(frame)
    return f"{numerator} ({numerator / denom * 100:.1f})" if denom else f"{numerator} (NA)"


def main() -> None:
    needed = ["stay_id", "subject_id", "t_start", "age", "female", "exposure_vulnerability_burden_score", "cancer", "metastatic_cancer", "lag_aki", "lag_lactate_high", "icu_group", "admission_group", "cvc_type_group", "exposure_lactate_available", "exposure_aki_assessable"] + [key for _, key in FLAG_LABELS if key not in {"corrected_cancer", "corrected_meta", "baseline_tpn"}]
    risk = pd.read_csv(RISK, usecols=list(dict.fromkeys(needed)))
    delta = pd.read_csv(DELTA)
    if delta.stay_id.duplicated().any():
        raise ValueError("Nonunique stay delta")
    data = risk.sort_values(["stay_id", "t_start"]).drop_duplicates("stay_id")
    treatments = pd.read_csv(TREATMENTS, usecols=["stay_id", "local_day", "tpn"])
    treatment_baseline = treatments.loc[treatments.local_day.eq(0), ["stay_id", "tpn"]].rename(columns={"tpn": "baseline_tpn"})
    if treatment_baseline.stay_id.duplicated().any():
        raise ValueError("Nonunique baseline treatment stay")
    data = data.merge(treatment_baseline, on="stay_id", how="left", validate="one_to_one")
    data = data.merge(delta, on="stay_id", how="left", validate="one_to_one")
    for key in ("delta_cancer", "delta_meta", "delta_score"):
        data[key] = data[key].fillna(0)
    data["corrected_cancer"] = data.cancer + data.delta_cancer
    data["corrected_meta"] = data.metastatic_cancer + data.delta_meta
    data["corrected_score"] = data.exposure_vulnerability_burden_score + data.delta_score
    data["group"] = pd.cut(data.corrected_score, [-1, 1, 3, float("inf")], labels=["Low", "Moderate", "High"])
    if data.groupby("group", observed=True).size().to_dict() != {"Low": 3046, "Moderate": 6090, "High": 4727}:
        raise AssertionError("Corrected baseline counts do not reconcile")
    grouped = [("Overall", data)] + [(name, data.loc[data.group == name]) for name in ("Low", "Moderate", "High")]
    rows: list[dict[str, str]] = []

    def add(label: str, format_one) -> None:
        rows.append({"Characteristic": label, **{name: format_one(frame) for name, frame in grouped}})

    add("ICU stays, n", lambda f: str(len(f)))
    add("Patients, n", lambda f: str(f.subject_id.nunique()))
    add("Age, years, median (IQR)", lambda f: f"{f.age.quantile(.5):.1f} ({f.age.quantile(.25):.1f}–{f.age.quantile(.75):.1f})")
    add("Documented vulnerability score, median (IQR)", lambda f: f"{f.corrected_score.quantile(.5):.1f} ({f.corrected_score.quantile(.25):.1f}–{f.corrected_score.quantile(.75):.1f})")
    add("Female, n (%)", lambda f: n_pct(f, "female"))
    for label, key in FLAG_LABELS:
        add(label, lambda f, k=key: n_pct(f, k))
    add("Creatinine rise among assessable stays, n (%)", lambda f: n_pct(f, "lag_aki", "exposure_aki_assessable"))
    add("High lactate among measured stays, n (%)", lambda f: n_pct(f, "lag_lactate_high", "exposure_lactate_available"))
    for value, label in (("cardiac", "ICU: cardiac"), ("medical", "ICU: medical"), ("mixed", "ICU: mixed"), ("neuro", "ICU: neuro"), ("surgical_trauma", "ICU: surgical trauma")):
        add(label, lambda f, v=value: f"{(f.icu_group == v).sum()} ({(f.icu_group == v).mean()*100:.1f})")
    for value, label in (("emergency", "Admission: emergency"), ("observation", "Admission: observation"), ("planned", "Admission: planned"), ("urgent", "Admission: urgent")):
        add(label, lambda f, v=value: f"{(f.admission_group == v).sum()} ({(f.admission_group == v).mean()*100:.1f})")
    for value, label in (("introducer_cordis", "CVC: introducer cordis"), ("multi_lumen", "CVC: multi lumen"), ("other_or_multiple", "CVC: other or multiple")):
        add(label, lambda f, v=value: f"{(f.cvc_type_group == v).sum()} ({(f.cvc_type_group == v).mean()*100:.1f})")
    output = pd.DataFrame(rows)
    output.to_csv(OUT, index=False)
    print(f"Wrote {len(output)} aggregate rows to {OUT}")


if __name__ == "__main__":
    main()

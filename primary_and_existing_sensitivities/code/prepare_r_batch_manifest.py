"""Stage the few missing *local-only* Cox risk frames and batch manifest.

Uses existing derived files; does not read raw D: data, fit models, or export
patient rows outside the secure_derived_inputs directory. The manifest and
code may be shared, but the source risk frames and local-only staged frames
must not enter a public package.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "ARIC_revision_20260920"
ROUTE = BASE / "route_A"
SEC = ROOT / "ARIC_revision_20261009" / "secure_derived_inputs"
STAGE = SEC / "statistics_batch"
DELTA = SEC / "final_score_deltas_stay_day3_exact48_union.csv.gz"
RISK = BASE / "secure_derived_inputs" / "revised_risk_day3.csv.gz"
REF = ROUTE / "secure_derived_inputs" / "bsi_reference_with_components.csv.gz"
EXACT = BASE / "secure_derived_inputs" / "revised_risk_exact48h.csv.gz"

ADJUST = (
    "age_per_10y + female + icu_group + admission_group + multiple_line + "
    "cvc_type_group + exposure_antibiotic_exposure + exposure_lactate_available + exposure_aki_assessable"
)
MAIN = "exposure_vulnerability_category + exposure_indication_present + " + ADJUST
MAIN_NO_AKI_ASSESS = MAIN.replace(" + exposure_aki_assessable", "")
MAIN_BOTH_LABS = MAIN.replace(" + exposure_lactate_available + exposure_aki_assessable", "")
COMPONENTS = (
    "documented_chronic_count + exposure_mechanical_ventilation + "
    "exposure_creatinine_rise_documented + exposure_lactate_high_documented + "
    "exposure_indication_present + " + ADJUST
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def legacy_module():
    src = BASE / "code" / "run_reanalysis.py"
    spec = importlib.util.spec_from_file_location("read_only_run_reanalysis_for_active_risk", src)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    def add(name: str, input_path: Path, rhs: str, transform: str = "none", delta: Path | None = DELTA) -> None:
        assert input_path.exists(), input_path
        manifest.append({"model_name": name, "input_path": str(input_path), "rhs": rhs,
                         "score_delta_path": "" if delta is None else str(delta), "transform": transform})

    add("primary_day3_final_r", RISK, MAIN)
    add("joint_states_final_r", RISK, "exposure_vulnerability_category * exposure_indication_present + " + ADJUST)
    add("record_frequency_indication_final_r", RISK, MAIN, "record_frequency_indication")
    add("without_creatinine_marker_final_r", RISK, MAIN, "without_creatinine_marker")
    add("nonoverlapping_cancer_final_r", RISK, MAIN, "nonoverlapping_cancer")
    add("both_labs_observed_final_r", RISK, MAIN_BOTH_LABS, "both_labs_observed")
    add("matched_48h_adjustment_day3_final_r", RISK, MAIN_NO_AKI_ASSESS)
    add("component_adjusted_final_r", RISK, MAIN + " + exposure_mechanical_ventilation + exposure_vasopressor + exposure_renal_replacement_therapy")
    add("recognized_pathogen_cause_specific_final_r", RISK, MAIN, "recognized_pathogen")
    add("day7_time_interactions_final_r", RISK, MAIN + " + vmod:late7 + vhigh:late7 + exposure_indication_present:late7")
    add("logtime_main_interactions_final_r", RISK, MAIN + " + vmod:log_t + vhigh:log_t + exposure_indication_present:log_t", "logtime_main")
    add("primary_exact48h_final_r", EXACT, MAIN_NO_AKI_ASSESS)
    add("bsi_components_final_r", REF, COMPONENTS)
    for hours in (24, 48):
        additional_path = ROUTE / "secure_derived_inputs" / f"bsi_additional_{hours}h_risk.csv.gz"
        additional = pd.read_csv(additional_path, usecols=["analysis_interval_id", "event"])
        ref = pd.read_csv(REF)
        selected = ref[ref.analysis_interval_id.isin(additional.analysis_interval_id)]
        assert len(selected) == len(additional)
        assert int(selected.event.sum()) == int(additional.event.sum())
        matched = STAGE / f"bsi_matched_reference_{hours}h.csv.gz"
        selected.to_csv(matched, index=False)
        add(f"bsi_lag{hours}_matched_reference_final_r", matched, MAIN_NO_AKI_ASSESS)
        add(f"bsi_lag{hours}_additional_lag_final_r", additional_path, MAIN_NO_AKI_ASSESS)
    # The active-episode-only risk set was not materialized in the old package;
    # rebuild once from its existing local derived daily/landmark/event tables.
    rev = legacy_module()
    daily, landmark, events = rev.load("day3")
    active, _, _ = rev.build_risk(daily, landmark, events, True)
    active_path = STAGE / "active_episode_only_day3.csv.gz"
    active.to_csv(active_path, index=False)
    add("active_episode_only_final_r", active_path, MAIN)
    immune = SEC / "immune_flag_excluded_day3_risk.csv.gz"
    if immune.exists():
        add("immune_excluded_final_r", immune, MAIN, delta=None)
    acute_rhs = (
        "exposure_mechanical_ventilation + exposure_creatinine_rise_documented + "
        "exposure_lactate_high_documented + exposure_indication_present + " + ADJUST
    )
    for name, filename in [
        ("bsi_acute_landmark_only_r", "preobservation_landmark_day3_risk.csv.gz"),
        ("bsi_acute_daily_prior_window_r", "preobservation_daily_prior_window_day3_risk.csv.gz"),
    ]:
        path = SEC / filename
        if path.exists():
            add(name, path, acute_rhs, delta=None)

    output = HERE / "cox_batch_manifest_local.tsv"
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "input_path", "rhs", "score_delta_path", "transform"], delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)
    source_paths = sorted({Path(z["input_path"]) for z in manifest} | {DELTA})
    summary = {"models": len(manifest), "manifest": str(output), "patient_rows_exported_to_public_area": False,
               "local_only_staged_frames": [str(matched) for matched in STAGE.glob("*.csv.gz")],
               "input_hashes": {str(p): sha256(p) for p in source_paths}}
    (HERE / "cox_batch_input_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("STAGED", len(manifest), "models; local-only risk frames in", STAGE)


if __name__ == "__main__":
    main()

"""Rebuild the ARIC day-3 cohort and BSI source tables from raw MIMIC-IV 3.1.

This extraction-only entry point deliberately does not call the legacy
`run_host_vulnerability_day3_landmark.run()` because that entry point also fits
superseded models and writes into historical output directories. Restricted
rows are written only below --output-root/secure_derived_inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts import rebuild_locked_mimic_package_v4 as locked
from scripts import run_host_vulnerability_day3_landmark as legacy
from src.analysis.host_vulnerability_indication import derive_chronic_vulnerability


CVC_ITEM_IDS = (224263, 224267, 224273, 225199)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def secure_write(frame: pd.DataFrame, path: Path, secure: Path) -> None:
    if not path.resolve().is_relative_to(secure.resolve()):
        raise ValueError("Patient-level output outside secure directory")
    frame.to_csv(path, index=False, compression="gzip")


def save_collection_master(blood: pd.DataFrame, positive: pd.DataFrame, cohort: pd.DataFrame, secure: Path) -> dict:
    source = blood.loc[blood.hadm_id.isin(set(cohort.hadm_id))].copy()
    source["has_non_cancelled"] = ~source.org_name.map(locked.norm).isin({"cancelled", "canceled"})
    source["positive"] = source.collection_key.isin(set(positive.collection_key))
    conflict = source.groupby("collection_key", sort=False).agg(
        subjects=("subject_id", "nunique"), admissions=("hadm_id", "nunique"),
        event_times=("event_time", "nunique"), exact_times=("charttime", "nunique"),
    )
    bad = conflict.subjects.gt(1) | conflict.admissions.gt(1) | conflict.exact_times.gt(1) | (conflict.exact_times.eq(0) & conflict.event_times.gt(1))
    if bad.any():
        raise ValueError(f"Ambiguous relevant collection keys: {int(bad.sum())}")
    master = source.groupby("collection_key", sort=False).agg(
        subject_id=("subject_id", "first"), hadm_id=("hadm_id", "first"),
        event_time=("event_time", "first"), charttime=("charttime", "first"),
        positive=("positive", "max"), has_non_cancelled=("has_non_cancelled", "max"),
        raw_rows=("collection_key", "size"),
    ).reset_index()
    master["event_time"] = master.charttime.fillna(master.event_time)
    assert master.collection_key.is_unique
    secure_write(master, secure / "culture_collections_raw_day3.csv.gz", secure)
    return {
        "relevant_blood_result_rows": int(len(source)),
        "relevant_unique_specimens": int(len(master)),
        "relevant_cancelled_only_specimens": int(master.has_non_cancelled.eq(0).sum()),
        "relevant_positive_specimens": int(master.positive.sum()),
        "ambiguous_relevant_collection_keys": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    out = args.output_root.resolve()
    if not (raw / "hosp" / "diagnoses_icd.csv.gz").is_file() or not (raw / "icu" / "procedureevents.csv.gz").is_file():
        raise FileNotFoundError("--raw-root is not a MIMIC-IV hosp/icu root")
    if out.is_relative_to(raw) or out.is_relative_to(PROJECT / "ARIC_revision_20260920"):
        raise ValueError("Output must not be inside raw or historical analysis directory")
    secure, aggregate = out / "secure_derived_inputs", out / "aggregate"
    for directory in (secure, aggregate, aggregate / "qc"):
        directory.mkdir(parents=True, exist_ok=True)
    paths = locked.Paths(base=raw, qc=aggregate / "qc", tables=aggregate, figures=aggregate, manuscript=aggregate)
    started = datetime.now().astimezone().isoformat()
    patients, admissions, icu, d_items = locked.load_core(paths)
    adult_all, adult_first = locked.adult_cohorts(patients, admissions, icu)
    print(f"Core adult ICU stays {len(adult_all):,}; first ICU stays per admission {len(adult_first):,}", flush=True)
    mapping = d_items.loc[d_items.itemid.isin(CVC_ITEM_IDS), ["itemid", "label"]].copy()
    if set(mapping.itemid) != set(CVC_ITEM_IDS):
        raise ValueError("Four frozen CVC item IDs not all in source dictionary")
    mapping["primary_include"] = 1
    mapping["sensitivity_group"] = ""
    mapping.to_csv(aggregate / "cvc_item_dictionary.csv", index=False)
    raw_cvc, valid_cvc, cvc = locked.build_cvc(paths, adult_first, mapping)
    secure_write(valid_cvc, secure / "valid_cvc_records_day3.csv.gz", secure)
    secure_write(cvc, secure / "merged_cvc_episodes_day3.csv.gz", secure)
    print(f"CVC valid rows {len(valid_cvc):,}; merged episodes {len(cvc):,}; stays {cvc.stay_id.nunique():,}", flush=True)
    del raw_cvc

    blood, positive, susceptibility, micro_qc = locked.microbiology_clean(paths, cvc)
    event_flags, qualifying_isolates = legacy.qualify_all_bsi_events(positive, adult_first)
    landmarks, landmark_qc = legacy.make_landmarks(cvc, "day3_calendar_landmark")
    eligible, first_all, first_clabsi, event_qc = legacy.select_scenario_events(
        "day3_calendar_landmark", landmarks, cvc, event_flags, qualifying_isolates, susceptibility, paths
    )
    days = legacy.build_landmark_days("day3_calendar_landmark", eligible, cvc, valid_cvc)
    assert eligible.stay_id.is_unique
    assert days.loc[days.local_day.eq(0)].stay_id.nunique() == len(eligible)
    collections = save_collection_master(blood, positive, eligible, secure)
    for name, frame in [
        ("landmark_cohort_day3", eligible), ("first_all_bsi_day3", first_all),
        ("clabsi_events_day3", first_clabsi), ("day_windows_day3", days),
    ]:
        secure_write(frame, secure / f"{name}.csv.gz", secure)
    del blood, positive, susceptibility, qualifying_isolates
    print(f"Day-3 eligible stays {len(eligible):,}; first line-timed BSI candidates {len(first_clabsi):,}; day windows {len(days):,}", flush=True)

    diagnoses = pd.read_csv(raw / "hosp" / "diagnoses_icd.csv.gz", usecols=["subject_id", "hadm_id", "icd_code", "icd_version"], low_memory=False)
    chronic = derive_chronic_vulnerability(diagnoses)
    chronic = chronic.loc[chronic.hadm_id.isin(set(eligible.hadm_id))]
    secure_write(chronic, secure / "legacy_chronic_flags_day3.csv.gz", secure)
    print(f"Discharge-code flag rows {len(chronic):,}; source diagnosis rows {len(diagnoses):,}", flush=True)

    checkpoint = {
        "stage": "raw_extraction_day3",
        "started_at": started,
        "completed_at": datetime.now().astimezone().isoformat(),
        "raw_version_evidence": (raw / "CHANGELOG.txt").open(encoding="utf-8").readline().strip(),
        "adult_icu_stays": int(len(adult_all)),
        "adult_first_icu_stays_per_admission": int(len(adult_first)),
        "valid_cvc_records": int(len(valid_cvc)),
        "merged_cvc_episodes": int(len(cvc)),
        "landmark_candidates": int(len(landmarks)),
        "eligible_day3_stays": int(len(eligible)),
        "eligible_day3_patients": int(eligible.subject_id.nunique()),
        "first_line_timed_bsi_candidates_before_episode_window": int(len(first_clabsi)),
        "prelandmark_exclusions": int(event_qc.loc[event_qc.metric.eq("stays_with_qualifying_bsi_before_landmark"), "value"].sum()),
        "day_windows_including_baseline": int(len(days)),
        "diagnosis_rows_all_MIMIC": int(len(diagnoses)),
        "diagnosis_coded_eligible_admissions": int(len(chronic)),
        "source_files_sha256": {
            "procedureevents.csv.gz": digest(raw / "icu" / "procedureevents.csv.gz"),
            "microbiologyevents.csv.gz": digest(raw / "hosp" / "microbiologyevents.csv.gz"),
            "diagnoses_icd.csv.gz": digest(raw / "hosp" / "diagnoses_icd.csv.gz"),
        },
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
    } | collections
    for name, frame in [("microbiology_source_qc", micro_qc), ("landmark_qc", landmark_qc), ("event_qc", event_qc)]:
        frame.to_csv(aggregate / f"{name}.csv", index=False)
    (aggregate / "raw_extraction_checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    print(json.dumps({k: checkpoint[k] for k in ["eligible_day3_stays", "eligible_day3_patients", "first_line_timed_bsi_candidates_before_episode_window", "day_windows_including_baseline"]}), flush=True)


if __name__ == "__main__":
    main()

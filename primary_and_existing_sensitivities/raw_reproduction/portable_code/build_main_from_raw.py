"""Build the single corrected day-3 ARIC main risk set from raw-source extraction.

Raw MIMIC files are read-only. Restricted rows are confined to
--output-root/secure_derived_inputs; aggregate outputs contain no IDs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.analysis.host_vulnerability_indication import CHRONIC_CONDITIONS, prepare_host_indication_person_days
from scripts import run_host_vulnerability_day3_landmark as legacy
from src.analysis import host_vulnerability_advanced as adv
from extract_day3_from_raw import digest, secure_write


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    sys.modules[name] = out
    spec.loader.exec_module(out)
    return out


LAB = module("aric_revised_labs", PROJECT / "ARIC_revision_20260920/code/rebuild_labs.py")
TREAT = module("aric_revised_treatments", PROJECT / "ARIC_revision_20260920/code/rebuild_treatments.py")
REANALYSIS = module("aric_reanalysis", PROJECT / "ARIC_revision_20260920/code/run_reanalysis.py")
CODING = module("aric_final_coding", PROJECT / "ARIC_revision_20260920/association_audit_20261003/code/coding_rules_final.py")
SAMPLING = module("aric_sampling", PROJECT / "ARIC_revision_20260920/route_A/code/run_sampling_analysis.py")

ANTIBIOTIC = re.compile(r"vancomycin|piperacillin|tazobactam|cefepime|ceftazidime|ceftriaxone|meropenem|imipenem|ertapenem|doripenem|aztreonam|linezolid|daptomycin", re.I)
CVC_LABELS = {224263: "multi_lumen", 224267: "cordis_introducer", 224273: "presep_catheter", 225199: "triple_introducer"}


def temporal_mark(days: pd.DataFrame, events: pd.DataFrame, group_col: str) -> pd.Series:
    """Exact [start,end) interval overlap, grouped to avoid quadratic filtering."""
    result = pd.Series(0, index=days.index, dtype="int8")
    by_group = {key: g.sort_values("starttime") for key, g in events.groupby(group_col, sort=False)}
    for key, indices in days.groupby(group_col, sort=False).groups.items():
        e = by_group.get(key)
        if e is None or e.empty:
            continue
        starts = e.starttime.astype("int64").to_numpy()
        max_ends = np.maximum.accumulate(e.endtime.astype("int64").to_numpy())
        lo = days.loc[indices, "day_start"].astype("int64").to_numpy()
        hi = days.loc[indices, "day_end"].astype("int64").to_numpy()
        pos = np.searchsorted(starts, hi, side="left") - 1
        hit = pos >= 0
        hit[hit] = max_ends[pos[hit]] > lo[hit]
        result.loc[indices] = hit.astype("int8")
    return result


def laboratory(days: pd.DataFrame, raw: Path, secure: Path, aggregate: Path) -> dict:
    LAB.self_test()
    windows = days.groupby("hadm_id").agg(window_start=("day_start", "min"), window_end=("day_end", "max"))
    windows["window_start"] -= pd.Timedelta(hours=48)
    items = pd.read_csv(raw / "hosp/d_labitems.csv.gz")
    items = items.loc[items.label.isin(["Creatinine", "Lactate"]) & items.fluid.eq("Blood"), ["itemid", "label", "fluid"]].copy()
    items["marker"] = items.label.str.lower()
    items.to_csv(aggregate / "lab_item_dictionary.csv", index=False)
    item_map = dict(zip(items.itemid, items.marker))
    retained = []
    unit_qc = []
    scanned = 0
    columns = ["hadm_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"]
    for number, chunk in enumerate(pd.read_csv(raw / "hosp/labevents.csv.gz", usecols=columns, chunksize=1_000_000, low_memory=False), 1):
        scanned += len(chunk)
        chunk = chunk.loc[chunk.hadm_id.isin(windows.index) & chunk.itemid.isin(item_map)].copy()
        if len(chunk):
            chunk["charttime"] = pd.to_datetime(chunk.charttime, errors="coerce")
            chunk["storetime"] = pd.to_datetime(chunk.storetime, errors="coerce")
            chunk = chunk.join(windows, on="hadm_id")
            chunk = chunk.loc[(chunk.charttime >= chunk.window_start) & (chunk.charttime < chunk.window_end)].copy()
            chunk["marker"] = chunk.itemid.map(item_map)
            chunk["valuenum"] = pd.to_numeric(chunk.valuenum, errors="coerce")
            qc = chunk.groupby(["marker", "valueuom"], dropna=False).size()
            unit_qc.extend({"marker": key[0], "valueuom": str(key[1]), "rows": int(n)} for key, n in qc.items())
            units = chunk.valueuom.fillna("").str.lower().str.replace(" ", "", regex=False)
            valid = (chunk.marker.eq("creatinine") & units.eq("mg/dl")) | (chunk.marker.eq("lactate") & units.eq("mmol/l"))
            chunk = chunk.loc[valid & chunk.valuenum.gt(0) & chunk.charttime.notna() & chunk.storetime.notna()].copy()
            chunk["available_time"] = chunk[["charttime", "storetime"]].max(axis=1)
            retained.append(chunk.drop(columns=["window_start", "window_end"]))
        if number % 10 == 0:
            print(f"Lab scan: {scanned:,} rows; {sum(map(len,retained)):,} retained", flush=True)
    labs = pd.concat(retained, ignore_index=True).sort_values(["hadm_id", "marker", "charttime"])
    secure_write(labs, secure / "blood_laboratory_observations_day3.csv.gz", secure)
    if unit_qc:
        pd.DataFrame(unit_qc).groupby(["marker", "valueuom"], as_index=False).rows.sum().to_csv(aggregate / "lab_source_units.csv", index=False)
    groups = {}
    for hadm, group in labs.groupby("hadm_id", sort=False):
        groups[int(hadm)] = {
            marker: (g.charttime.to_numpy(dtype="datetime64[ns]").astype("int64"),
                     g.available_time.to_numpy(dtype="datetime64[ns]").astype("int64"), g.valuenum.to_numpy())
            for marker, g in group.sort_values("charttime").groupby("marker")
        }
    patients = pd.read_csv(raw / "hosp/patients.csv.gz", usecols=["subject_id", "anchor_age", "anchor_year"])
    admissions = pd.read_csv(raw / "hosp/admissions.csv.gz", usecols=["subject_id", "hadm_id", "admittime"], parse_dates=["admittime"])
    age = admissions.loc[admissions.hadm_id.isin(windows.index)].merge(patients, on="subject_id", how="left")
    age["age"] = age.anchor_age + age.admittime.dt.year - age.anchor_year
    age_map = age.set_index("hadm_id").age
    rows = []
    for row in days.itertuples(index=False):
        value = LAB.array_markers(groups.get(int(row.hadm_id), {}), row.day_start, row.day_end)
        value.update(stay_id=int(row.stay_id), local_day=int(row.local_day), age=float(age_map.get(row.hadm_id, np.nan)))
        rows.append(value)
    feature = pd.DataFrame(rows)
    feature["aki_assessable"] = feature.creatinine_pair_available
    assert not feature.duplicated(["stay_id", "local_day"]).any()
    secure_write(feature, secure / "corrected_features_day3.csv.gz", secure)
    summary = {"raw_labevents_rows_scanned": scanned, "permitted_laboratory_rows": int(len(labs)), "day_features": int(len(feature)),
               "lactate_measured_days": int(feature.lactate_available.sum()), "creatinine_pair_assessable_days": int(feature.aki_assessable.sum()),
               "aki_positive_days": int(feature.aki.eq(1).sum()), "lactate_high_days": int(feature.lactate_high.eq(1).sum())}
    print(f"Laboratory complete: {summary['day_features']:,} day windows", flush=True)
    return summary


def treatments(days: pd.DataFrame, raw: Path, secure: Path, aggregate: Path) -> dict:
    stays = set(days.stay_id)
    ids = set(TREAT.MAPPING["vasopressor"] + TREAT.MAPPING["tpn"])
    inputs = []
    scanned_input = 0
    for chunk in pd.read_csv(raw / "icu/inputevents.csv.gz", usecols=["stay_id", "itemid", "starttime", "endtime", "statusdescription"], chunksize=500_000, low_memory=False):
        scanned_input += len(chunk)
        c = chunk.loc[chunk.stay_id.isin(stays) & chunk.itemid.isin(ids) & ~chunk.statusdescription.eq("Rewritten")].copy()
        if len(c):
            inputs.append(c)
    inp = pd.concat(inputs, ignore_index=True)
    proc = pd.read_csv(raw / "icu/procedureevents.csv.gz", usecols=["stay_id", "itemid", "starttime", "endtime", "statusdescription"], low_memory=False)
    scanned_proc = len(proc)
    proc = proc.loc[proc.stay_id.isin(stays) & ~proc.statusdescription.eq("Rewritten")].copy()
    for frame in (inp, proc):
        frame["starttime"] = pd.to_datetime(frame.starttime, errors="coerce")
        frame["endtime"] = pd.to_datetime(frame.endtime, errors="coerce")
        frame.dropna(subset=["starttime", "endtime"], inplace=True)
    output = days[["stay_id", "local_day"]].copy()
    for variable, v_ids in TREAT.MAPPING.items():
        src = inp if variable in ["vasopressor", "tpn"] else proc
        events = src.loc[src.itemid.isin(v_ids) & src.endtime.gt(src.starttime)].drop_duplicates(["stay_id", "itemid", "starttime", "endtime"])
        output[variable] = temporal_mark(days, events, "stay_id")
    secure_write(output, secure / "corrected_treatments_day3.csv.gz", secure)
    d_items = pd.read_csv(raw / "icu/d_items.csv.gz", usecols=["itemid", "label", "linksto"])
    mapping = pd.DataFrame([{"variable": k, "itemid": i} for k, vs in TREAT.MAPPING.items() for i in vs]).merge(d_items, on="itemid", how="left")
    mapping.to_csv(aggregate / "treatment_item_dictionary.csv", index=False)
    summary = {"inputevents_rows_scanned": scanned_input, "procedureevents_rows_scanned_second_pass": scanned_proc, "day_treatment_features": int(len(output))}
    summary.update({f"{col}_positive_days": int(output[col].sum()) for col in TREAT.MAPPING})
    print("Treatment intervals complete", flush=True)
    return summary


def antibiotics(days: pd.DataFrame, raw: Path, secure: Path) -> dict:
    hadms = set(days.hadm_id)
    chunks = []
    scanned = 0
    for part in pd.read_csv(raw / "hosp/prescriptions.csv.gz", usecols=["hadm_id", "starttime", "stoptime", "drug"], chunksize=500_000, low_memory=False):
        scanned += len(part)
        match = part.hadm_id.isin(hadms) & part.drug.fillna("").str.contains(ANTIBIOTIC, na=False)
        chunk = part.loc[match, ["hadm_id", "starttime", "stoptime"]].copy()
        if len(chunk):
            chunks.append(chunk)
    rx = pd.concat(chunks, ignore_index=True)
    rx["starttime"] = pd.to_datetime(rx.starttime, errors="coerce")
    rx["stoptime"] = pd.to_datetime(rx.stoptime, errors="coerce")
    rx["stoptime"] = rx.stoptime.fillna(rx.starttime + pd.Timedelta(days=1))
    rx = rx.dropna(subset=["starttime", "stoptime"])
    rx = rx.loc[rx.stoptime.gt(rx.starttime)].rename(columns={"stoptime": "endtime"})
    feature = days[["stay_id", "local_day"]].copy()
    feature["antibiotic_exposure"] = temporal_mark(days, rx, "hadm_id")
    secure_write(feature, secure / "antibiotic_exposure_day3.csv.gz", secure)
    print("Prescription intervals complete", flush=True)
    return {"prescriptions_rows_scanned": scanned, "relevant_broad_antibiotic_prescriptions": int(len(rx)),
            "antibiotic_positive_day_windows": int(feature.antibiotic_exposure.sum())}


def cvc_type(days: pd.DataFrame, valid_cvc: pd.DataFrame) -> pd.Series:
    out = pd.Series("unknown", index=days.index, dtype="object")
    by_stay = {key: g for key, g in valid_cvc.groupby("stay_id", sort=False)}
    for stay, indices in days.groupby("stay_id", sort=False).groups.items():
        e = by_stay.get(stay)
        if e is None:
            continue
        for idx in indices:
            row = days.loc[idx]
            active = e.loc[e.starttime.lt(row.day_start + pd.Timedelta(days=1)) & e.endtime.gt(row.day_start)]
            broad = set()
            for item in active.itemid:
                label = CVC_LABELS.get(int(item))
                if label == "multi_lumen": broad.add("multi_lumen")
                elif label in ("cordis_introducer", "triple_introducer"): broad.add("introducer_cordis")
                elif label == "presep_catheter": broad.add("presep_catheter")
            if not broad: group = "unknown"
            elif len(broad) > 1 or "presep_catheter" in broad: group = "other_or_multiple"
            else: group = next(iter(broad))
            out.at[idx] = group
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    root = args.output_root.resolve()
    secure, aggregate = root / "secure_derived_inputs", root / "aggregate"
    if not (aggregate / "raw_extraction_checkpoint.json").exists():
        raise RuntimeError("Run extract_day3_from_raw.py first")
    if root.is_relative_to(raw) or root.is_relative_to(PROJECT / "ARIC_revision_20260920"):
        raise ValueError("Output cannot be inside raw or historical directory")
    started = datetime.now().astimezone().isoformat()
    days = pd.read_csv(secure / "day_windows_day3.csv.gz", parse_dates=["day_start", "day_end", "landmark_time", "followup_end"])
    print(f"Raw-derived day windows: {len(days):,}", flush=True)
    lab_qc = laboratory(days, raw, secure, aggregate)
    treatment_qc = treatments(days, raw, secure, aggregate)
    antibiotic_qc = antibiotics(days, raw, secure)
    valid_cvc = pd.read_csv(secure / "valid_cvc_records_day3.csv.gz", parse_dates=["starttime", "endtime"])
    chronic = pd.read_csv(secure / "legacy_chronic_flags_day3.csv.gz")
    features = pd.read_csv(secure / "corrected_features_day3.csv.gz")
    treatment = pd.read_csv(secure / "corrected_treatments_day3.csv.gz")
    antibiotic = pd.read_csv(secure / "antibiotic_exposure_day3.csv.gz")
    state = days.merge(features, on=["stay_id", "local_day"], validate="one_to_one", suffixes=("", "_revised"))
    if "age_revised" in state:
        state["age"] = state.age_revised
        state = state.drop(columns="age_revised")
    state = state.merge(treatment, on=["stay_id", "local_day"], validate="one_to_one")
    state = state.merge(antibiotic, on=["stay_id", "local_day"], validate="one_to_one")
    state["high_frequency_iv_meds"] = 0
    state["recent_positive_culture"] = 0
    state["sofa_proxy"] = 0
    state["cvc_type_group"] = cvc_type(state, valid_cvc)
    state = prepare_host_indication_person_days(state, chronic)
    first_all = pd.read_csv(secure / "first_all_bsi_day3.csv.gz", parse_dates=["event_time"])
    first_clabsi = pd.read_csv(secure / "clabsi_events_day3.csv.gz", parse_dates=["event_time"])
    state, outcome_qc = legacy.add_outcomes(state, first_all, first_clabsi)
    secure_write(state, secure / "patient_day_states_day3.csv.gz", secure)
    outcome_qc.to_csv(aggregate / "outcome_assignment_qc.csv", index=False)
    print(f"Patient-day states: {len(state):,}; generating continuous risk intervals", flush=True)
    REANALYSIS.ORIGINAL = secure
    REANALYSIS.SEC = secure
    REANALYSIS.OUT = aggregate
    d, lm, ev = REANALYSIS.load("day3")
    risk, end, zero = REANALYSIS.build_risk(d, lm, ev)
    secure_write(risk, secure / "risk_before_icd_correction_day3.csv.gz", secure)
    print(f"Risk before ICD correction: {len(risk):,} intervals, {int(risk.event.sum()):,} events", flush=True)
    CODING.RAW_DIAGNOSES = raw / "hosp/diagnoses_icd.csv.gz"
    stays = risk[["subject_id", "hadm_id", "stay_id", "cancer", "metastatic_cancer"]].drop_duplicates("stay_id")
    delta = CODING.stay_flag_deltas(stays)
    secure_write(delta, secure / "coding_delta_day3.csv.gz", secure)
    risk = CODING.correct_frame(risk, delta)
    if "chronic_vulnerability_score" in risk:
        risk["chronic_vulnerability_score"] = risk[CHRONIC_CONDITIONS].sum(axis=1).astype(int)
    if "vulnerability_burden_score" in risk:
        risk["vulnerability_burden_score"] = risk.chronic_vulnerability_score + risk["acute_vulnerability_score"].astype(int)
    if "vulnerability_burden_score_full_sensitivity" in risk:
        risk["vulnerability_burden_score_full_sensitivity"] = risk.chronic_vulnerability_score + risk["acute_vulnerability_score_full_sensitivity"].astype(int)
    assert risk.corrected_chronic_score.eq(risk[CHRONIC_CONDITIONS].sum(axis=1)).all()
    secure_write(risk, secure / "risk_final_day3.csv.gz", secure)
    validated = SAMPLING.validate_risk(risk)
    collections = pd.read_csv(secure / "culture_collections_raw_day3.csv.gz", parse_dates=["event_time", "charttime"])
    assigned = SAMPLING.assign_collections(risk, collections.loc[collections.has_non_cancelled.eq(1)].copy())
    sampling, occasions = SAMPLING.add_outcomes(risk, assigned)
    secure_write(sampling, secure / "sampling_intervals_final_day3.csv.gz", secure)
    aggregate_counts = {
        **validated, "all_relevant_specimens": int(len(collections)),
        "cancelled_only_relevant_specimens": int(collections.has_non_cancelled.eq(0).sum()),
        "noncancelled_specimens_in_risk": int(len(assigned)), "sampling_occasions_in_risk": int(len(occasions)),
        "zero_time_events_adjusted_to_one_second": zero,
        "coded_stays_with_cancer_change": int(delta.delta_cancer.ne(0).sum()),
        "coded_stays_with_metastatic_change": int(delta.delta_meta.ne(0).sum()),
        "coded_stays_with_score_change": int(delta.delta_score.ne(0).sum()),
        "old_report_reference": {"intervals": 60272, "stays": 13863, "patients": 12757, "events": 268,
                                 "risk_person_days": 56370.34172453703, "noncancelled_specimens_in_risk": 19160},
        "lab_qc": lab_qc, "treatment_qc": treatment_qc, "antibiotic_qc": antibiotic_qc,
        "started_at": started, "completed_at": datetime.now().astimezone().isoformat(),
        "software": {"python": platform.python_version(), "pandas": pd.__version__},
        "input_hashes": {"labevents.csv.gz": digest(raw / "hosp/labevents.csv.gz"),
                         "inputevents.csv.gz": digest(raw / "icu/inputevents.csv.gz"),
                         "prescriptions.csv.gz": digest(raw / "hosp/prescriptions.csv.gz"),
                         "extract_day3_from_raw.py": digest(PROJECT / "extract_day3_from_raw.py"),
                         "build_main_from_raw.py": digest(Path(__file__))},
    }
    (aggregate / "main_risk_checkpoint.json").write_text(json.dumps(aggregate_counts, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: aggregate_counts[k] for k in ["intervals", "stays", "patients", "events", "risk_person_days", "noncancelled_specimens_in_risk"]}), flush=True)


if __name__ == "__main__":
    main()

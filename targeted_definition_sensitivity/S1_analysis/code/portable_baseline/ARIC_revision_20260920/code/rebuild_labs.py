"""Rebuild temporally valid blood laboratory markers; raw MIMIC files stay read-only.

Run with the revision project's Python environment. Outputs containing identifiers
are restricted to this revision's secure_derived_inputs directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "secure_derived_inputs" / "host_vulnerability_indication_day3_landmark"
SECURE = ROOT / "secure_derived_inputs"
AUDIT = ROOT / "audit_internal"
SCENARIOS = ("day3", "exact48h")
H48_NS = int(pd.Timedelta(hours=48).value)


def valid_day_markers(labs: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Retain missingness; use only values known strictly before covariate-day end.

    A creatinine rise compares each current measurement with strictly earlier
    blood measurements in the preceding 48 hours. This is a creatinine-rise
    marker, not a complete KDIGO AKI diagnosis (no urine-output criterion).
    """
    if labs.empty:
        return {"aki": np.nan, "lactate_high": np.nan, "lactate_available": 0,
                "creatinine_available": 0, "creatinine_pair_available": 0}
    known = labs.loc[labs["available_time"] < end]
    current = known.loc[(known["charttime"] >= start) & (known["charttime"] < end)]
    lactate = current.loc[current["marker"].eq("lactate"), "valuenum"]
    creatinine = current.loc[current["marker"].eq("creatinine")]
    previous = known.loc[known["marker"].eq("creatinine")]
    rise_results = []
    for row in creatinine.itertuples(index=False):
        baseline = previous.loc[(previous["charttime"] < row.charttime)
                                & (previous["charttime"] >= row.charttime - pd.Timedelta(hours=48)), "valuenum"]
        if len(baseline):
            rise_results.append(bool(row.valuenum >= baseline.min() + 0.3 - 1e-12))
    return {"aki": float(any(rise_results)) if rise_results else np.nan,
            "lactate_high": float(lactate.max() >= 2) if len(lactate) else np.nan,
            "lactate_available": int(len(lactate) > 0),
            "creatinine_available": int(len(creatinine) > 0),
            "creatinine_pair_available": int(len(rise_results) > 0)}


def array_markers(arrays: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Equivalent indexed implementation for the real daily data."""
    result = {"aki": np.nan, "lactate_high": np.nan, "lactate_available": 0,
              "creatinine_available": 0, "creatinine_pair_available": 0}
    start_ns, end_ns = start.value, end.value
    for marker in ("lactate", "creatinine"):
        times, available, values = arrays.get(marker, (np.array([], dtype=np.int64),) * 3)
        left, right = np.searchsorted(times, [start_ns, end_ns], side="left")
        indices = np.arange(left, right)
        indices = indices[available[indices] < end_ns]
        if not len(indices):
            continue
        if marker == "lactate":
            result["lactate_available"] = 1
            result["lactate_high"] = float(values[indices].max() >= 2)
            continue
        result["creatinine_available"] = 1
        rises = []
        for index in indices:
            base_left, base_right = np.searchsorted(times, [times[index] - H48_NS, times[index]], side="left")
            base = np.arange(base_left, base_right)
            base = base[available[base] < end_ns]
            if len(base):
                rises.append(values[index] >= values[base].min() + 0.3 - 1e-12)
        if rises:
            result["creatinine_pair_available"] = 1
            result["aki"] = float(any(rises))
    return result


def self_test() -> None:
    # A future lower creatinine must not turn a preceding measurement positive.
    times = pd.to_datetime(["2020-01-01 08:00", "2020-01-02 08:00", "2020-01-03 08:00"])
    labs = pd.DataFrame({"charttime": times, "available_time": times,
                         "marker": "creatinine", "valuenum": [1.4, 1.3, 0.8]})
    day = valid_day_markers(labs, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))
    assert day["aki"] == 0
    labs.loc[1, "valuenum"] = 1.7
    assert valid_day_markers(labs, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))["aki"] == 1
    labs.loc[1, "available_time"] = pd.Timestamp("2020-01-03 12:00")
    assert np.isnan(valid_day_markers(labs, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))["aki"])
    assert np.isnan(valid_day_markers(labs.iloc[:0], pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))["lactate_high"])
    # Identical charttimes cannot provide one another's strictly previous baseline.
    same_time = labs.iloc[:2].copy()
    same_time["charttime"] = pd.Timestamp("2020-01-02 08:00")
    same_time["available_time"] = same_time["charttime"]
    assert np.isnan(valid_day_markers(same_time, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))["aki"])
    for frame in (labs, same_time, labs.iloc[:0]):
        arrays = {"creatinine": (frame.charttime.to_numpy(dtype="datetime64[ns]").astype("int64"), frame.available_time.to_numpy(dtype="datetime64[ns]").astype("int64"), frame.valuenum.to_numpy())}
        expected = valid_day_markers(frame, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))
        actual = array_markers(arrays, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))
        for key in expected:
            assert actual[key] == expected[key] or (pd.isna(actual[key]) and pd.isna(expected[key]))
    print("Laboratory temporal/missingness tests passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, default=Path((__import__("os").environ.get("MIMIC_ROOT", "MIMIC_ROOT") + '')))
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    SECURE.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    states = {scenario: pd.read_csv(SOURCE / f"patient_day_states_{scenario}.csv.gz",
                                   parse_dates=["day_start", "day_end"]) for scenario in SCENARIOS}
    combined = pd.concat([d[["hadm_id", "day_start", "day_end"]] for d in states.values()])
    windows = combined.groupby("hadm_id").agg(window_start=("day_start", "min"), window_end=("day_end", "max"))
    windows["window_start"] -= pd.Timedelta(hours=48)
    items = pd.read_csv(args.mimic_root / "hosp" / "d_labitems.csv.gz")
    items = items.loc[items["label"].isin(["Creatinine", "Lactate"]) & items["fluid"].eq("Blood")].copy()
    items["marker"] = items["label"].str.lower()
    items.to_csv(AUDIT / "corrected_lab_item_dictionary.csv", index=False)
    item_map = dict(zip(items.itemid, items.marker))
    cache = SECURE / "blood_laboratory_observations.csv.gz"
    unit_rows, scan_rows, retained_rows = [], 0, 0
    if args.use_cache and cache.exists():
        labs = pd.read_csv(cache, parse_dates=["charttime", "storetime", "available_time"])
    else:
        retained = []
        columns = ["hadm_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"]
        for number, chunk in enumerate(pd.read_csv(args.mimic_root / "hosp" / "labevents.csv.gz", usecols=columns,
                                                    chunksize=1_000_000, low_memory=False), 1):
            scan_rows += len(chunk)
            chunk = chunk.loc[chunk.hadm_id.isin(windows.index) & chunk.itemid.isin(item_map)].copy()
            if not chunk.empty:
                chunk["charttime"] = pd.to_datetime(chunk.charttime, errors="coerce")
                chunk["storetime"] = pd.to_datetime(chunk.storetime, errors="coerce")
                chunk = chunk.join(windows, on="hadm_id")
                chunk = chunk.loc[(chunk.charttime >= chunk.window_start) & (chunk.charttime < chunk.window_end)].copy()
                chunk["marker"] = chunk.itemid.map(item_map)
                chunk["valuenum"] = pd.to_numeric(chunk.valuenum, errors="coerce")
                for key, group in chunk.groupby(["marker", "valueuom"], dropna=False):
                    unit_rows.append({"marker": key[0], "valueuom": str(key[1]), "rows": len(group)})
                normalized_units = chunk.valueuom.fillna("").str.lower().str.replace(" ", "", regex=False)
                valid_units = (chunk.marker.eq("creatinine") & normalized_units.eq("mg/dl")) | (chunk.marker.eq("lactate") & normalized_units.eq("mmol/l"))
                chunk = chunk.loc[valid_units & chunk.valuenum.notna() & chunk.valuenum.gt(0)
                                  & chunk.charttime.notna() & chunk.storetime.notna()].copy()
                chunk["available_time"] = chunk[["charttime", "storetime"]].max(axis=1)
                chunk = chunk.drop(columns=["window_start", "window_end"])
                retained_rows += len(chunk)
                retained.append(chunk)
            if number % 10 == 0:
                print(f"Scanned {scan_rows:,} lab records; retained {retained_rows:,} permitted-window blood records", flush=True)
        labs = pd.concat(retained, ignore_index=True).sort_values(["hadm_id", "marker", "charttime"])
        labs.to_csv(cache, index=False, compression="gzip")
        if unit_rows:
            pd.DataFrame(unit_rows).groupby(["marker", "valueuom"], as_index=False)["rows"].sum().to_csv(AUDIT / "corrected_lab_source_units.csv", index=False)
    groups = {}
    for hadm, group in labs.groupby("hadm_id", sort=False):
        groups[int(hadm)] = {marker: (g.charttime.to_numpy(dtype="datetime64[ns]").astype("int64"), g.available_time.to_numpy(dtype="datetime64[ns]").astype("int64"), g.valuenum.to_numpy())
                             for marker, g in group.sort_values("charttime").groupby("marker")}
    patients = pd.read_csv(args.mimic_root / "hosp" / "patients.csv.gz", usecols=["subject_id", "anchor_age", "anchor_year"])
    admissions = pd.read_csv(args.mimic_root / "hosp" / "admissions.csv.gz", usecols=["subject_id", "hadm_id", "admittime"], parse_dates=["admittime"])
    ages = admissions.loc[admissions.hadm_id.isin(windows.index)].merge(patients, on="subject_id", how="left")
    ages["age"] = ages.anchor_age + ages.admittime.dt.year - ages.anchor_year
    age_map = ages.set_index("hadm_id").age
    qc = []
    for scenario, frame in states.items():
        rows = []
        for row in frame.itertuples(index=False):
            result = array_markers(groups.get(int(row.hadm_id), {}), row.day_start, row.day_end)
            result.update(stay_id=int(row.stay_id), local_day=int(row.local_day), age=float(age_map.get(row.hadm_id, np.nan)))
            rows.append(result)
        corrected = pd.DataFrame(rows)
        corrected["aki_assessable"] = corrected["creatinine_pair_available"]
        assert not corrected.duplicated(["stay_id", "local_day"]).any()
        corrected.to_csv(SECURE / f"corrected_features_{scenario}.csv.gz", index=False, compression="gzip")
        for field in ["aki", "lactate_high", "lactate_available", "creatinine_available", "creatinine_pair_available", "age"]:
            qc.append({"scenario": scenario, "field": field, "rows": len(corrected),
                       "nonmissing": int(corrected[field].notna().sum()), "missing": int(corrected[field].isna().sum()),
                       "positive": int(corrected[field].eq(1).sum()) if field != "age" else np.nan,
                       "old_new_disagreement": int((corrected[field].fillna(-999).to_numpy() != frame[field].fillna(-999).to_numpy()).sum()) if field in frame else np.nan})
        print(f"Completed {scenario}: {len(corrected):,} day rows; no patient records printed", flush=True)
    pd.DataFrame(qc).to_csv(AUDIT / "corrected_features_qc.csv", index=False)
    manifest = {"definition": "Blood Lactate >=2 mmol/L; strictly prior-48h Creatinine rise >=0.3 mg/dL; missingness retained",
                "time_rule": "Daily [day_start,day_end); max(charttime,storetime) < day_end; features must be lagged for next-risk-interval models",
                "age": "anchor_age + year(admittime) - anchor_year; oldest deidentified age group remains approximate",
                "source_labevents_bytes": (args.mimic_root / "hosp" / "labevents.csv.gz").stat().st_size,
                "scanned_rows_this_run": scan_rows, "retained_laboratory_rows": len(labs), "itemids": {str(k): v for k,v in item_map.items()},
                "not_a_validated_kdigo_algorithm": True}
    (AUDIT / "corrected_features_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

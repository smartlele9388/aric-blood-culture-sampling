"""Derive stay-level first competing exits from raw-built risk and landmark stages.

This is a stage-only transformation: it does not read D: raw files or fit models.
Patient-level output remains inside secure_derived_inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    secure, aggregate = root / "secure_derived_inputs", root / "aggregate"
    risk = pd.read_csv(secure / "risk_final_day3.csv.gz",
                       parse_dates=["day_start", "stop_datetime", "observed_end", "risk_end", "index_interval_end"])
    landmark = pd.read_csv(secure / "landmark_cohort_day3.csv.gz",
                           parse_dates=["landmark_time", "deathtime", "outtime", "dischtime"])
    first = risk.sort_values(["stay_id", "day_start"]).drop_duplicates("stay_id")
    last = risk.sort_values(["stay_id", "day_start"]).drop_duplicates("stay_id", keep="last")
    assert len(first) == len(landmark) and first.stay_id.is_unique
    assert (last.set_index("stay_id").stop_datetime == first.set_index("stay_id").observed_end).all()
    columns = ["stay_id", "observed_end", "risk_end", "event_eligible", "index_interval_end", "exposure_vulnerability_category"]
    t = landmark[["subject_id", "stay_id", "landmark_time", "deathtime", "outtime", "dischtime"]].merge(
        first[columns], on="stay_id", validate="one_to_one")
    t["line_window_end"] = (t.index_interval_end - pd.Timedelta(nanoseconds=1)).dt.floor("D") + pd.Timedelta(days=2)
    t["time_days"] = (t.observed_end - t.landmark_time).dt.total_seconds() / 86400
    t["cause"] = 0
    # Frozen precedence: BSI > death > ICU/hospital discharge > attribution end.
    t.loc[t.risk_end.eq(t.line_window_end), "cause"] = 4
    t.loc[t.risk_end.eq(t.outtime) | t.risk_end.eq(t.dischtime), "cause"] = 3
    t.loc[t.risk_end.eq(t.deathtime), "cause"] = 2
    t.loc[t.event_eligible, "cause"] = 1
    t = t.rename(columns={"exposure_vulnerability_category": "baseline_group"})
    assert t.time_days.ge(0).all() and t.time_days.le(28 + 1e-10).all()
    assert t.cause.eq(1).sum() == risk.event.sum()
    t.to_csv(secure / "competing_events_day3.csv.gz", index=False, compression="gzip")
    cause_counts = t.groupby("cause", as_index=False).size().rename(columns={"size": "stays"})
    cause_counts["percent"] = cause_counts.stays / len(t) * 100
    cause_counts.to_csv(aggregate / "competing_exit_counts_day3.csv", index=False)
    public_counts = {str(row.cause): ("1-10" if 0 < row.stays < 11 else int(row.stays)) for row in cause_counts.itertuples()}
    detail = {"source": "raw-built risk_final_day3 and landmark_cohort_day3 only; no new raw scan",
              "stays": int(len(t)), "BSI_events": int(t.cause.eq(1).sum()),
              "cause_codes": {"0": "administrative", "1": "first study-defined detected BSI",
                              "2": "death", "3": "ICU or hospital discharge", "4": "line-attribution window end"},
              "tie_priority": "BSI > death > ICU/hospital discharge > line-attribution window end",
              "causes_with_small_nonzero_counts_suppressed": public_counts}
    (aggregate / "competing_exit_definition.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    print(json.dumps({"stays": detail["stays"], "BSI_events": detail["BSI_events"], "cause_counts": public_counts}), flush=True)


if __name__ == "__main__":
    main()

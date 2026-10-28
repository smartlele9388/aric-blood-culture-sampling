"""Validate 268 selected event timestamps and export organism aggregates only."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path((__import__("os").environ.get("ARIC_PROJECT_ROOT", "ARIC_PROJECT_ROOT") + ''))
SEC = ROOT / "ARIC_revision_20260920" / "secure_derived_inputs"
EVENTS = ROOT / "secure_derived_inputs" / "host_vulnerability_indication_day3_landmark" / "clabsi_events_day3.csv.gz"
HIST = ROOT / "ARIC_revision_20260920" / "results" / "bsi_organism_distribution.csv"
OUT = ROOT / "ARIC_revision_20261009" / "documents"


def clean(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower()).strip()


def main() -> None:
    risk = pd.read_csv(SEC / "revised_risk_day3.csv.gz", usecols=["stay_id", "event", "event_time"], parse_dates=["event_time"])
    selected = risk.loc[risk.event.eq(1), ["stay_id", "event_time"]].copy()
    if len(selected) != 268 or selected.stay_id.duplicated().any():
        raise AssertionError("Expected 268 unique first BSI stays")
    audit = pd.read_csv(SEC / "event_audit_day3.csv.gz", usecols=["stay_id", "event_time", "event_eligible"], parse_dates=["event_time"])
    a = selected.merge(audit, on="stay_id", how="left", validate="one_to_one", suffixes=("_risk", "_audit"))
    if len(a) != 268 or not a.event_eligible.all() or not a.event_time_risk.eq(a.event_time_audit).all():
        raise AssertionError("Selected event and eligibility audit do not match")
    source = pd.read_csv(EVENTS, usecols=["stay_id", "event_time", "org_name"], parse_dates=["event_time"])
    linked = selected.merge(source, on="stay_id", how="left", validate="one_to_one", suffixes=("_risk", "_source"))
    if len(linked) != 268 or linked.org_name.isna().any() or not linked.event_time_risk.eq(linked.event_time_source).all():
        raise AssertionError("Selected event time does not match organism-bearing source event")
    rows = []
    for i, names in enumerate(linked.org_name):
        for name in set(clean(x) for x in names.split("; ")):
            if name:
                rows.append((i, name))
    pairs = pd.DataFrame(rows, columns=["local_event_index", "organism"])
    totals = pairs.groupby("organism").local_event_index.nunique().sort_values(ascending=False)
    historic = pd.read_csv(HIST).groupby("organism").events.sum().sort_index()
    pd.testing.assert_series_equal(totals.sort_index(), historic, check_names=False)
    top = totals.head(10)
    public = pd.DataFrame({"Organism label": list(top.index) + ["Other organism labels"], "Events with this organism": list(map(int, top.values)) + [int(totals.iloc[10:].sum())]})
    public.to_csv(OUT / "pathogen_event_summary.csv", index=False)
    manifest = {
        "selected_first_BSI_events": 268,
        "eligible_audit_events_matched_at_identical_timestamp": 268,
        "organism_source_events_matched_at_identical_timestamp": 268,
        "event_organism_pairs": len(pairs),
        "unique_organism_labels": len(totals),
        "events_with_multiple_distinct_organisms": int(pairs.groupby("local_event_index").size().gt(1).sum()),
        "aggregate_matches_prior_source_audit": True,
        "counting_rule": "One event counted once per distinct normalized organism; sums may exceed 268 because of polymicrobial events. Labels follow source EHR; no clinical adjudication.",
        "source_sha256": {
            "risk_day3": hashlib.sha256((SEC / "revised_risk_day3.csv.gz").read_bytes()).hexdigest(),
            "event_audit_day3": hashlib.sha256((SEC / "event_audit_day3.csv.gz").read_bytes()).hexdigest(),
            "organism_source_events": hashlib.sha256(EVENTS.read_bytes()).hexdigest(),
        },
        "patient_rows_exported": False,
    }
    (OUT / "pathogen_event_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(public.to_string(index=False))
    print(json.dumps({k:v for k,v in manifest.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()

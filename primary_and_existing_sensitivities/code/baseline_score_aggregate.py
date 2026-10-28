"""Compute the corrected baseline score summary from local restricted tables.

Writes only aggregate values. Never writes patient identifiers or rows.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path((__import__("os").environ.get("ARIC_PROJECT_ROOT", "ARIC_PROJECT_ROOT") + ''))
RISK = ROOT / "ARIC_revision_20260920" / "secure_derived_inputs" / "revised_risk_day3.csv.gz"
DELTA = ROOT / "ARIC_revision_20261009" / "secure_derived_inputs" / "final_score_deltas_stay.csv.gz"
OUT = ROOT / "ARIC_revision_20261009" / "documents" / "baseline_corrected_score_aggregate.csv"


def main() -> None:
    risk = pd.read_csv(
        RISK,
        usecols=["stay_id", "t_start", "exposure_vulnerability_burden_score"],
    )
    delta = pd.read_csv(DELTA, usecols=["stay_id", "delta_score"])
    if delta.stay_id.duplicated().any():
        raise ValueError("Delta table must have one row per stay")
    baseline = risk.sort_values(["stay_id", "t_start"]).drop_duplicates("stay_id")
    baseline = baseline.merge(delta, on="stay_id", how="left", validate="one_to_one")
    baseline["delta_score"] = baseline["delta_score"].fillna(0)
    baseline["corrected_score"] = baseline["exposure_vulnerability_burden_score"] + baseline["delta_score"]
    baseline["group"] = pd.cut(
        baseline.corrected_score,
        bins=[-1, 1, 3, float("inf")],
        labels=["Low", "Moderate", "High"],
    )
    result = []
    for name, frame in [("Overall", baseline), *list(baseline.groupby("group", observed=True))]:
        result.append({
            "group": str(name),
            "stays": len(frame),
            "score_median": frame.corrected_score.quantile(0.5),
            "score_q1": frame.corrected_score.quantile(0.25),
            "score_q3": frame.corrected_score.quantile(0.75),
        })
    summary = pd.DataFrame(result)
    expected = {"Overall": 13863, "Low": 3046, "Moderate": 6090, "High": 4727}
    actual = dict(zip(summary.group, summary.stays))
    if actual != expected:
        raise AssertionError(f"Corrected group counts disagree with prior aggregate: {actual}")
    summary.to_csv(OUT, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

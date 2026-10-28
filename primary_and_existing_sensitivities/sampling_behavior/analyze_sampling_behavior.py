"""Descriptive day-3 blood-culture sampling behavior; aggregate exports only.

Protocol: SAMPLING_BEHAVIOR_PROTOCOL_20261011.md. Requires authorized local
derived risk tables and the previously source-checked collection cache. Raw
MIMIC files are neither read nor written by this script.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "ARIC_revision_20260920"
REV = ROOT / "ARIC_revision_20261009"
RISK = OLD / "secure_derived_inputs" / "revised_risk_day3.csv.gz"
COLLECTIONS = OLD / "route_A" / "secure_derived_inputs" / "sampling_collections_source_checked.csv.gz"
PRIOR_INTERVALS = OLD / "route_A" / "secure_derived_inputs" / "sampling_intervals_day3.csv.gz"
EXITS = OLD / "secure_derived_inputs" / "competing_events_day3.csv.gz"
DELTA = REV / "secure_derived_inputs" / "final_score_deltas_stay.csv.gz"
SAMPLING_MODULE = OLD / "route_A" / "code" / "run_sampling_analysis.py"
HORIZONS = (1, 3, 7, 14, 28)
GROUPS = ("Overall", "low", "moderate", "high")
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261011


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def legacy_sampling_module():
    spec = importlib.util.spec_from_file_location("frozen_route_a_sampling", SAMPLING_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import frozen Route A assignment")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantiles(values: pd.Series | np.ndarray) -> tuple[float | None, float | None, float | None]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None, None, None
    return tuple(float(x) for x in np.quantile(a, [0.25, 0.5, 0.75]))


def bootstrap_proportions(stay: pd.DataFrame) -> dict[tuple[str, int], tuple[float, float]]:
    """Resample patients globally; retain all same-patient stays as one block."""
    patients, patient_index = np.unique(stay.subject_id.to_numpy(), return_inverse=True)
    g = stay.baseline_group.to_numpy()
    first = stay.first_sample_day.to_numpy(dtype=float)
    cells: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for group in GROUPS:
        selected = np.ones(len(stay), dtype=bool) if group == "Overall" else g == group
        den = np.bincount(patient_index, weights=selected.astype(int), minlength=len(patients)).astype(int)
        for day in (0, *HORIZONS):
            hit = np.isfinite(first) if day == 0 else np.isfinite(first) & (first <= day + 1e-10)
            num = np.bincount(patient_index, weights=(selected & hit).astype(int), minlength=len(patients)).astype(int)
            cells[(group, day)] = (num, den)
    samples = {key: np.empty(BOOTSTRAP_REPLICATES, dtype=float) for key in cells}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for iteration in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(patients), size=len(patients))
        weight = np.bincount(draw, minlength=len(patients))
        for key, (num, den) in cells.items():
            denominator = int(weight @ den)
            samples[key][iteration] = float(weight @ num) / denominator if denominator else np.nan
    result = {}
    for key, values in samples.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Empty group in patient bootstrap: {key}")
        result[key] = tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))
    return result


def aalen_johansen_first_sampling(stay: pd.DataFrame, horizons: tuple[int, ...]) -> dict[int, float]:
    """First sampling versus non-sampling exit; censor only at day 28.

    At tied times, first sampling and competing exits are processed before
    administrative censoring. A terminal diagnostic specimen is event 1.
    """
    times = np.where(stay.sampled.to_numpy(), stay.first_sample_day.to_numpy(), stay.followup_days.to_numpy())
    kinds = np.where(stay.sampled.to_numpy(), 1, np.where(stay.cause.to_numpy() == 0, 0, 2))
    frame = pd.DataFrame({"time": times, "kind": kinds})
    at_time = frame.groupby("time", sort=True).kind.value_counts().unstack(fill_value=0)
    for kind in (0, 1, 2):
        if kind not in at_time.columns:
            at_time[kind] = 0
    number_at_risk = len(stay)
    survival = 1.0
    cumulative_sampling = 0.0
    curves: list[tuple[float, float]] = []
    for time, counts in at_time.iterrows():
        d_sampling = int(counts[1])
        d_exit = int(counts[2])
        d_censor = int(counts[0])
        cumulative_sampling += survival * d_sampling / number_at_risk
        survival *= 1 - (d_sampling + d_exit) / number_at_risk
        number_at_risk -= d_sampling + d_exit + d_censor
        curves.append((float(time), cumulative_sampling))
    if number_at_risk != 0:
        raise RuntimeError("Aalen-Johansen risk accounting did not close")
    result = {}
    for horizon in horizons:
        eligible = [value for time, value in curves if time <= horizon + 1e-10]
        result[horizon] = eligible[-1] if eligible else 0.0
    return result


def main() -> None:
    for path in (RISK, COLLECTIONS, PRIOR_INTERVALS, EXITS, DELTA, SAMPLING_MODULE):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_files = {
        "risk_day3": RISK,
        "source_checked_collections": COLLECTIONS,
        "locked_sampling_intervals": PRIOR_INTERVALS,
        "competing_exits_day3": EXITS,
        "final_score_delta_day3": DELTA,
        "frozen_assignment_code": SAMPLING_MODULE,
    }
    source_hashes = {label: sha256(path) for label, path in source_files.items()}
    sampling = legacy_sampling_module()
    cols = [
        "subject_id", "hadm_id", "stay_id", "day_start", "stop_datetime", "landmark_time",
        "t_start", "t_stop", "duration_days", "event", "exposure_vulnerability_burden_score",
    ]
    risk = pd.read_csv(RISK, usecols=cols, parse_dates=["day_start", "stop_datetime", "landmark_time"])
    risk = risk.reset_index(drop=True)
    check = sampling.validate_risk(risk)
    if (check["intervals"], check["stays"], check["patients"], check["events"]) != (60272, 13863, 12757, 268):
        raise RuntimeError("Locked cohort count changed")
    if not np.isclose(check["risk_person_days"], 56370.34172453703, atol=1e-8):
        raise RuntimeError("Locked risk time changed")

    delta = pd.read_csv(DELTA, usecols=["stay_id", "delta_cancer", "delta_meta", "delta_score"])
    if delta.stay_id.duplicated().any() or not (delta.delta_score == delta.delta_cancer + delta.delta_meta).all():
        raise RuntimeError("Invalid final score delta")
    first = risk.sort_values(["stay_id", "day_start"]).drop_duplicates("stay_id", keep="first").copy()
    first = first.merge(delta[["stay_id", "delta_score"]], on="stay_id", how="left", validate="one_to_one")
    if first.delta_score.isna().any():
        raise RuntimeError("Missing corrected score delta")
    first["baseline_score"] = first.exposure_vulnerability_burden_score + first.delta_score
    if not np.allclose(first.baseline_score, np.rint(first.baseline_score)):
        raise RuntimeError("Baseline score not integer-valued")
    first["baseline_group"] = np.where(first.baseline_score <= 1, "low", np.where(first.baseline_score <= 3, "moderate", "high"))
    first = first[["stay_id", "subject_id", "hadm_id", "landmark_time", "baseline_score", "baseline_group"]]

    collections = pd.read_csv(COLLECTIONS, parse_dates=["event_time", "charttime"])
    if collections.collection_key.isna().any() or collections.collection_key.duplicated().any():
        raise RuntimeError("Collection keys incomplete or duplicated")
    corrected = collections.loc[collections.has_non_cancelled.eq(1)].copy()
    assigned = sampling.assign_collections(risk, corrected)
    if len(assigned) != 19160:
        raise RuntimeError(f"Assigned specimen count changed: {len(assigned)}")
    if assigned.charttime.isna().any() or assigned.event_time.isna().any():
        raise RuntimeError("Assigned collection lacks exact charttime")
    if not assigned.event_time.eq(assigned.charttime).all():
        raise RuntimeError("Assigned collection time differs from exact charttime")
    assigned["stay_id"] = risk.stay_id.to_numpy()[assigned.risk_row.to_numpy()]
    assigned["subject_id_risk"] = risk.subject_id.to_numpy()[assigned.risk_row.to_numpy()]
    if not assigned.subject_id.eq(assigned.subject_id_risk).all():
        raise RuntimeError("Assigned patient does not match risk row")
    occasion_by_row = assigned.groupby(["risk_row", "event_time"], sort=False).size().rename("specimens").reset_index()
    occasions = assigned.groupby(["stay_id", "event_time"], sort=False).size().rename("specimens").reset_index()
    if len(occasions) != 16086 or len(occasion_by_row) != len(occasions):
        raise RuntimeError("Same-timestamp occasion count differs from locked analysis")
    if occasions.duplicated(["stay_id", "event_time"]).any():
        raise RuntimeError("Duplicate stay occasion")
    old = pd.read_csv(PRIOR_INTERVALS, usecols=["stay_id", "day_start", "collection_count", "occasion_count"], parse_dates=["day_start"])
    if len(old) != len(risk) or not old[["stay_id", "day_start"]].equals(risk[["stay_id", "day_start"]]):
        raise RuntimeError("Prior sampling interval order differs from risk input")
    new_collection_count = np.bincount(assigned.risk_row, minlength=len(risk))
    new_occasion_count = np.bincount(occasion_by_row.risk_row, minlength=len(risk))
    if not np.array_equal(new_collection_count, old.collection_count.to_numpy()) or not np.array_equal(new_occasion_count, old.occasion_count.to_numpy()):
        raise RuntimeError("Assigned counts differ from locked Route A sampling intervals")

    spans = risk.groupby("stay_id", sort=False).agg(
        followup_end=("stop_datetime", "max"),
        risk_days=("duration_days", "sum"),
        intervals=("stay_id", "size"),
        detected_bsi=("event", "sum"),
    ).reset_index()
    stay = first.merge(spans, on="stay_id", how="left", validate="one_to_one")
    stay["followup_days"] = (stay.followup_end - stay.landmark_time).dt.total_seconds() / 86400
    if not np.allclose(stay.followup_days, stay.risk_days, atol=2 / 86400):
        raise RuntimeError("Stay follow-up differs from sum of risk intervals")
    exits = pd.read_csv(EXITS, usecols=["stay_id", "subject_id", "time_days", "cause", "observed_end"])
    if exits.stay_id.duplicated().any() or len(exits) != len(stay):
        raise RuntimeError("Competing exit rows do not match stays")
    stay = stay.merge(exits, on=["stay_id", "subject_id"], how="left", validate="one_to_one")
    # For BSI, `risk_end` is the later scheduled non-BSI horizon; `observed_end`
    # is the actual event stop and must match the final risk interval exactly.
    stay.observed_end = pd.to_datetime(stay.observed_end)
    exit_time_gap_seconds = (stay.followup_end - stay.observed_end).dt.total_seconds().abs()
    if stay.cause.isna().any() or (exit_time_gap_seconds > 2).any():
        raise RuntimeError("Existing competing exit file and risk stops disagree")
    if ((stay.cause == 0) & (stay.time_days < 28 - 1e-8)).any():
        raise RuntimeError("Unexpected administrative censoring before day 28")

    occasions = occasions.sort_values(["stay_id", "event_time"], kind="stable")
    oc_count = occasions.groupby("stay_id", sort=False).size().rename("occasion_count").reset_index()
    specimen_count = assigned.groupby("stay_id", sort=False).size().rename("specimen_count").reset_index()
    first_occ = occasions.groupby("stay_id", sort=False).event_time.min().rename("first_sample_time").reset_index()
    stay = stay.merge(oc_count, on="stay_id", how="left", validate="one_to_one")
    stay = stay.merge(specimen_count, on="stay_id", how="left", validate="one_to_one")
    stay = stay.merge(first_occ, on="stay_id", how="left", validate="one_to_one")
    stay[["occasion_count", "specimen_count"]] = stay[["occasion_count", "specimen_count"]].fillna(0).astype(int)
    stay["sampled"] = stay.occasion_count > 0
    stay["first_sample_day"] = (stay.first_sample_time - stay.landmark_time).dt.total_seconds() / 86400
    if (stay.loc[stay.sampled, "first_sample_day"] < -1e-9).any() or (stay.loc[stay.sampled, "first_sample_day"] > stay.loc[stay.sampled, "followup_days"] + 2 / 86400).any():
        raise RuntimeError("First collection outside follow-up")
    if stay.loc[stay.cause.eq(1) & ~stay.sampled].shape[0]:
        raise RuntimeError("Detected BSI exit without an assigned sampling occasion")
    if int(stay.specimen_count.sum()) != 19160 or int(stay.occasion_count.sum()) != 16086:
        raise RuntimeError("Stay collection/occasion totals changed")
    first_at_terminal = stay.sampled & ((stay.first_sample_time - stay.followup_end).dt.total_seconds().abs() <= 1e-9)

    occasions["gap_hours"] = occasions.groupby("stay_id").event_time.diff().dt.total_seconds() / 3600
    gaps = occasions.loc[occasions.gap_hours.notna(), ["stay_id", "gap_hours"]].copy()
    if (gaps.gap_hours <= 0).any():
        raise RuntimeError("Nonpositive gap between unique occasion times")
    gaps = gaps.merge(first[["stay_id", "baseline_group"]], on="stay_id", how="left", validate="many_to_one")
    stay_median_gap = gaps.groupby("stay_id").gap_hours.median().rename("stay_median_gap_hours").reset_index()
    stay_median_gap = stay_median_gap.merge(first[["stay_id", "baseline_group"]], on="stay_id", how="left", validate="one_to_one")
    if int(len(gaps)) != 16086 - int(stay.sampled.sum()):
        raise RuntimeError("First plus subsequent occasion partition differs")
    ci = bootstrap_proportions(stay)

    group_rows = []
    frequency_rows = []
    gap_rows = []
    cif_rows = []
    aj_max_absolute_difference = 0.0
    for group in GROUPS:
        g = stay if group == "Overall" else stay.loc[stay.baseline_group.eq(group)]
        aj = aalen_johansen_first_sampling(g, HORIZONS)
        sampled = g.loc[g.sampled]
        d25, d50, d75 = quantiles(g.followup_days)
        p = float(g.sampled.mean())
        lo, hi = ci[(group, 0)]
        person_days = float(g.risk_days.sum())
        group_rows.append({
            "baseline_group": group, "eligible_stays": len(g), "distinct_patients": g.subject_id.nunique(),
            "risk_intervals": int(g.intervals.sum()), "risk_person_days": person_days,
            "followup_days_q1": d25, "followup_days_median": d50, "followup_days_q3": d75,
            "sampled_stays": int(g.sampled.sum()), "no_sampling_stays": int((~g.sampled).sum()),
            "sampling_coverage": p, "sampling_coverage_cluster_bootstrap_ci_lower": lo,
            "sampling_coverage_cluster_bootstrap_ci_upper": hi,
            "specimen_keys": int(g.specimen_count.sum()), "same_time_collapsed_occasions": int(g.occasion_count.sum()),
            "occasions_per_1000_at_risk_person_days": float(g.occasion_count.sum() / person_days * 1000),
            "first_occasions": len(sampled), "subsequent_occasions": int((g.occasion_count - g.sampled.astype(int)).sum()),
            "detected_bsi_events": int(g.detected_bsi.sum()),
        })
        c25, c50, c75 = quantiles(sampled.occasion_count)
        for label, mask in [
            ("1", sampled.occasion_count.eq(1)), ("2", sampled.occasion_count.eq(2)),
            ("3-4", sampled.occasion_count.between(3, 4)), (">=5", sampled.occasion_count.ge(5)),
        ]:
            frequency_rows.append({
                "baseline_group": group, "sampled_stays_denominator": len(sampled), "occasion_count_category": label,
                "stays": int(mask.sum()), "fraction_among_sampled": float(mask.mean()),
                "occasion_count_q1_among_sampled": c25, "occasion_count_median_among_sampled": c50,
                "occasion_count_q3_among_sampled": c75,
            })
        gg = gaps if group == "Overall" else gaps.loc[gaps.baseline_group.eq(group)]
        sg = stay_median_gap if group == "Overall" else stay_median_gap.loc[stay_median_gap.baseline_group.eq(group)]
        f25, f50, f75 = quantiles(sampled.first_sample_day * 24)
        gap25, gap50, gap75 = quantiles(gg.gap_hours)
        staygap25, staygap50, staygap75 = quantiles(sg.stay_median_gap_hours)
        gap_rows.append({
            "baseline_group": group, "sampled_stays_denominator": len(sampled),
            "first_sample_hours_q1": f25, "first_sample_hours_median": f50, "first_sample_hours_q3": f75,
            "stays_with_2plus_occasions": len(sg), "consecutive_unique_occasion_gaps": len(gg),
            "all_gap_hours_q1": gap25, "all_gap_hours_median": gap50, "all_gap_hours_q3": gap75,
            "per_stay_median_gap_hours_q1": staygap25, "per_stay_median_gap_hours_median": staygap50,
            "per_stay_median_gap_hours_q3": staygap75,
        })
        for horizon in HORIZONS:
            num = int((g.first_sample_day <= horizon + 1e-10).sum())
            direct = num / len(g)
            aj_max_absolute_difference = max(aj_max_absolute_difference, abs(aj[horizon] - direct))
            competing = int(((~g.sampled) & (g.cause != 0) & (g.time_days <= horizon + 1e-10)).sum())
            # For sampled stays, exits before first sampling would be invalid and were checked above.
            loh, hih = ci[(group, horizon)]
            cif_rows.append({
                "baseline_group": group, "days_since_landmark": horizon, "eligible_stays_denominator": len(g),
                "first_sampling_by_horizon": num, "first_sampling_cumulative_incidence": direct,
                "aalen_johansen_cumulative_incidence_check": aj[horizon],
                "patient_cluster_bootstrap_ci_lower": loh, "patient_cluster_bootstrap_ci_upper": hih,
                "non_sampling_competing_exits_by_horizon": competing,
                "administratively_censored_unsampled_by_horizon": int(((~g.sampled) & g.cause.eq(0) & (g.time_days <= horizon + 1e-10)).sum()),
            })

    if aj_max_absolute_difference > 1e-10:
        raise RuntimeError(f"Aalen-Johansen and direct cohort fraction disagree: {aj_max_absolute_difference}")

    summary = pd.DataFrame(group_rows)
    frequency = pd.DataFrame(frequency_rows)
    gap_summary = pd.DataFrame(gap_rows)
    cif = pd.DataFrame(cif_rows)
    if int(summary.loc[summary.baseline_group.ne("Overall"), "eligible_stays"].sum()) != len(stay):
        raise RuntimeError("Baseline groups do not partition cohort")
    if int(summary.loc[summary.baseline_group.ne("Overall"), "same_time_collapsed_occasions"].sum()) != 16086:
        raise RuntimeError("Baseline groups do not partition occasions")
    if int(summary.loc[summary.baseline_group.ne("Overall"), "subsequent_occasions"].sum()) != len(gaps):
        raise RuntimeError("Subsequent occasion count and intervals do not agree")
    if frequency.groupby("baseline_group").stays.sum().to_dict() != summary.set_index("baseline_group").sampled_stays.to_dict():
        raise RuntimeError("Occasion-count categories do not partition sampled stays")

    qc = {
        "protocol": "SAMPLING_BEHAVIOR_PROTOCOL_20261011.md",
        "analysis_type": "revision-stage descriptive; not preregistered; no new regression",
        "source_hashes_sha256": source_hashes,
        "locked_cohort": check,
        "source_checked_collection_rows": len(collections),
        "source_checked_cancelled_only_rows": int(collections.has_non_cancelled.eq(0).sum()),
        "assigned_noncancelled_specimen_keys": len(assigned),
        "assigned_exact_charttime_missing": int(assigned.charttime.isna().sum()),
        "same_time_collapsed_occasions": len(occasions),
        "occasion_rows_match_old_sampling_interval_cache": True,
        "specimen_rows_match_old_sampling_interval_cache": True,
        "first_occasions": int(stay.sampled.sum()),
        "subsequent_occasions": len(gaps),
        "risk_exit_time_max_abs_discrepancy_seconds": float(exit_time_gap_seconds.max()),
        "exit_cause_counts_all_stays": {str(int(k)): int(v) for k, v in stay.cause.value_counts().sort_index().items()},
        "unsampled_detected_bsi_exits": int((stay.cause.eq(1) & ~stay.sampled).sum()),
        "first_sampling_at_terminal_time_stays": int(first_at_terminal.sum()),
        "first_sampling_at_terminal_bsi_time_stays": int((first_at_terminal & stay.cause.eq(1)).sum()),
        "aalen_johansen_vs_direct_max_absolute_difference": aj_max_absolute_difference,
        "administrative_censors_before_day28": int((stay.cause.eq(0) & stay.time_days.lt(28 - 1e-8)).sum()),
        "baseline_groups_frozen_at_first_interval": True,
        "bootstrap_sampling_unit": "patient with all ICU stays retained",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "quantile_definition": "numpy linear interpolation, 25th/50th/75th percentiles",
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "patient_rows_exported": False,
    }
    summary.to_csv(HERE / "sampling_behavior_group_summary.csv", index=False)
    frequency.to_csv(HERE / "sampling_behavior_occasion_distribution.csv", index=False)
    gap_summary.to_csv(HERE / "sampling_behavior_gap_summary.csv", index=False)
    cif.to_csv(HERE / "sampling_behavior_first_sampling_cif.csv", index=False)
    (HERE / "sampling_behavior_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "complete", "stays": len(stay), "specimens": len(assigned), "occasions": len(occasions),
        "sampled_stays": int(stay.sampled.sum()), "groups": summary[["baseline_group", "eligible_stays", "sampled_stays"]].to_dict("records"),
    }, indent=2))


if __name__ == "__main__":
    main()

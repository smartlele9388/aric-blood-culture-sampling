"""Independent arithmetic audit of saved ARIC Cox and sampling Poisson models.

This script reads the local patient-level *derived* intervals, never writes them,
and emits only aggregate diagnostic JSON. Historical sources/outputs are read-only.
Run from the project venv: python audit_saved_primary_models.py [--final].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm
from statsmodels.duration.hazard_regression import PHReg

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "ARIC_revision_20260920"
OUT = Path(__file__).resolve().parent
FINAL = BASE / "association_audit_20261003" / "corrected_models_final"
RISK = BASE / "secure_derived_inputs" / "revised_risk_day3.csv.gz"
SAMPLE = BASE / "route_A" / "secure_derived_inputs" / "sampling_intervals_day3.csv.gz"
DELTA = ROOT / "ARIC_revision_20261009" / "secure_derived_inputs" / "final_score_deltas_stay.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_risk(risk: pd.DataFrame) -> dict:
    n = len(risk)
    assert risk[["subject_id", "stay_id", "t_start", "t_stop", "duration_days", "event"]].notna().all().all()
    assert (risk.t_stop > risk.t_start).all()
    assert np.allclose(risk.t_stop - risk.t_start, risk.duration_days, atol=1e-12)
    assert risk.groupby("stay_id").event.sum().le(1).all()
    assert risk.event.isin([0, 1]).all()
    assert risk.loc[risk.event.eq(1), "t_stop"].gt(0).all()
    ordered = risk.sort_values(["stay_id", "t_start", "t_stop"])
    gap = ordered.groupby("stay_id", sort=False).t_start.shift(-1) - ordered.t_stop
    assert np.allclose(gap.dropna(), 0, atol=1e-12)
    return {
        "intervals": n,
        "events": int(risk.event.sum()),
        "stays": int(risk.stay_id.nunique()),
        "patients": int(risk.subject_id.nunique()),
        "risk_person_days": float(risk.duration_days.sum()),
        "delayed_entry_intervals": int(risk.t_start.gt(0).sum()),
        "event_at_integer_day_boundary": int(np.isclose(risk.loc[risk.event.eq(1), "t_stop"] % 1, 0).sum()),
        "max_events_per_stay": int(risk.groupby("stay_id").event.sum().max()),
        "max_abs_duration_difference_days": float(np.abs(risk.t_stop - risk.t_start - risk.duration_days).max()),
    }


def score_row_risk_set_check(risk: pd.DataFrame, finite: np.ndarray, entry: np.ndarray,
                             score_rows: np.ndarray) -> dict:
    # PHReg has one unstratified time-since-landmark axis. A row can contribute
    # partial-likelihood information iff it enters at least one event risk set.
    event_times = np.sort(risk.loc[risk.event.eq(1), "t_stop"].to_numpy(dtype=float))
    next_index = np.searchsorted(event_times, entry, side="right")
    possible = next_index < len(event_times)
    possible[possible] &= event_times[next_index[possible]] <= risk.t_stop.to_numpy(dtype=float)[possible]
    return {
        "finite_score_rows": int(finite.sum()),
        "nonfinite_score_rows": int((~finite).sum()),
        "event_rows_with_nonfinite_score": int((risk.event.to_numpy(dtype=bool) & ~finite).sum()),
        "nonfinite_rows_entering_an_event_risk_set": int((~finite & possible).sum()),
        "finite_rows_entering_no_event_risk_set": int((finite & ~possible).sum()),
        "finite_rows_entering_no_event_risk_set_with_nonzero_score": int(
            (finite & ~possible & (np.abs(np.nan_to_num(score_rows, nan=0.0)).max(axis=1) > 1e-12)).sum()
        ),
        "rows_entering_any_event_risk_set": int(possible.sum()),
    }


def score_residual_set_recursion_check(model: PHReg) -> dict:
    """Audit the set recursion used by PHReg.score_residuals, not its fitted risk sums."""
    s = model.surv
    assert s.nstrat == 1
    at_risk: set[int] = set()
    discrepancies = []
    for i in range(len(s.ufailt[0]) - 1, -1, -1):
        at_risk.update(int(x) for x in s.risk_enter[0][i])
        t = s.ufailt[0][i]
        direct = set(np.flatnonzero((s.entry_s[0] <= t) & (s.time_s[0] >= t)).tolist())
        discrepancies.append(len(at_risk.symmetric_difference(direct)))
        at_risk.difference_update(int(x) for x in s.risk_exit[0][i])
    return {"event_times_checked": len(discrepancies),
            "event_times_with_risk_set_mismatch": int(np.count_nonzero(discrepancies)),
            "largest_symmetric_difference_rows": int(max(discrepancies, default=0)),
            "total_symmetric_difference_rows": int(sum(discrepancies))}


def check_cox(risk: pd.DataFrame, name: str, coefficients: Path, covariance: Path, rhs: str) -> dict:
    risk = risk.copy()
    risk["exposure_vulnerability_category"] = pd.Categorical(
        risk.exposure_vulnerability_category, categories=["low", "moderate", "high"]
    )
    design = patsy.dmatrix(rhs, risk, return_type="dataframe", NA_action="raise")
    design = design.drop(columns="Intercept", errors="ignore")
    saved = pd.read_csv(coefficients)
    with np.load(covariance, allow_pickle=False) as bundle:
        terms = bundle["terms"].tolist()
        beta = bundle["coefficients"].astype(float)
        cov = bundle["covariance"].astype(float)
    assert list(design.columns) == terms == saved.term.tolist()
    assert len(beta) == design.shape[1]
    entry = np.nextafter(risk.t_start.to_numpy(dtype=float), np.inf)
    model = PHReg(
        risk.t_stop.to_numpy(dtype=float), design.to_numpy(dtype=float),
        status=risk.event.to_numpy(dtype=int), entry=entry, ties="efron"
    )
    gradient = np.asarray(model.score(beta), dtype=float)
    breslow_gradient = np.asarray(model.breslow_gradient(beta), dtype=float)
    hessian = np.asarray(model.hessian(beta), dtype=float)
    rows = np.asarray(model.score_residuals(beta), dtype=float)
    finite = np.isfinite(rows).all(axis=1)
    assert np.isfinite(rows[finite]).all()
    # Reconstruct the patient-cluster sandwich without using the saved routine.
    group, inv = np.unique(risk.subject_id.to_numpy()[finite], return_inverse=True)
    cluster_scores = np.zeros((len(group), len(beta)), dtype=float)
    np.add.at(cluster_scores, inv, rows[finite])
    meat = cluster_scores.T @ cluster_scores
    bread = np.linalg.inv(-hessian)
    rebuilt_cov = bread @ meat @ bread
    se = np.sqrt(np.diag(cov))
    z = norm.ppf(0.975)
    ci_lower = np.exp(beta - z * se)
    ci_upper = np.exp(beta + z * se)
    p_value = 2 * norm.sf(np.abs(beta / se))
    checks = score_row_risk_set_check(risk, finite, entry, rows)
    result = {
        "model": name,
        "formula": rhs,
        "ties": "efron",
        "entry_rule": "nextafter(t_start, +infinity)",
        "input": summarize_risk(risk),
        "parameters": len(beta),
        "design_rank": int(np.linalg.matrix_rank(design.to_numpy())),
        "information_rank": int(np.linalg.matrix_rank(-hessian)),
        "information_condition": float(np.linalg.cond(-hessian)),
        "max_abs_gradient": float(np.abs(gradient).max()),
        "partial_log_likelihood": float(model.loglike(beta)),
        "max_abs_sum_score_residuals_minus_gradient": float(np.abs(rows[finite].sum(axis=0) - gradient).max()),
        "max_abs_sum_score_residuals_minus_breslow_gradient": float(
            np.abs(rows[finite].sum(axis=0) - breslow_gradient).max()
        ),
        "distinct_event_times": int(risk.loc[risk.event.eq(1), "t_stop"].nunique()),
        "event_tie_groups": int(risk.loc[risk.event.eq(1), "t_stop"].value_counts().gt(1).sum()),
        "events_in_tie_groups": int(risk.loc[risk.event.eq(1), "t_stop"].value_counts().loc[lambda s: s.gt(1)].sum()),
        "max_abs_saved_vs_rebuilt_covariance": float(np.abs(rebuilt_cov - cov).max()),
        "saved_covariance_min_eigenvalue": float(np.linalg.eigvalsh((cov + cov.T) / 2).min()),
        "saved_covariance_rank": int(np.linalg.matrix_rank(cov)),
        "max_abs_saved_vs_table_coefficient": float(np.abs(beta - saved.coefficient.to_numpy(dtype=float)).max()),
        "max_abs_saved_vs_table_robust_se": float(np.abs(se - saved.robust_se.to_numpy(dtype=float)).max()),
        "max_abs_saved_vs_table_hr": float(np.abs(np.exp(beta) - saved.hazard_ratio.to_numpy(dtype=float)).max()),
        "max_abs_saved_vs_table_ci": float(max(np.abs(ci_lower - saved.ci_lower.to_numpy(dtype=float)).max(),
                                               np.abs(ci_upper - saved.ci_upper.to_numpy(dtype=float)).max())),
        "max_abs_saved_vs_table_p_value": float(np.abs(p_value - saved.p_value.to_numpy(dtype=float)).max()),
        "score_row_check": checks,
        "score_residual_set_recursion_vs_counting_process_rule": score_residual_set_recursion_check(model),
        "coefficient_file_sha256": sha256(coefficients),
        "covariance_file_sha256": sha256(covariance),
    }
    assert result["information_rank"] == len(beta)
    assert checks["event_rows_with_nonfinite_score"] == 0
    assert result["max_abs_saved_vs_rebuilt_covariance"] < 1e-8
    assert result["max_abs_saved_vs_table_coefficient"] < 1e-12
    assert result["max_abs_saved_vs_table_ci"] < 1e-10
    return result


def check_poisson(sample: pd.DataFrame, name: str, coefficients: Path, covariance: Path, rhs: str,
                  outcome: str = "collection_count") -> dict:
    sample = sample.copy()
    sample["exposure_vulnerability_category"] = pd.Categorical(
        sample.exposure_vulnerability_category, categories=["low", "moderate", "high"]
    )
    sample["time_band"] = pd.Categorical(
        sample.time_band, categories=["0 to <2", "2 to <7", "7 to <14", "14 to 28"]
    )
    design = patsy.dmatrix(rhs, sample, return_type="dataframe", NA_action="raise")
    saved = pd.read_csv(coefficients)
    with np.load(covariance, allow_pickle=False) as bundle:
        beta = bundle["coefficients"].astype(float)
        cov = bundle["covariance"].astype(float)
        # The 2026-09-20 Poisson writer saved pandas Index labels as a
        # pickled object array. Never unpickle them; verify CSV labels instead.
    assert set(design.columns) == set(saved.term.tolist())
    # The historical object-valued category and final ordered Categorical use
    # different dummy-column order. Explicit term alignment is essential.
    design = design[saved.term.tolist()]
    x = design.to_numpy(dtype=float)
    y = sample[outcome].to_numpy(dtype=float)
    duration = sample.duration_days.to_numpy(dtype=float)
    assert (duration > 0).all()
    assert np.allclose(duration, sample.t_stop.to_numpy() - sample.t_start.to_numpy(), atol=1e-12)
    assert np.isfinite(np.log(duration)).all()
    mu = np.exp(x @ beta + np.log(duration))
    score_rows = x * (y - mu)[:, None]
    score = score_rows.sum(axis=0)
    info = x.T @ (x * mu[:, None])
    bread = np.linalg.inv(info)
    group, inv = np.unique(sample.subject_id.to_numpy(), return_inverse=True)
    collapsed = np.zeros((len(group), len(beta)), dtype=float)
    np.add.at(collapsed, inv, score_rows)
    correction = len(group) / (len(group) - 1) * (len(y) - 1) / (len(y) - len(beta))
    rebuilt_cov = correction * bread @ (collapsed.T @ collapsed) @ bread
    se = np.sqrt(np.diag(cov))
    table_ci_l = np.exp(beta - 1.96 * se)
    table_ci_u = np.exp(beta + 1.96 * se)
    result = {
        "model": name,
        "formula": rhs,
        "input_intervals": len(sample),
        "input_patients": int(sample.subject_id.nunique()),
        "outcome_variable": outcome,
        "outcome_count": int(y.sum()),
        "risk_person_days": float(duration.sum()),
        "offset": "log(actual interval duration in days)",
        "min_duration_days": float(duration.min()),
        "max_duration_days": float(duration.max()),
        "parameters": len(beta),
        "design_rank": int(np.linalg.matrix_rank(x)),
        "information_rank": int(np.linalg.matrix_rank(info)),
        "information_condition": float(np.linalg.cond(info)),
        "max_abs_score": float(np.abs(score).max()),
        "pearson_dispersion": float((((y - mu) ** 2 / mu).sum()) / (len(y) - len(beta))),
        "min_expected_collection_count": float(mu.min()),
        "max_expected_collection_count": float(mu.max()),
        "patient_cluster_finite_sample_correction": float(correction),
        "saved_terms_array_note": "terms.npy may require pickle; verified design against CSV labels without unpickling",
        "max_abs_saved_vs_rebuilt_covariance": float(np.abs(rebuilt_cov - cov).max()),
        "saved_covariance_rank": int(np.linalg.matrix_rank(cov)),
        "saved_covariance_min_eigenvalue": float(np.linalg.eigvalsh((cov + cov.T) / 2).min()),
        "max_abs_saved_vs_table_coefficient": float(np.abs(beta - saved.log_rate_ratio.to_numpy(dtype=float)).max()),
        "max_abs_saved_vs_table_robust_se": float(np.abs(se - saved.robust_se.to_numpy(dtype=float)).max()),
        "max_abs_saved_vs_table_irr": float(np.abs(np.exp(beta) - saved.rate_ratio.to_numpy(dtype=float)).max()),
        "max_abs_saved_vs_table_ci": float(max(np.abs(table_ci_l - saved.ci_lower.to_numpy(dtype=float)).max(),
                                               np.abs(table_ci_u - saved.ci_upper.to_numpy(dtype=float)).max())),
        "coefficient_file_sha256": sha256(coefficients),
        "covariance_file_sha256": sha256(covariance),
    }
    assert result["information_rank"] == len(beta)
    assert result["max_abs_saved_vs_rebuilt_covariance"] < 1e-8
    assert result["max_abs_saved_vs_table_ci"] < 1e-10
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="also verify final corrected score after local ICD delta reconstruction")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    risk = pd.read_csv(RISK)
    sample = pd.read_csv(SAMPLE)
    original_rhs = (
        "C(exposure_vulnerability_category, Treatment(reference='low')) + "
        "exposure_indication_present + age_per_10y + female + C(icu_group) + "
        "C(admission_group) + multiple_line + C(cvc_type_group) + "
        "exposure_antibiotic_exposure + exposure_lactate_available + exposure_aki_assessable"
    )
    sampling_rhs = original_rhs + " + C(time_band, Treatment(reference='0 to <2'))"
    result = {
        "original_cox": check_cox(risk, "historical_primary_day3", BASE / "results/primary_day3_coefficients.csv",
                                  BASE / "results/primary_day3_covariance.npz", original_rhs),
        "route_a_poisson": check_poisson(sample, "route_a_sampling_primary_specimens",
                                         BASE / "route_A/results/sampling_primary_specimens_coefficients.csv",
                                         BASE / "route_A/results/sampling_primary_specimens_covariance.npz", sampling_rhs),
        "local_input_hashes": {"risk_day3_sha256": sha256(RISK), "sampling_intervals_sha256": sha256(SAMPLE)},
        "patient_rows_written": False,
    }
    if args.final:
        sys.path.insert(0, str(BASE / "association_audit_20261003/code"))
        from coding_rules_final import correct_frame

        # Reuse the restricted local ICD mapping delta rather than rescanning
        # the large D: source. This patient-level intermediate is never copied
        # into the public audit output.
        delta = pd.read_csv(DELTA)
        assert delta.stay_id.is_unique and set(delta.stay_id) == set(risk.stay_id)
        assert delta.delta_cancer.add(delta.delta_meta).eq(delta.delta_score).all()
        corrected_risk = correct_frame(risk, delta)
        corrected_sample = correct_frame(sample, delta)
        assert corrected_sample.collection_count.sum() == sample.collection_count.sum()
        result["final_corrected_cox"] = check_cox(
            corrected_risk, "final_corrected_primary_day3", FINAL / "primary_day3_final_coefficients.csv",
            FINAL / "primary_day3_final_covariance.npz", original_rhs
        )
        result["final_corrected_poisson"] = check_poisson(
            corrected_sample, "final_corrected_sampling_primary_specimens",
            FINAL / "sampling_primary_specimens_final_coefficients.csv",
            FINAL / "sampling_primary_specimens_final_covariance.npz", sampling_rhs
        )
        result["coding_delta_reconstructed_locally"] = {
            "stays_with_flag_change": int((delta.delta_cancer.ne(0) | delta.delta_meta.ne(0)).sum()),
            "intervals_changing_score_category": int(
                corrected_risk.exposure_vulnerability_category.astype(str).ne(risk.exposure_vulnerability_category).sum()
            ),
        }
        result["local_input_hashes"]["restricted_stay_delta_sha256"] = sha256(DELTA)
    target = OUT / ("model_arithmetic_audit_final.json" if args.final else "model_arithmetic_audit_original.json")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: {"max_abs_saved_vs_rebuilt_covariance": v["max_abs_saved_vs_rebuilt_covariance"]}
                      for k, v in result.items() if isinstance(v, dict) and "max_abs_saved_vs_rebuilt_covariance" in v}, indent=2))
    print("WROTE", target)


if __name__ == "__main__":
    main()

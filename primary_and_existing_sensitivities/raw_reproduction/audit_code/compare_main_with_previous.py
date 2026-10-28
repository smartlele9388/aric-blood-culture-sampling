"""Aggregate-only reconciliation of raw-rebuilt main analysis with prior audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def maybe_small(n: int):
    return 0 if n == 0 else ("1-10" if n < 11 else n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    root = a.output_root.resolve()
    project = root.parents[1]
    out = root / "aggregate"
    old_risk = project / "ARIC_revision_20260920/secure_derived_inputs/revised_risk_day3.csv.gz"
    prior_r = project / "ARIC_revision_20261009/statistics_audit/r_survival_primary_final_full_precision.csv"
    prior_p = project / "ARIC_revision_20260920/association_audit_20261003/corrected_models_final/sampling_primary_specimens_final_coefficients.csv"
    new = pd.read_csv(root / "secure_derived_inputs/risk_final_day3.csv.gz")
    old = pd.read_csv(old_risk)
    key = ["stay_id", "t_start"]
    columns = key + ["subject_id", "hadm_id", "t_stop", "event", "duration_days", "age_per_10y", "female",
                     "icu_group", "admission_group", "multiple_line", "cvc_type_group", "exposure_antibiotic_exposure",
                     "exposure_lactate_available", "exposure_aki_assessable", "exposure_indication_present"]
    merged = new[columns].merge(old[columns], on=key, how="outer", suffixes=("_new", "_old"), indicator=True, validate="one_to_one")
    common = merged._merge.eq("both")
    discrepancies = {}
    for name in [c for c in columns if c not in key]:
        left, right = merged.loc[common, f"{name}_new"], merged.loc[common, f"{name}_old"]
        if name in ("t_stop", "duration_days", "age_per_10y"):
            mismatch = ~np.isclose(left, right, rtol=0, atol=1e-10, equal_nan=True)
        else:
            mismatch = left.fillna("<missing>").astype(str).ne(right.fillna("<missing>").astype(str))
        discrepancies[name] = maybe_small(int(mismatch.sum()))
    report = {
        "new_raw_rebuilt_intervals": len(new), "prior_derived_intervals": len(old),
        "matched_risk_keys": int(common.sum()),
        "new_only_risk_keys": maybe_small(int(merged._merge.eq("left_only").sum())),
        "prior_only_risk_keys": maybe_small(int(merged._merge.eq("right_only").sum())),
        "core_field_disagreements_on_matched_keys": discrepancies,
        "privacy": "No identifiers, patient rows or small nonzero cell counts exported",
    }
    models = []
    for name, new_file, old_file, coef_col in [
        ("Cox", out / "raw_reproduced_main_cox_full_precision.csv", prior_r, "coefficient"),
        ("sampling Poisson", out / "raw_reproduced_sampling_primary_specimens_coefficients.csv", prior_p, "log_rate_ratio"),
    ]:
        current = pd.read_csv(new_file)
        previous = pd.read_csv(old_file)
        merged_coef = current[["term", coef_col]].merge(previous[["term", coef_col]], on="term", how="outer",
                                                       suffixes=("_new", "_previous"), indicator=True, validate="one_to_one")
        merged_coef.insert(0, "model", name)
        merged_coef["absolute_difference"] = (merged_coef[f"{coef_col}_new"] - merged_coef[f"{coef_col}_previous"]).abs()
        models.append(merged_coef)
        report[name] = {"new_coefficients": len(current), "previous_coefficients": len(previous),
                        "matched_terms": int(merged_coef._merge.eq("both").sum()),
                        "max_absolute_log_coefficient_difference": float(merged_coef.absolute_difference.max())}
    new_r_cov = pd.read_csv(out / "raw_reproduced_main_cox_covariance.csv", index_col=0)
    old_r_cov = pd.read_csv(project / "ARIC_revision_20261009/statistics_audit/r_survival_primary_final_covariance.csv", index_col=0)
    assert list(new_r_cov.index) == list(old_r_cov.index) and list(new_r_cov.columns) == list(old_r_cov.columns)
    report["Cox"]["max_absolute_robust_covariance_difference"] = float(
        np.abs(new_r_cov.to_numpy() - old_r_cov.to_numpy()).max())
    new_p_cov = np.load(out / "raw_reproduced_sampling_primary_specimens_covariance.npz")
    old_p_cov = np.load(project / "ARIC_revision_20260920/association_audit_20261003/corrected_models_final/sampling_primary_specimens_final_covariance.npz")
    new_p_terms = pd.read_csv(out / "raw_reproduced_sampling_primary_specimens_coefficients.csv").term.tolist()
    old_p_terms = pd.read_csv(prior_p).term.tolist()
    assert set(new_p_terms) == set(old_p_terms)
    new_p_order = [new_p_terms.index(term) for term in old_p_terms]
    report["sampling Poisson"]["max_absolute_robust_covariance_difference"] = float(
        np.abs(new_p_cov["covariance"][np.ix_(new_p_order, new_p_order)] - old_p_cov["covariance"]).max())
    pd.concat(models, ignore_index=True).to_csv(out / "raw_vs_prior_main_coefficients.csv", index=False)
    (out / "raw_vs_prior_main_reconciliation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

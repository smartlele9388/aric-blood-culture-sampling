"""Compare independent R survival Cox fit with historical statsmodels output.

Only aggregate model quantities are exported. Local patient-level input is read
to verify the observed-information matrix, then discarded in memory.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import patsy
from statsmodels.duration.hazard_regression import PHReg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "ARIC_revision_20260920"
FINAL = BASE / "association_audit_20261003" / "corrected_models_final"
sys.path.insert(0, str(BASE / "association_audit_20261003/code"))
from coding_rules_final import correct_frame  # noqa: E402

RHS = (
    "C(exposure_vulnerability_category, Treatment(reference='low')) + "
    "exposure_indication_present + age_per_10y + female + C(icu_group) + "
    "C(admission_group) + multiple_line + C(cvc_type_group) + "
    "exposure_antibiotic_exposure + exposure_lactate_available + exposure_aki_assessable"
)


def r_to_python(name: str) -> str:
    if name.startswith("exposure_vulnerability_category"):
        level = name.removeprefix("exposure_vulnerability_category")
        return f"C(exposure_vulnerability_category, Treatment(reference='low'))[T.{level}]"
    for variable in ("icu_group", "admission_group", "cvc_type_group"):
        if name.startswith(variable):
            level = name.removeprefix(variable)
            return f"C({variable})[T.{level}]"
    return name


def check(mode: str, risk: pd.DataFrame) -> dict:
    risk = risk.copy()
    risk["exposure_vulnerability_category"] = pd.Categorical(
        risk.exposure_vulnerability_category, categories=["low", "moderate", "high"]
    )
    x = patsy.dmatrix(RHS, risk, return_type="dataframe").drop(columns="Intercept")
    source = BASE / "results" if mode == "original" else FINAL
    stem = "primary_day3" if mode == "original" else "primary_day3_final"
    py_table = pd.read_csv(source / f"{stem}_coefficients.csv").set_index("term")
    with np.load(source / f"{stem}_covariance.npz", allow_pickle=False) as npz:
        py_beta = npz["coefficients"]
        py_cov = npz["covariance"]
        py_terms = npz["terms"].tolist()
    assert list(x.columns) == py_terms
    r_table = pd.read_csv(HERE / f"r_survival_primary_{mode}_full_precision.csv")
    r_cov = pd.read_csv(HERE / f"r_survival_primary_{mode}_covariance.csv", index_col=0).to_numpy()
    r_naive = pd.read_csv(HERE / f"r_survival_primary_{mode}_naive_covariance.csv", index_col=0).to_numpy()
    mapped = [r_to_python(term) for term in r_table.term]
    ix = [py_terms.index(term) for term in mapped]
    assert len(ix) == len(set(ix)) == len(py_terms)
    p = py_table.loc[mapped].reset_index()
    frame = pd.DataFrame({
        "r_term": r_table.term,
        "python_term": mapped,
        "coefficient": r_table.coefficient,
        "hazard_ratio": r_table.hazard_ratio,
        "r_robust_se": r_table.robust_se,
        "python_robust_se": p.robust_se,
        "r_ci_lower": r_table.ci_lower,
        "r_ci_upper": r_table.ci_upper,
        "python_ci_lower": p.ci_lower,
        "python_ci_upper": p.ci_upper,
        "r_p_value": r_table.p_value,
        "python_p_value": p.p_value,
    })
    frame["robust_se_difference_r_minus_python"] = frame.r_robust_se - frame.python_robust_se
    frame["robust_se_relative_difference"] = frame.r_robust_se / frame.python_robust_se - 1
    frame.to_csv(HERE / f"r_vs_python_primary_{mode}_full_precision.csv", index=False)
    model = PHReg(
        risk.t_stop.to_numpy(dtype=float), x.to_numpy(dtype=float),
        status=risk.event.to_numpy(dtype=int),
        entry=np.nextafter(risk.t_start.to_numpy(dtype=float), np.inf), ties="efron"
    )
    bread = np.linalg.inv(-model.hessian(py_beta))
    bread_r_order = bread[np.ix_(ix, ix)]
    py_cov_r_order = py_cov[np.ix_(ix, ix)]
    result = {
        "mode": mode,
        "terms": len(ix),
        "max_abs_coefficient_difference": float(np.max(np.abs(r_table.coefficient.to_numpy() - py_beta[ix]))),
        "max_abs_naive_covariance_difference": float(np.max(np.abs(r_naive - bread_r_order))),
        "max_abs_robust_covariance_difference": float(np.max(np.abs(r_cov - py_cov_r_order))),
        "max_abs_robust_se_difference": float(np.max(np.abs(frame.robust_se_difference_r_minus_python))),
        "max_abs_relative_robust_se_difference": float(np.max(np.abs(frame.robust_se_relative_difference))),
        "terms_crossing_p_0_05_between_implementations": int(((frame.r_p_value < .05) != (frame.python_p_value < .05)).sum()),
        "partial_log_likelihood_python": float(model.loglike(py_beta)),
        "partial_log_likelihood_r": float(pd.read_csv(HERE / f"r_survival_primary_{mode}_diagnostics.csv").loglik.iloc[0]),
        "robust_variance_interpretation": "Coefficients and naive information agree; clustered sandwich differs because score-residual implementations differ. Use R survival::coxph cluster(subject_id) for revised inference.",
    }
    return result


def main():
    original = pd.read_csv(BASE / "secure_derived_inputs/revised_risk_day3.csv.gz")
    delta = pd.read_csv(ROOT / "ARIC_revision_20261009/secure_derived_inputs/final_score_deltas_stay.csv.gz")
    final = correct_frame(original, delta)
    result = {"original": check("original", original), "final": check("final", final)}
    (HERE / "r_vs_python_primary_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

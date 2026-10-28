"""Model-level aggregate effect tables for reviewer figures and supplement."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BATCH = HERE / "r_batch_results"
FINAL = ROOT / "ARIC_revision_20260920/association_audit_20261003/corrected_models_final"
ACUTE = ROOT / "ARIC_revision_20261009/analysis_core/results"
COX_TERMS = {
    "exposure_vulnerability_categorymoderate", "exposure_vulnerability_categoryhigh",
    "exposure_indication_present", "documented_chronic_count",
    "exposure_mechanical_ventilation", "exposure_creatinine_rise_documented",
    "exposure_lactate_high_documented", "ph_lactate_available",
}
POISSON_TERMS = {
    "C(exposure_vulnerability_category, Treatment(reference='low'))[T.moderate]",
    "C(exposure_vulnerability_category, Treatment(reference='low'))[T.high]",
    "exposure_indication_present", "documented_chronic_count",
    "prior_ventilation", "prior_creatinine_rise", "prior_high_lactate",
}


def main():
    cox = []
    for source in [BATCH, HERE / "r_ph_adjustment_results"]:
        for path in source.glob("*_coefficients.csv"):
            d = pd.read_csv(path)
            d = d[d.term.isin(COX_TERMS)].copy()
            if len(d):
                d["source_file"] = str(path)
                cox.append(d)
    out = pd.concat(cox, ignore_index=True)
    out.to_csv(HERE / "key_cox_results_for_figures_tables.csv", index=False)
    sampling = []
    for source in [FINAL, ACUTE]:
        for path in source.glob("sampling*_coefficients.csv"):
            d = pd.read_csv(path)
            d = d[d.term.isin(POISSON_TERMS)].copy()
            if len(d):
                d["source_file"] = str(path)
                sampling.append(d)
    pd.concat(sampling, ignore_index=True).to_csv(HERE / "key_sampling_results_for_figures_tables.csv", index=False)
    print("COX", len(out), "sampling", sum(map(len, sampling)))


if __name__ == "__main__":
    main()

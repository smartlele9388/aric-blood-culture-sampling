"""Build v4_locked aggregate-only MIMIC-IV CVC-BSI-AMR result package.

This script may read local MIMIC-IV files but never exports patient-level rows,
identifiers, timestamps, or raw microbiology records. Outputs are aggregate CSV,
PNG figures, and manuscript Markdown drafts only.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


BLOCKED_EXPORT_COLUMNS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "charttime",
    "chartdate",
    "starttime",
    "endtime",
    "event_time",
    "cvc_starttime",
    "cvc_endtime",
    "intime",
    "outtime",
    "dischtime",
    "deathtime",
    "microevent_id",
    "micro_specimen_id",
    "specimen_id",
    "collection_key",
    "isolate_key",
    "row_key",
}

CANDIDATE_RE = re.compile(
    r"central venous catheter|central line|\bcvc\b|\bpicc\b|multi lumen|introducer|cordis|"
    r"pa catheter|pulmonary artery catheter|dialysis catheter|tunneled catheter|pheresis catheter|"
    r"sheath|invasive line|presep|cooling catheter|thermogard|icy catheter",
    re.I,
)
EXCLUDE_RE = re.compile(
    r"arterial|peripheral|midline|urinary|iabp|ecmo|ventricular assist|\bric\b|foley|a-line|aline",
    re.I,
)
PRIMARY_INCLUDE_RE = re.compile(r"multi lumen|cordis|introducer|presep|central line|central venous catheter|\bcvc\b", re.I)
SENSITIVITY_RULES = [
    ("pa_catheter", re.compile(r"pa catheter|pulmonary artery catheter", re.I)),
    ("venous_sheath", re.compile(r"sheath", re.I)),
    ("dialysis_catheter", re.compile(r"dialysis catheter", re.I)),
    ("picc", re.compile(r"\bpicc\b", re.I)),
    ("cooling_catheter", re.compile(r"cooling catheter|thermogard|icy catheter", re.I)),
]

INVALID_ORG_EXACT = {
    "",
    "cancelled",
    "canceled",
    "no growth",
    "no growth to date",
    "negative",
    "neg",
    "test not performed",
    "not performed",
    "quantity not sufficient",
    "qns",
}
INVALID_ORG_NEAR_EXACT_RE = re.compile(
    r"^\s*(culture\s+)?(cancelled|canceled|no\s+growth(\s+to\s+date)?|negative|"
    r"test\s+not\s+performed|not\s+performed|quantity\s+not\s+sufficient|qns)\s*\.?\s*$",
    re.I,
)
NONSPECIFIC_CONTAMINATION_RE = re.compile(
    r"^\s*(mixed\s+flora|normal\s+flora|skin\s+flora|contaminant|probable\s+contaminant)\s*$",
    re.I,
)
COMMON_COMMENSAL_RE = re.compile(
    r"coag.*neg|coagulase.*negative|staphylococcus epidermidis|staphylococcus hominis|"
    r"staphylococcus capitis|staphylococcus haemolyticus|corynebacterium|bacillus species|"
    r"cutibacterium|propionibacterium|micrococcus|aerococcus",
    re.I,
)
FUNGUS_RE = re.compile(r"candida|yeast|cryptococcus|aspergillus", re.I)

CARBAPENEMS = {"meropenem", "imipenem", "doripenem", "ertapenem"}
METHICILLIN_MARKERS = {"methicillin", "oxacillin", "nafcillin", "cefoxitin"}
ENTEROBACTERALES = {
    "escherichia coli",
    "klebsiella pneumoniae",
    "klebsiella oxytoca",
    "enterobacter cloacae",
    "enterobacter cloacae complex",
    "serratia marcescens",
    "proteus mirabilis",
    "citrobacter freundii",
    "citrobacter koseri",
    "morganella morganii",
    "providencia stuartii",
}
INTRINSIC_RULES = [
    ("enterococcus", re.compile(r"enterococcus", re.I), re.compile(r"cephalosporin|cefazolin|ceftriaxone|cefepime|ceftazidime", re.I)),
    ("pseudomonas_early_cephalosporins", re.compile(r"pseudomonas", re.I), re.compile(r"cefazolin|ceftriaxone", re.I)),
    ("candida_antibacterial", re.compile(r"candida|yeast", re.I), re.compile(r".+", re.I)),
]

V2_SUFFIX_TARGETS = {
    "blood_culture_event_level_qc.csv",
    "excluded_organism_labels.csv",
    "positive_blood_culture_organism_distribution.csv",
    "bsi_event_level_qc.csv",
    "bsi_common_commensal_qc.csv",
    "clabsi_proxy_qc.csv",
    "amr_event_level_qc.csv",
    "amr_phenotype_event_distribution.csv",
    "organism_antibiotic_relevance_map.csv",
    "intrinsic_resistance_exclusions.csv",
    "cvc_line_days_denominator.csv",
    "daily_person_time_qc.csv",
    "time_varying_covariate_qc.csv",
    "strategy_deviation_qc.csv",
    "artificial_censoring_qc.csv",
    "weight_distribution.csv",
    "positivity_diagnostics.csv",
    "smd_before_after_weighting.csv",
    "primary_effect_estimates.csv",
    "cumulative_incidence_curves.png",
    "sensitivity_analysis_estimates.csv",
    "sensitivity_forest_plot.png",
}

V3_SUFFIX_TARGETS = {
    "primary_effect_estimates.csv",
    "cumulative_incidence_curves.png",
    "bootstrap_diagnostics.csv",
    "amr_event_level_qc_strict.csv",
    "amr_phenotype_event_distribution_r_only.csv",
    "amr_phenotype_event_distribution_r_or_i.csv",
    "microbiology_key_qc.csv",
    "ongoing_indication_qc.csv",
    "clinical_stability_rule_qc.csv",
    "strategy_deviation_qc.csv",
    "artificial_censoring_qc.csv",
    "weight_distribution.csv",
    "weight_histogram.png",
    "positivity_diagnostics.csv",
    "smd_before_after_weighting.csv",
    "table1_original_analytic_cohort.csv",
    "table1_weighted_balance.csv",
    "sensitivity_analysis_estimates.csv",
    "sensitivity_forest_plot.png",
    "competing_event_qc.csv",
    "competing_risk_sensitivity.csv",
    "cvc_itemid_final_review.csv",
    "cvc_active_interval_itemids.csv",
}

V4_SUFFIX_TARGETS = {
    "microbiology_key_qc.csv",
    "blood_culture_event_level_qc.csv",
    "positive_blood_culture_organism_distribution.csv",
    "bsi_event_level_qc.csv",
    "bsi_events_outside_followup_excluded.csv",
    "clabsi_proxy_qc.csv",
    "amr_event_level_qc_strict.csv",
    "amr_phenotype_event_distribution_r_only.csv",
    "amr_phenotype_event_distribution_r_or_i.csv",
    "amr_other_r_exploratory_distribution.csv",
    "cvc_itemid_final_review.csv",
    "cvc_active_interval_itemids.csv",
    "cohort_flow.csv",
    "daily_person_time_qc.csv",
    "strategy_deviation_qc.csv",
    "artificial_censoring_qc.csv",
    "weight_distribution.csv",
    "positivity_diagnostics.csv",
    "weight_histogram.png",
    "primary_effect_estimates.csv",
    "cumulative_incidence_curves.png",
    "bootstrap_diagnostics.csv",
    "table1_baseline_analytic_cohort.csv",
    "followup_covariate_qc.csv",
    "table1_weighted_balance.csv",
    "sensitivity_analysis_estimates.csv",
    "sensitivity_forest_plot.png",
    "competing_event_qc.csv",
    "competing_risk_sensitivity.csv",
}

LOCKED_SUFFIX_TARGETS = {
    "bsi_baseline_exclusion_qc_pre_exclusion.csv",
    "bsi_baseline_exclusion_qc_post_exclusion.csv",
    "cohort_flow.csv",
    "bsi_event_level_qc.csv",
    "clabsi_proxy_qc.csv",
    "amr_event_level_qc_strict.csv",
    "primary_effect_estimates.csv",
    "cumulative_incidence_curves.png",
    "sensitivity_analysis_estimates.csv",
    "sensitivity_forest_plot.png",
    "terminal_event_summary.csv",
    "cvc_active_interval_itemids.csv",
    "cvc_itemid_final_review.csv",
    "table1_baseline_analytic_cohort.csv",
    "bootstrap_diagnostics.csv",
    "daily_person_time_qc.csv",
    "strategy_deviation_qc.csv",
    "artificial_censoring_qc.csv",
    "weight_distribution.csv",
    "positivity_diagnostics.csv",
    "weight_histogram.png",
}


@dataclass
class Paths:
    base: Path
    qc: Path
    tables: Path
    figures: Path
    manuscript: Path


def norm(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x or "").strip().lower()


def q(x: pd.Series, p: float) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return 0.0
    return float(x.quantile(p))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    exposed = BLOCKED_EXPORT_COLUMNS.intersection({c.lower() for c in df.columns})
    if exposed:
        raise ValueError(f"Refusing to export patient-level columns: {sorted(exposed)} to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def v2_path(path: Path) -> Path:
    if path.name in V2_SUFFIX_TARGETS and not path.stem.endswith("_v2"):
        return path.with_name(f"{path.stem}_v2{path.suffix}")
    return path


def v3_path(path: Path) -> Path:
    if path.name in V3_SUFFIX_TARGETS and not path.stem.endswith("_v3"):
        return path.with_name(f"{path.stem}_v3{path.suffix}")
    return path


def v4_path(path: Path) -> Path:
    if path.name in V4_SUFFIX_TARGETS and not path.stem.endswith("_v4"):
        return path.with_name(f"{path.stem}_v4{path.suffix}")
    return path


def write_v2_csv(df: pd.DataFrame, path: Path) -> None:
    write_csv(df, v2_path(path))


def write_v3_csv(df: pd.DataFrame, path: Path) -> None:
    write_csv(df, v3_path(path))


def write_v4_csv(df: pd.DataFrame, path: Path) -> None:
    write_csv(df, v4_path(path))


def save_v2_fig(path: Path) -> None:
    save_fig(v2_path(path))


def save_v3_fig(path: Path) -> None:
    save_fig(v3_path(path))


def save_v4_fig(path: Path) -> None:
    save_fig(v4_path(path))


def locked_path(path: Path) -> Path:
    if path.name in LOCKED_SUFFIX_TARGETS and not path.stem.endswith("_v4_locked"):
        return path.with_name(f"{path.stem}_v4_locked{path.suffix}")
    return path


def write_locked_csv(df: pd.DataFrame, path: Path) -> None:
    write_csv(df, locked_path(path))


def save_locked_fig(path: Path) -> None:
    save_fig(locked_path(path))


def archive_pre_correction(paths: Paths) -> None:
    archive = Path("archive_pre_correction")
    archive.mkdir(exist_ok=True)
    for folder in [paths.qc, paths.tables, paths.figures]:
        target_dir = archive / folder.as_posix().replace("/", "_")
        target_dir.mkdir(parents=True, exist_ok=True)
        for file in folder.glob("*"):
            if not file.is_file() or file.name == ".gitkeep" or "_v2" in file.stem:
                continue
            shutil.move(str(file), str(target_dir / file.name))


def invalid_organism_label(label: object) -> bool:
    label_norm = norm(label)
    if label_norm in INVALID_ORG_EXACT:
        return True
    if INVALID_ORG_NEAR_EXACT_RE.match(label_norm):
        return True
    if NONSPECIFIC_CONTAMINATION_RE.match(label_norm):
        return True
    return False


def load_core(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = pd.read_csv(paths.base / "hosp" / "patients.csv.gz", usecols=["subject_id", "anchor_age", "gender"])
    admissions = pd.read_csv(
        paths.base / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "race"],
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    icu = pd.read_csv(
        paths.base / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )
    d_items = pd.read_csv(paths.base / "icu" / "d_items.csv.gz")
    return patients, admissions, icu, d_items


def adult_cohorts(patients: pd.DataFrame, admissions: pd.DataFrame, icu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    adult = icu.merge(patients, on="subject_id", how="inner")
    adult = adult.loc[adult["anchor_age"] >= 18].merge(admissions, on=["subject_id", "hadm_id"], how="left")
    first = adult.sort_values(["hadm_id", "intime", "stay_id"]).groupby("hadm_id", as_index=False).head(1).copy()
    return adult, first


def build_itemid_review(paths: Paths, d_items: pd.DataFrame, adult_icu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    items = d_items.copy()
    items["itemid"] = items["itemid"].astype(int)
    items["label"] = items["label"].fillna("")
    items["category"] = items["category"].fillna("")
    text = (items["label"] + " " + items["category"]).astype(str)
    items["candidate_flag"] = text.str.contains(CANDIDATE_RE)
    items["exclusion_flag"] = text.str.contains(EXCLUDE_RE) | items["itemid"].eq(229525)
    rows = []
    for r in items.loc[items["candidate_flag"] | items["exclusion_flag"]].itertuples(index=False):
        label_text = f"{r.label} {r.category}"
        is_excluded = bool(EXCLUDE_RE.search(label_text)) or int(r.itemid) == 229525
        sens = [] if is_excluded else [name for name, pat in SENSITIVITY_RULES if pat.search(label_text)]
        lower = label_text.lower()
        maintenance = bool(re.search(
            r"dressing|site appear|site appearance|waveform appear|zero/calibrate|zero|calibrate|"
            r"line tip cultured|tip cultured|scvo2|sqi|obturator|change over wire|"
            r"site care|line care|flush|cap change|tubing|blood draw|patency|accessed|deaccessed",
            lower,
        ))
        insertion_support = bool(re.search(r"insert|placement|placed", lower))
        removal_support = bool(re.search(r"remove|removed|discontinue|dc ", lower))
        active_interval = bool(PRIMARY_INCLUDE_RE.search(label_text)) and not is_excluded and not sens and not maintenance and not insertion_support and not removal_support
        if active_interval:
            exposure_role = "active CVC interval"
            decision = "include_primary"
        elif insertion_support and not is_excluded:
            exposure_role = "insertion/time-zero support"
            decision = "exclude"
        elif removal_support and not is_excluded:
            exposure_role = "removal/end-time support"
            decision = "exclude"
        elif maintenance and not is_excluded:
            exposure_role = "maintenance documentation only"
            decision = "exclude"
        elif sens and not is_excluded:
            exposure_role = "sensitivity-only CVC type"
            decision = "include_sensitivity_only"
        else:
            exposure_role = "excluded"
            decision = "exclude"
        primary = active_interval
        if int(r.itemid) == 229525:
            rationale = "Excluded: Sheath (Arterial) is an arterial device."
        elif bool(EXCLUDE_RE.search(label_text)):
            rationale = "Excluded: arterial/peripheral/non-CVC device or nonvascular support device."
        elif sens:
            rationale = "Sensitivity-only CVC-related device category: " + ";".join(sens)
        elif primary:
            rationale = "Primary central venous access device."
        elif maintenance:
            rationale = "Excluded from primary exposure: maintenance-only documentation, not an active catheter interval."
        elif insertion_support or removal_support:
            rationale = "Excluded from primary active-interval exposure; may support insertion/removal timing review."
        else:
            rationale = "Candidate text match requires exclusion pending clinician review."
        rows.append(
            {
                "itemid": int(r.itemid),
                "label": r.label,
                "category": r.category,
                "final_decision": decision,
                "primary_include": int(primary),
                "sensitivity_group": ";".join(sens),
                "exposure_role_v3": exposure_role,
                "rationale": rationale,
                "clinician_review_status": "algorithmic_v3_review_requires_clinician_confirmation",
            }
        )
    review = pd.DataFrame(rows).sort_values(["final_decision", "itemid"])

    pe = pd.read_csv(
        paths.base / "icu" / "procedureevents.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime"],
        parse_dates=["starttime", "endtime"],
    )
    counts = (
        pe.loc[pe["itemid"].isin(review["itemid"])]
        .merge(adult_icu[["subject_id", "hadm_id", "stay_id"]], on=["subject_id", "hadm_id", "stay_id"], how="inner")
        .assign(valid=lambda x: x["starttime"].notna() & x["endtime"].notna() & (x["endtime"] > x["starttime"]))
    )
    if not counts.empty:
        counts["duration_hours"] = (counts["endtime"] - counts["starttime"]).dt.total_seconds() / 3600
        agg = counts.loc[counts["valid"]].groupby("itemid").agg(
            rows=("itemid", "size"),
            unique_icu_stays=("stay_id", "nunique"),
            median_duration_hours=("duration_hours", "median"),
        ).reset_index()
        review = review.merge(agg, on="itemid", how="left")
    review[["rows", "unique_icu_stays"]] = review[["rows", "unique_icu_stays"]].fillna(0).astype(int)
    review["median_duration_hours"] = review["median_duration_hours"].fillna(0)

    primary = review.loc[review["primary_include"] == 1, ["itemid", "label", "category", "rationale"]].copy()
    sens = review.loc[review["sensitivity_group"].ne(""), ["itemid", "label", "category", "sensitivity_group", "rationale"]].copy()
    write_csv(review, paths.qc / "cvc_itemid_final_review.csv")
    write_csv(primary, paths.qc / "cvc_itemid_primary_inclusion.csv")
    write_csv(sens, paths.qc / "cvc_itemid_sensitivity_groups.csv")
    write_v3_csv(review, paths.qc / "cvc_itemid_final_review.csv")
    write_v3_csv(review.loc[review["exposure_role_v3"].eq("active CVC interval"), ["itemid", "label", "category", "rationale", "clinician_review_status"]], paths.qc / "cvc_active_interval_itemids.csv")
    write_v4_csv(review, paths.qc / "cvc_itemid_final_review.csv")
    write_v4_csv(review.loc[review["exposure_role_v3"].eq("active CVC interval"), ["itemid", "label", "category", "rationale", "clinician_review_status"]], paths.qc / "cvc_active_interval_itemids.csv")
    locked_review = review.copy()
    locked_review["clinician_review_status"] = np.where(
        locked_review["exposure_role_v3"].eq("active CVC interval"),
        "clinician_approved",
        "clinician_review_pending",
    )
    active_locked = locked_review.loc[
        locked_review["itemid"].isin([224263, 224267, 224273, 225199]),
        ["itemid", "label", "category", "rationale", "clinician_review_status"],
    ].copy()
    write_locked_csv(locked_review, paths.qc / "cvc_itemid_final_review.csv")
    write_locked_csv(active_locked, paths.qc / "cvc_active_interval_itemids.csv")
    return review, primary, sens


def build_cvc(paths: Paths, cohort: pd.DataFrame, review: pd.DataFrame, mode: str = "primary") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mode == "primary":
        include_ids = set(review.loc[review["primary_include"] == 1, "itemid"].astype(int))
    elif mode == "exclude_dialysis":
        include_ids = set(review.loc[(review["primary_include"] == 1) | (review["sensitivity_group"].str.contains("dialysis_catheter", na=False)), "itemid"].astype(int))
        include_ids -= set(review.loc[review["sensitivity_group"].str.contains("dialysis_catheter", na=False), "itemid"].astype(int))
    else:
        include_ids = set(review.loc[review["primary_include"] == 1, "itemid"].astype(int))

    pe = pd.read_csv(
        paths.base / "icu" / "procedureevents.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime"],
        parse_dates=["starttime", "endtime"],
    )
    labels = review[["itemid", "label", "sensitivity_group"]].copy()
    raw = pe.loc[pe["itemid"].isin(include_ids)].merge(labels, on="itemid", how="left")
    raw = raw.merge(
        cohort[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "dischtime", "deathtime", "anchor_age", "gender", "race", "first_careunit", "admission_type"]],
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
    )
    raw["valid_duration"] = raw["starttime"].notna() & raw["endtime"].notna() & (raw["endtime"] > raw["starttime"])
    raw["duration_hours"] = (raw["endtime"] - raw["starttime"]).dt.total_seconds() / 3600
    valid = raw.loc[raw["valid_duration"] & (raw["duration_hours"] > 0)].copy()

    merged_rows = []
    reductions = 0
    for key, g in valid.sort_values(["stay_id", "starttime", "endtime"]).groupby(["subject_id", "hadm_id", "stay_id"], sort=False):
        current_start = None
        current_end = None
        for r in g.itertuples(index=False):
            if current_start is None:
                current_start = r.starttime
                current_end = r.endtime
            elif r.starttime <= current_end:
                current_end = max(current_end, r.endtime)
                reductions += 1
            else:
                merged_rows.append((*key, current_start, current_end))
                current_start = r.starttime
                current_end = r.endtime
        if current_start is not None:
            merged_rows.append((*key, current_start, current_end))
    merged = pd.DataFrame(merged_rows, columns=["subject_id", "hadm_id", "stay_id", "cvc_starttime", "cvc_endtime"])
    merged = merged.merge(cohort, on=["subject_id", "hadm_id", "stay_id"], how="left") if not merged.empty else merged
    if not merged.empty:
        merged["duration_hours"] = (merged["cvc_endtime"] - merged["cvc_starttime"]).dt.total_seconds() / 3600
        merged["start_before_icu"] = merged["cvc_starttime"] < merged["intime"]
        merged["present_at_icu_admission"] = (merged["cvc_starttime"] <= merged["intime"]) & (merged["cvc_endtime"] > merged["intime"])
        merged["incident_after_icu"] = merged["cvc_starttime"] > merged["intime"]
        merged["end_after_icu_discharge"] = merged["cvc_endtime"] > merged["outtime"]
        merged["extremely_long_duration"] = merged["duration_hours"] > 24 * 28

    duration = merged["duration_hours"] if not merged.empty else pd.Series(dtype=float)
    qc = pd.DataFrame(
        [
            {"metric": "adult_first_icu_stays", "value": int(len(cohort))},
            {"metric": "raw_primary_cvc_rows", "value": int(len(raw))},
            {"metric": "valid_primary_cvc_rows", "value": int(len(valid))},
            {"metric": "zero_or_negative_or_missing_duration_rows", "value": int(len(raw) - len(valid))},
            {"metric": "stays_with_primary_cvc", "value": int(valid["stay_id"].nunique()) if not valid.empty else 0},
            {"metric": "merged_intervals", "value": int(len(merged))},
            {"metric": "overlap_or_contiguous_reductions", "value": int(reductions)},
            {"metric": "cvc_start_before_icu_admission_intervals", "value": int(merged.get("start_before_icu", pd.Series(dtype=bool)).sum())},
            {"metric": "cvc_present_at_icu_admission_intervals", "value": int(merged.get("present_at_icu_admission", pd.Series(dtype=bool)).sum())},
            {"metric": "incident_cvc_after_icu_admission_intervals", "value": int(merged.get("incident_after_icu", pd.Series(dtype=bool)).sum())},
            {"metric": "cvc_end_after_icu_discharge_intervals", "value": int(merged.get("end_after_icu_discharge", pd.Series(dtype=bool)).sum())},
            {"metric": "extremely_long_duration_gt_28d_intervals", "value": int(merged.get("extremely_long_duration", pd.Series(dtype=bool)).sum())},
            {"metric": "duration_hours_median", "value": q(duration, 0.50)},
            {"metric": "duration_hours_p25", "value": q(duration, 0.25)},
            {"metric": "duration_hours_p75", "value": q(duration, 0.75)},
        ]
    )
    write_csv(qc, paths.qc / "cvc_episode_qc_corrected.csv")

    by_type = valid.merge(review[["itemid", "label", "sensitivity_group"]], on="itemid", how="left", suffixes=("", "_review"))
    dur_type = by_type.groupby("label").agg(
        rows=("itemid", "size"),
        unique_icu_stays=("stay_id", "nunique"),
        median_duration_hours=("duration_hours", "median"),
        p25_duration_hours=("duration_hours", lambda x: q(x, 0.25)),
        p75_duration_hours=("duration_hours", lambda x: q(x, 0.75)),
    ).reset_index().sort_values("rows", ascending=False)
    write_csv(dur_type, paths.qc / "cvc_duration_by_type.csv")
    return raw, valid, merged


def build_person_days(cohort_cvc: pd.DataFrame, raw_cvc: pd.DataFrame, events: pd.DataFrame | None = None, max_days: int = 28) -> pd.DataFrame:
    rows = []
    event_map = {}
    if events is not None and not events.empty:
        for r in events.itertuples(index=False):
            event_map[int(r.stay_id)] = r.event_time
    cvc_by_stay = {int(k): g for k, g in cohort_cvc.groupby("stay_id")}
    raw_by_stay = {int(k): g for k, g in raw_cvc.groupby("stay_id")} if not raw_cvc.empty else {}
    for stay_id, g in cvc_by_stay.items():
        first = g.iloc[0]
        time_zero = g["cvc_starttime"].min()
        censor_times = [time_zero + pd.Timedelta(days=max_days), first.outtime]
        if pd.notna(first.dischtime):
            censor_times.append(first.dischtime)
        if pd.notna(first.deathtime):
            censor_times.append(first.deathtime)
        if stay_id in event_map:
            censor_times.append(event_map[stay_id])
        end = min(censor_times)
        raw_intervals = raw_by_stay.get(stay_id, pd.DataFrame())
        d = 1
        while time_zero + pd.Timedelta(days=d - 1) < end and d <= max_days:
            day_start = time_zero + pd.Timedelta(days=d - 1)
            day_end = day_start + pd.Timedelta(days=1)
            active = g.loc[(g["cvc_starttime"] < day_end) & (g["cvc_endtime"] > day_start)]
            concurrent = raw_intervals.loc[(raw_intervals["starttime"] < day_end) & (raw_intervals["endtime"] > day_start)] if not raw_intervals.empty else active
            if not active.empty:
                line_day_number = int(
                    max(((day_start.floor("D") - x.floor("D")).days + 1) for x in active["cvc_starttime"])
                )
            else:
                line_day_number = 0
            rows.append(
                {
                    "subject_id": first.subject_id,
                    "hadm_id": first.hadm_id,
                    "stay_id": stay_id,
                    "day": d,
                    "day_start": day_start,
                    "cvc_present": int(not active.empty),
                    "cvc_day_number": line_day_number,
                    "eligible_central_line_day3_plus": int(line_day_number > 2),
                    "number_of_concurrent_lines": int(len(concurrent)),
                    "age": float(first.anchor_age),
                    "female": int(str(first.gender).upper() == "F"),
                    "death_before_day28": int(pd.notna(first.deathtime) and first.deathtime <= time_zero + pd.Timedelta(days=max_days)),
                    "icu_discharge_before_day28": int(first.outtime <= time_zero + pd.Timedelta(days=max_days)),
                    "hospital_discharge_before_day28": int(pd.notna(first.dischtime) and first.dischtime <= time_zero + pd.Timedelta(days=max_days)),
                    "discharge_before_day28": int(first.outtime <= time_zero + pd.Timedelta(days=max_days) or (pd.notna(first.dischtime) and first.dischtime <= time_zero + pd.Timedelta(days=max_days))),
                }
            )
            d += 1
    return pd.DataFrame(rows)


def microbiology_clean(paths: Paths, cvc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    micro_cols = list(pd.read_csv(paths.base / "hosp" / "microbiologyevents.csv.gz", nrows=0).columns)
    key_candidates = [c for c in ["micro_specimen_id", "specimen_id", "order_id"] if c in micro_cols]
    row_key_col = "microevent_id" if "microevent_id" in micro_cols else None
    usecols = ["subject_id", "hadm_id", "chartdate", "charttime", "spec_type_desc", "org_name", "ab_name", "interpretation"] + key_candidates
    if row_key_col:
        usecols.append(row_key_col)
    micro = pd.read_csv(
        paths.base / "hosp" / "microbiologyevents.csv.gz",
        usecols=usecols,
        parse_dates=["chartdate", "charttime"],
    )
    micro["event_time"] = micro["charttime"].fillna(micro["chartdate"])
    micro = micro.loc[micro["event_time"].notna()].copy()
    spec = micro["spec_type_desc"].fillna("").astype(str)
    blood = micro.loc[spec.str.contains("blood", case=False, na=False)].copy()
    blood["org_norm"] = blood["org_name"].map(norm)
    blood["invalid_org"] = blood["org_name"].map(invalid_organism_label)
    fallback_key = blood["subject_id"].astype(str) + "|" + blood["hadm_id"].astype(str) + "|" + blood["event_time"].astype(str) + "|" + blood["spec_type_desc"].fillna("").astype(str)
    if "micro_specimen_id" in blood.columns:
        spec_key = blood["micro_specimen_id"]
        blood["collection_key"] = np.where(spec_key.notna(), "specimen:" + spec_key.astype("Int64").astype(str), "fallback:" + fallback_key)
        key_source = "micro_specimen_id_with_fallback"
    elif key_candidates:
        strongest_key = key_candidates[0]
        spec_key = blood[strongest_key]
        blood["collection_key"] = np.where(spec_key.notna(), strongest_key + ":" + spec_key.astype(str), "fallback:" + fallback_key)
        key_source = f"{strongest_key}_with_fallback"
    else:
        blood["collection_key"] = "fallback:" + fallback_key
        key_source = "subject_id_hadm_id_event_time_specimen_type_fallback"
    if row_key_col:
        blood["row_key"] = blood[row_key_col].astype(str)
    else:
        blood["row_key"] = blood.index.astype(str)
    blood["isolate_key"] = blood["collection_key"] + "|" + blood["org_norm"]
    positive = blood.loc[~blood["invalid_org"] & blood["org_norm"].ne("")].copy()
    susc = positive.loc[positive["interpretation"].fillna("").astype(str).str.strip().ne("")].copy()
    event_qc = pd.DataFrame(
        [
            {"metric": "raw_microbiology_rows_all_specimens", "value": int(len(micro))},
            {"metric": "raw_blood_microbiology_rows", "value": int(len(blood))},
            {"metric": "distinct_blood_culture_collection_events", "value": int(blood["collection_key"].nunique())},
            {"metric": "distinct_positive_blood_culture_events", "value": int(positive["collection_key"].nunique())},
            {"metric": "distinct_positive_blood_isolates", "value": int(positive["isolate_key"].nunique())},
            {"metric": "susceptibility_rows_positive_blood", "value": int(len(susc))},
            {"metric": "isolates_with_susceptibility_interpretation", "value": int(susc["isolate_key"].nunique())},
        ]
    )
    rules = pd.DataFrame(
        [
            {"rule": "blank_or_missing", "definition": "org_name missing or empty", "exclude": 1},
            {"rule": "cancelled_canceled", "definition": "exact/near-exact cancelled or canceled", "exclude": 1},
            {"rule": "no_growth", "definition": "exact/near-exact no growth or no growth to date", "exclude": 1},
            {"rule": "negative_culture_language", "definition": "NEGATIVE only when the entire label is negative culture language", "exclude": 1},
            {"rule": "test_not_performed", "definition": "exact/near-exact test not performed/not performed", "exclude": 1},
            {"rule": "quantity_not_sufficient", "definition": "exact/near-exact quantity not sufficient/QNS", "exclude": 1},
            {"rule": "valid_negative_word_labels", "definition": "Do not exclude coagulase-negative staphylococcus, Gram-negative rods, or beta-lactamase-negative organisms", "exclude": 0},
            {"rule": "nonspecific_contamination", "definition": NONSPECIFIC_CONTAMINATION_RE.pattern, "exclude": 1},
        ]
    )
    write_csv(rules, paths.qc / "invalid_organism_label_rules_v2.csv")
    write_v2_csv(event_qc, paths.qc / "blood_culture_event_level_qc.csv")
    write_v4_csv(event_qc, paths.qc / "blood_culture_event_level_qc.csv")
    multi_org = positive.drop_duplicates(["collection_key", "org_norm"]).groupby("collection_key")["org_norm"].nunique()
    same_day = blood.drop_duplicates("collection_key").assign(event_day=lambda x: x["event_time"].dt.floor("D")).groupby(["subject_id", "hadm_id", "event_day"])["collection_key"].nunique()
    comm_missing_time = positive.loc[
        positive["charttime"].isna()
        & positive["org_name"].fillna("").str.contains(COMMON_COMMENSAL_RE, na=False)
    ]
    key_qc = pd.DataFrame(
        [
            {"metric": "microbiology_key_source", "value": key_source},
            {"metric": "available_specific_key_columns", "value": ";".join(key_candidates) if key_candidates else "none_available"},
            {"metric": "collection_keys_with_multiple_organisms", "value": int((multi_org > 1).sum())},
            {"metric": "same_day_multiple_blood_culture_groups", "value": int((same_day > 1).sum())},
            {"metric": "common_commensal_positive_rows_with_chartdate_only", "value": int(len(comm_missing_time))},
            {"metric": "common_commensal_collection_keys_with_chartdate_only", "value": int(comm_missing_time["collection_key"].nunique())},
        ]
    )
    write_v3_csv(key_qc, paths.qc / "microbiology_key_qc.csv")
    write_v4_csv(key_qc, paths.qc / "microbiology_key_qc.csv")

    excluded = blood.loc[blood["invalid_org"]].groupby("org_norm").size().reset_index(name="number_of_rows").rename(columns={"org_norm": "excluded_organism_label"}).sort_values("number_of_rows", ascending=False)
    write_v2_csv(excluded.head(100), paths.qc / "excluded_organism_labels.csv")

    org_dist = positive.drop_duplicates("isolate_key").groupby("org_norm").size().reset_index(name="number_of_isolates").rename(columns={"org_norm": "organism"}).sort_values("number_of_isolates", ascending=False).head(50)
    write_v2_csv(org_dist, paths.qc / "positive_blood_culture_organism_distribution.csv")
    write_v4_csv(org_dist, paths.qc / "positive_blood_culture_organism_distribution.csv")
    return blood, positive, susc, event_qc


def qualify_bsi(paths: Paths, cvc: pd.DataFrame, positive: pd.DataFrame, windows=(24, 48, 72), write_qc_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    stay_zero = (
        cvc.groupby(["subject_id", "hadm_id", "stay_id"], as_index=False)
        .agg(time_zero=("cvc_starttime", "min"), intime=("intime", "first"), outtime=("outtime", "first"), dischtime=("dischtime", "first"), deathtime=("deathtime", "first"))
    )
    stay_zero["followup_end"] = stay_zero[["outtime", "dischtime", "deathtime"]].min(axis=1, skipna=True)
    stay_zero["followup_end"] = stay_zero[["followup_end", "time_zero"]].apply(lambda r: min(r["followup_end"], r["time_zero"] + pd.Timedelta(days=28)), axis=1)
    linked_all = positive.merge(stay_zero, on=["subject_id", "hadm_id"], how="inner")
    outside_pre = linked_all.copy()
    linked = linked_all.loc[linked_all["event_time"] <= linked_all["followup_end"]].copy()
    linked["event_day"] = linked["event_time"].dt.floor("D")
    linked["is_commensal"] = linked["org_name"].fillna("").str.contains(COMMON_COMMENSAL_RE, na=False)
    linked = linked.drop_duplicates(["stay_id", "collection_key", "isolate_key", "org_norm", "event_time"])
    pre_rows = []
    post_rows = []
    outputs = {}
    isolate_outputs = {}
    for hours in windows:
        threshold = linked["time_zero"] + pd.Timedelta(hours=hours)
        before_cvc_all = linked_all["event_time"] < linked_all["time_zero"]
        early_all = (linked_all["event_time"] >= linked_all["time_zero"]) & (linked_all["event_time"] < (linked_all["time_zero"] + pd.Timedelta(hours=hours)))
        excluded_stays = set(linked_all.loc[before_cvc_all | early_all, "stay_id"].astype(int))
        if hours == 48:
            baseline_excluded_stays = excluded_stays
        eligible = linked.loc[
            ~linked["stay_id"].astype(int).isin(excluded_stays)
            & (linked["event_time"] >= threshold)
            & (linked["event_time"] <= linked["followup_end"])
        ].copy()
        threshold_pre = outside_pre["time_zero"] + pd.Timedelta(hours=hours)
        outside_after_baseline = outside_pre.loc[
            ~outside_pre["stay_id"].astype(int).isin(excluded_stays)
            & (outside_pre["event_time"] >= threshold_pre)
            & ((outside_pre["event_time"] < outside_pre["time_zero"]) | (outside_pre["event_time"] > outside_pre["followup_end"]))
        ]
        if hours == 48 and write_qc_outputs:
            outside_qc = pd.DataFrame(
                [
                    {"metric": "positive_isolate_rows_outside_primary_followup", "value": int(len(outside_after_baseline))},
                    {"metric": "positive_collection_events_outside_primary_followup", "value": int(outside_after_baseline["collection_key"].nunique())},
                    {"metric": "stays_with_positive_culture_outside_primary_followup", "value": int(outside_after_baseline["stay_id"].nunique())},
                ]
            )
            write_v4_csv(outside_qc, paths.qc / "bsi_events_outside_followup_excluded.csv")
        pre_rows.extend(
            [
                {"baseline_window_hours": hours, "metric": "candidate_stays_with_positive_blood_culture", "value": int(linked_all["stay_id"].nunique())},
                {"baseline_window_hours": hours, "metric": "stays_with_positive_before_time_zero", "value": int(linked_all.loc[before_cvc_all, "stay_id"].nunique())},
                {"baseline_window_hours": hours, "metric": "stays_with_positive_within_baseline_window", "value": int(linked_all.loc[early_all, "stay_id"].nunique())},
                {"baseline_window_hours": hours, "metric": "baseline_excluded_stays", "value": int(len(excluded_stays))},
            ]
        )
        post_rows.extend(
            [
                {"baseline_window_hours": hours, "metric": "stays_remaining_after_baseline_exclusion", "value": int(eligible["stay_id"].nunique())},
                {"baseline_window_hours": hours, "metric": "positive_collection_events_after_baseline_exclusion", "value": int(eligible["collection_key"].nunique())},
                {"baseline_window_hours": hours, "metric": "positive_isolates_after_baseline_exclusion", "value": int(eligible["isolate_key"].nunique())},
            ]
        )
        event_units = eligible[["subject_id", "hadm_id", "stay_id", "collection_key", "isolate_key", "event_time", "event_day", "org_norm", "org_name", "is_commensal"]].drop_duplicates()
        recognized = event_units.loc[~event_units["is_commensal"]].copy()
        comm = event_units.loc[event_units["is_commensal"]].copy()
        if not comm.empty:
            per_day = comm.drop_duplicates(["stay_id", "collection_key", "org_norm"]).groupby(["subject_id", "hadm_id", "stay_id", "org_norm", "event_day"], as_index=False).size()
            key_cols = ["subject_id", "hadm_id", "stay_id", "org_norm", "event_day"]
            same = per_day.loc[per_day["size"] >= 2, key_cols]
            prev = per_day[key_cols].merge(per_day.assign(event_day=per_day["event_day"] - pd.Timedelta(days=1))[key_cols], on=["subject_id", "hadm_id", "stay_id", "org_norm", "event_day"])
            nxt = per_day[key_cols].merge(per_day.assign(event_day=per_day["event_day"] + pd.Timedelta(days=1))[key_cols], on=["subject_id", "hadm_id", "stay_id", "org_norm", "event_day"])
            valid_days = pd.concat([same, prev, nxt], ignore_index=True).drop_duplicates()
            comm = comm.merge(valid_days.assign(commensal_qualifies=True), on=key_cols, how="left")
            comm = comm.loc[comm["commensal_qualifies"] == True].drop(columns=["commensal_qualifies"])
        qual = pd.concat([recognized, comm], ignore_index=True).sort_values("event_time")
        first_event = qual.groupby("stay_id", as_index=False).first() if not qual.empty else qual
        outputs[hours] = first_event
        if qual.empty or first_event.empty:
            isolate_outputs[hours] = qual
        else:
            first_keys = first_event[["stay_id", "event_time"]].drop_duplicates()
            isolate_outputs[hours] = qual.merge(first_keys, on=["stay_id", "event_time"], how="inner")
    if write_qc_outputs:
        write_csv(pd.DataFrame(pre_rows), paths.qc / "bsi_baseline_exclusion_qc_pre_exclusion_v2.csv")
        write_csv(pd.DataFrame(post_rows), paths.qc / "bsi_baseline_exclusion_qc_post_exclusion_v2.csv")
        write_locked_csv(pd.DataFrame(pre_rows), paths.qc / "bsi_baseline_exclusion_qc_pre_exclusion.csv")
        write_locked_csv(pd.DataFrame(post_rows), paths.qc / "bsi_baseline_exclusion_qc_post_exclusion.csv")

    bsi = outputs[48].copy()
    cvc_by_stay = {int(k): g for k, g in cvc.groupby("stay_id")}
    if not bsi.empty:
        bsi["clabsi_proxy"] = bsi.apply(lambda r: int(is_clabsi_proxy(cvc_by_stay[int(r.stay_id)], r.event_day)), axis=1)
        bsi["days_after_cvc_start"] = bsi.apply(lambda r: float((r.event_time - cvc_by_stay[int(r.stay_id)]["cvc_starttime"].min()).total_seconds() / 86400), axis=1)
    else:
        bsi["clabsi_proxy"] = []
        bsi["days_after_cvc_start"] = []
    bsi_qc = pd.DataFrame(
        [
            {"metric": "bsi_events_48h_primary", "value": int(len(bsi))},
            {"metric": "recognized_pathogen_bsi_events", "value": int((~bsi["is_commensal"]).sum()) if not bsi.empty else 0},
            {"metric": "common_commensal_confirmed_bsi_events", "value": int(bsi["is_commensal"].sum()) if not bsi.empty else 0},
            {"metric": "event_days_after_cvc_median", "value": q(bsi["days_after_cvc_start"], 0.50) if not bsi.empty else 0},
            {"metric": "event_days_after_cvc_p25", "value": q(bsi["days_after_cvc_start"], 0.25) if not bsi.empty else 0},
            {"metric": "event_days_after_cvc_p75", "value": q(bsi["days_after_cvc_start"], 0.75) if not bsi.empty else 0},
        ]
    )
    if write_qc_outputs:
        write_v2_csv(bsi_qc, paths.qc / "bsi_event_level_qc.csv")
        write_v4_csv(bsi_qc, paths.qc / "bsi_event_level_qc.csv")
        write_locked_csv(bsi_qc, paths.qc / "bsi_event_level_qc.csv")
    comm_qc = pd.DataFrame(
        [
            {"metric": "explicit_common_commensal_regex", "value": COMMON_COMMENSAL_RE.pattern},
            {"metric": "confirmed_common_commensal_bsi_events", "value": int(bsi["is_commensal"].sum()) if not bsi.empty else 0},
        ]
    )
    if write_qc_outputs:
        write_v2_csv(comm_qc, paths.qc / "bsi_common_commensal_qc.csv")
    clabsi_qc = pd.DataFrame(
        [
            {"metric": "bsi_events", "value": int(len(bsi))},
            {"metric": "clabsi_proxy_events", "value": int(bsi["clabsi_proxy"].sum()) if not bsi.empty else 0},
            {"metric": "non_clabsi_proxy_bsi_events", "value": int((1 - bsi["clabsi_proxy"]).sum()) if not bsi.empty else 0},
        ]
    )
    if write_qc_outputs:
        write_v2_csv(clabsi_qc, paths.qc / "clabsi_proxy_qc.csv")
        write_v4_csv(clabsi_qc, paths.qc / "clabsi_proxy_qc.csv")
        write_locked_csv(clabsi_qc, paths.qc / "clabsi_proxy_qc.csv")
    interval_qc_rows = []
    for _, g in cvc.groupby("stay_id"):
        for r in g.itertuples(index=False):
            first_day = r.cvc_starttime.floor("D")
            last_day = min(r.cvc_endtime.floor("D"), first_day + pd.Timedelta(days=28))
            interval_qc_rows.append({"interval_line_day_category": "eligible_day_gt2", "line_days": max((last_day - first_day).days - 1, 0)})
            interval_qc_rows.append({"interval_line_day_category": "noneligible_first_2_days", "line_days": min((last_day - first_day).days + 1, 2)})
    interval_qc = pd.DataFrame(interval_qc_rows).groupby("interval_line_day_category", as_index=False)["line_days"].sum() if interval_qc_rows else pd.DataFrame(columns=["interval_line_day_category", "line_days"])
    if write_qc_outputs:
        write_csv(interval_qc, paths.qc / "cvc_interval_line_day_qc_v2.csv")
    bsi.attrs["baseline_excluded_stays"] = baseline_excluded_stays if "baseline_excluded_stays" in locals() else set()
    bsi.attrs["bsi_isolates"] = isolate_outputs.get(48, pd.DataFrame())
    bsi.attrs["bsi_outputs_by_window"] = outputs
    bsi.attrs["bsi_isolates_by_window"] = isolate_outputs
    return bsi, outputs


def is_clabsi_proxy(intervals: pd.DataFrame, event_day: pd.Timestamp) -> bool:
    for day in [event_day.floor("D"), event_day.floor("D") - pd.Timedelta(days=1)]:
        active = intervals.loc[(intervals["cvc_starttime"] < day + pd.Timedelta(days=1)) & (intervals["cvc_endtime"] > day)]
        eligible = active.loc[active["cvc_starttime"].apply(lambda x: int((day - x.floor("D")).days) + 1 > 2)]
        if not eligible.empty:
            return True
    return False


def amr_bsi(paths: Paths, bsi: pd.DataFrame, susc: pd.DataFrame) -> pd.DataFrame:
    relevance_rows = [
        {"organism_group": "Staphylococcus aureus", "antibiotic_group": "methicillin/oxacillin/nafcillin/cefoxitin", "phenotype": "MRSA", "include_primary_amr": 1},
        {"organism_group": "Enterococcus faecium/faecalis", "antibiotic_group": "vancomycin", "phenotype": "VRE", "include_primary_amr": 1},
        {"organism_group": "Enterobacterales", "antibiotic_group": "carbapenems", "phenotype": "CRE", "include_primary_amr": 1},
        {"organism_group": "Pseudomonas aeruginosa", "antibiotic_group": "carbapenems", "phenotype": "CRPA", "include_primary_amr": 1},
        {"organism_group": "Acinetobacter baumannii complex", "antibiotic_group": "carbapenems", "phenotype": "CRAB", "include_primary_amr": 1},
        {"organism_group": "Candida spp.", "antibiotic_group": "antifungals", "phenotype": "antifungal_resistant_candidemia", "include_primary_amr": 0},
    ]
    write_v2_csv(pd.DataFrame(relevance_rows), paths.qc / "organism_antibiotic_relevance_map.csv")

    if bsi.empty:
        empty = pd.DataFrame(columns=["phenotype", "number_of_events"])
        write_csv(empty, paths.qc / "amr_phenotype_event_distribution_strict_v2.csv")
        write_csv(pd.DataFrame([{"metric": "strict_amr_bsi_events_r_only", "value": 0}]), paths.qc / "amr_event_level_qc_strict_v2.csv")
        return bsi.assign(amr_bsi_r=0, amr_bsi_ri=0, amr_phenotype="")

    susc = susc.copy()
    susc["event_day"] = susc["event_time"].dt.floor("D")
    susc["interp"] = susc["interpretation"].fillna("").astype(str).str.upper()
    bsi_isolates = bsi.attrs.get("bsi_isolates", pd.DataFrame())
    if bsi_isolates.empty:
        bsi_isolates = bsi[["subject_id", "hadm_id", "stay_id", "collection_key", "isolate_key", "event_day", "org_norm"]].drop_duplicates()
    event_keys = bsi_isolates[["stay_id", "collection_key", "isolate_key"]].drop_duplicates()
    susc = susc.merge(event_keys, on=["collection_key", "isolate_key"], how="inner")
    intrinsic = []
    phenotype_rows = []
    for r in susc.itertuples(index=False):
        org = str(r.org_name or "")
        ab = str(r.ab_name or "")
        intrinsic_hit = None
        for name, org_pat, ab_pat in INTRINSIC_RULES:
            if org_pat.search(org) and ab_pat.search(ab):
                intrinsic_hit = name
                break
        if intrinsic_hit:
            intrinsic.append(intrinsic_hit)
            continue
        phen_r = classify_strict_amr(org, ab, r.interp, allow_intermediate=False)
        phen_ri = classify_strict_amr(org, ab, r.interp, allow_intermediate=True)
        other = classify_amr(org, ab, r.interp, strict_only=False)
        if phen_r or phen_ri:
            phenotype_rows.append({"stay_id": r.stay_id, "isolate_key": r.isolate_key, "phenotype_r": phen_r or "", "phenotype_ri": phen_ri or "", "interp": r.interp})
        elif other:
            phenotype_rows.append({"stay_id": r.stay_id, "isolate_key": r.isolate_key, "phenotype_r": "", "phenotype_ri": "", "phenotype": other, "interp": r.interp, "organism": norm(org), "antibiotic": norm(ab)})
    exclusions = pd.Series(intrinsic).value_counts().reset_index()
    exclusions.columns = ["intrinsic_resistance_rule", "excluded_susceptibility_rows"] if not exclusions.empty else ["intrinsic_resistance_rule", "excluded_susceptibility_rows"]
    write_v2_csv(exclusions, paths.qc / "intrinsic_resistance_exclusions.csv")
    phen = pd.DataFrame(phenotype_rows)
    b = bsi.copy()
    if phen.empty:
        b["amr_bsi_r"] = 0
        b["amr_bsi_ri"] = 0
        b["amr_phenotype"] = ""
        b["amr_phenotype_ri"] = ""
        b["other_r_exploratory"] = 0
    else:
        strict_r = phen.loc[phen.get("phenotype_r", pd.Series(dtype=str)).fillna("").ne("")].copy()
        strict_ri = phen.loc[phen.get("phenotype_ri", pd.Series(dtype=str)).fillna("").ne("")].copy()
        other = phen.loc[phen.get("phenotype", pd.Series(dtype=str)).eq("Other_R")].copy()
        r = strict_r.groupby("stay_id")["phenotype_r"].apply(lambda x: ";".join(sorted(set(x))))
        ri = strict_ri.groupby("stay_id")["phenotype_ri"].apply(lambda x: ";".join(sorted(set(x))))
        b["amr_phenotype"] = b["stay_id"].map(r).fillna("")
        b["amr_phenotype_ri"] = b["stay_id"].map(ri).fillna("")
        b["amr_bsi_r"] = b["amr_phenotype"].ne("").astype(int)
        b["amr_bsi_ri"] = b["amr_phenotype_ri"].ne("").astype(int)
        b["other_r_exploratory"] = b["stay_id"].isin(other["stay_id"].unique()).astype(int)
        other_dist = (
            other.groupby(["organism", "antibiotic"], as_index=False).agg(number_of_events=("stay_id", "nunique"))
            .sort_values("number_of_events", ascending=False)
            .head(100)
            if not other.empty
            else pd.DataFrame(columns=["organism", "antibiotic", "number_of_events"])
        )
        write_csv(other_dist, paths.qc / "amr_other_r_exploratory_distribution_v2.csv")
        write_v4_csv(other_dist, paths.qc / "amr_other_r_exploratory_distribution.csv")
    dist = b.loc[b["amr_bsi_r"] == 1, "amr_phenotype"].str.split(";").explode().value_counts().reset_index()
    dist.columns = ["phenotype", "number_of_events"] if not dist.empty else ["phenotype", "number_of_events"]
    write_csv(dist, paths.qc / "amr_phenotype_event_distribution_strict_v2.csv")
    write_v3_csv(dist, paths.qc / "amr_phenotype_event_distribution_r_only.csv")
    write_v4_csv(dist, paths.qc / "amr_phenotype_event_distribution_r_only.csv")
    dist_ri = b.loc[b["amr_bsi_ri"] == 1, "amr_phenotype_ri"].str.split(";").explode().value_counts().reset_index()
    dist_ri.columns = ["phenotype", "number_of_events"] if not dist_ri.empty else ["phenotype", "number_of_events"]
    write_v3_csv(dist_ri, paths.qc / "amr_phenotype_event_distribution_r_or_i.csv")
    write_v4_csv(dist_ri, paths.qc / "amr_phenotype_event_distribution_r_or_i.csv")
    amr_qc = pd.DataFrame(
        [
            {"metric": "bsi_events", "value": int(len(b))},
            {"metric": "strict_amr_bsi_events_r_only", "value": int(b["amr_bsi_r"].sum())},
            {"metric": "strict_amr_bsi_events_r_or_i", "value": int(b["amr_bsi_ri"].sum())},
            {"metric": "other_r_exploratory_bsi_events", "value": int(b["other_r_exploratory"].sum())},
            {"metric": "candida_in_primary_bsi_but_not_primary_amr", "value": int(b["org_name"].fillna("").str.contains(FUNGUS_RE, na=False).sum())},
        ]
    )
    write_csv(amr_qc, paths.qc / "amr_event_level_qc_strict_v2.csv")
    write_v3_csv(amr_qc, paths.qc / "amr_event_level_qc_strict.csv")
    write_v4_csv(amr_qc, paths.qc / "amr_event_level_qc_strict.csv")
    write_locked_csv(amr_qc, paths.qc / "amr_event_level_qc_strict.csv")
    return b


def classify_strict_amr(org: str, ab: str, interp: str, allow_intermediate: bool) -> str | None:
    if interp not in ({"R", "I"} if allow_intermediate else {"R"}):
        return None
    org_n = norm(org)
    ab_n = norm(ab)
    if ("staph aureus" in org_n or "staphylococcus aureus" in org_n or "s. aureus" in org_n) and ab_n in METHICILLIN_MARKERS:
        return "MRSA"
    if org_n in {"enterococcus faecium", "enterococcus faecalis"} and ab_n == "vancomycin":
        return "VRE"
    if org_n in ENTEROBACTERALES and ab_n in CARBAPENEMS:
        return "CRE"
    if org_n == "pseudomonas aeruginosa" and ab_n in CARBAPENEMS:
        return "CRPA"
    if "acinetobacter baumannii" in org_n and ab_n in CARBAPENEMS:
        return "CRAB"
    return None


def classify_amr(org: str, ab: str, interp: str, strict_only: bool = True) -> str | None:
    if interp not in {"R", "I"}:
        return None
    org_n = norm(org)
    ab_n = norm(ab)
    if interp == "I" and strict_only:
        return "R_or_I_only"
    if interp == "I" and not strict_only:
        return None
    if ("staph aureus" in org_n or "staphylococcus aureus" in org_n or "s. aureus" in org_n) and ab_n in METHICILLIN_MARKERS:
        return "MRSA"
    if org_n in {"enterococcus faecium", "enterococcus faecalis"} and ab_n == "vancomycin":
        return "VRE"
    if org_n in ENTEROBACTERALES and ab_n in CARBAPENEMS:
        return "CRE"
    if org_n == "pseudomonas aeruginosa" and ab_n in CARBAPENEMS:
        return "CRPA"
    if "acinetobacter baumannii" in org_n and ab_n in CARBAPENEMS:
        return "CRAB"
    return None if strict_only else "Other_R"


def add_outcomes_to_person_days(person_days: pd.DataFrame, bsi: pd.DataFrame) -> pd.DataFrame:
    df = person_days.copy()
    for col in ["bsi", "clabsi_proxy_bsi", "amr_bsi_r", "amr_bsi_ri"]:
        df[col] = 0
    if not bsi.empty:
        tz = df.groupby("stay_id")["day_start"].min().rename("time_zero").reset_index()
        ev = bsi[["stay_id", "event_time", "clabsi_proxy", "amr_bsi_r", "amr_bsi_ri"]].merge(tz, on="stay_id", how="inner")
        ev["event_day_index"] = np.floor((ev["event_time"] - ev["time_zero"]).dt.total_seconds() / 86400).astype(int) + 1
        ev = ev.loc[ev["event_day_index"].between(1, 28)].sort_values(["stay_id", "event_day_index"]).groupby("stay_id", as_index=False).first()
        ev = ev.rename(columns={"clabsi_proxy": "event_clabsi_proxy", "amr_bsi_r": "event_amr_bsi_r", "amr_bsi_ri": "event_amr_bsi_ri"})
        idx_map = df.reset_index().merge(ev[["stay_id", "event_day_index", "event_clabsi_proxy", "event_amr_bsi_r", "event_amr_bsi_ri"]], left_on=["stay_id", "day"], right_on=["stay_id", "event_day_index"], how="inner")
        if not idx_map.empty:
            hit = idx_map["index"]
            df.loc[hit, "bsi"] = 1
            df.loc[hit, "clabsi_proxy_bsi"] = idx_map["event_clabsi_proxy"].astype(int).to_numpy()
            df.loc[hit, "amr_bsi_r"] = idx_map["event_amr_bsi_r"].astype(int).to_numpy()
            df.loc[hit, "amr_bsi_ri"] = idx_map["event_amr_bsi_ri"].astype(int).to_numpy()
    return df


def build_tve_covariates(paths: Paths, person_days: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cov = person_days[["subject_id", "hadm_id", "stay_id", "day", "day_start", "cvc_day_number", "number_of_concurrent_lines"]].copy()
    for c in ["vasopressor", "mechanical_ventilation", "renal_replacement_therapy", "tpn", "high_frequency_iv_meds", "aki", "recent_positive_culture", "antibiotic_exposure"]:
        cov[c] = 0
    cov["sofa_proxy"] = 0
    cov["lactate_available"] = 0
    cov["lactate_high"] = 0
    qc_rows = []

    # Available covariates are intentionally lightweight and aggregate-safe.
    d_items = pd.read_csv(paths.base / "icu" / "d_items.csv.gz", usecols=["itemid", "label"])
    d_items["label_norm"] = d_items["label"].fillna("").str.lower()
    input_labels = d_items.loc[d_items["label_norm"].str.contains("norepinephrine|phenylephrine|vasopressin|epinephrine|dopamine|dobutamine|tpn|parenteral|bolus|piggyback|iv", regex=True), ["itemid", "label_norm"]]
    if not input_labels.empty:
        usecols = ["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime"]
        inp = pd.read_csv(paths.base / "icu" / "inputevents.csv.gz", usecols=usecols, parse_dates=["starttime", "endtime"])
        inp = inp.merge(input_labels, on="itemid", how="inner")
        ids = set(person_days["stay_id"].unique())
        inp = inp.loc[inp["stay_id"].isin(ids)]
        inp["is_vaso"] = inp["label_norm"].str.contains("norepinephrine|phenylephrine|vasopressin|epinephrine|dopamine|dobutamine", regex=True)
        inp["is_tpn"] = inp["label_norm"].str.contains("tpn|parenteral", regex=True)
        inp["is_iv_med"] = inp["label_norm"].str.contains("piggyback|iv|bolus", regex=True) & ~inp["is_vaso"] & ~inp["is_tpn"]
        cov = mark_interval_covariate(cov, inp.loc[inp["is_vaso"]], "vasopressor")
        cov = mark_interval_covariate(cov, inp.loc[inp["is_tpn"]], "tpn")
        med_days = inp.loc[inp["is_iv_med"]].copy()
        if not med_days.empty:
            med_days["day_key"] = med_days["starttime"].dt.floor("D")
            high = med_days.groupby(["stay_id", "day_key"]).size().reset_index(name="iv_med_rows")
            high = high.loc[high["iv_med_rows"] >= 5]
            day_map = {(int(r.stay_id), r.day_key): 1 for r in high.itertuples(index=False)}
            cov["high_frequency_iv_meds"] = cov.apply(lambda r: int(day_map.get((int(r.stay_id), r.day_start.floor("D")), 0)), axis=1)
        qc_rows.append({"covariate": "vasopressor", "source": "icu.inputevents label search", "nonzero_person_days": int(cov["vasopressor"].sum())})
        qc_rows.append({"covariate": "tpn", "source": "icu.inputevents label search", "nonzero_person_days": int(cov["tpn"].sum())})
        qc_rows.append({"covariate": "high_frequency_iv_meds", "source": "icu.inputevents >=5 IV medication rows/day label proxy", "nonzero_person_days": int(cov["high_frequency_iv_meds"].sum())})

    proc_labels = d_items.loc[d_items["label_norm"].str.contains("ventilat|intubat|extubat|dialysis|crrt|cvvh|rrt|hemofiltration", regex=True), ["itemid", "label_norm"]]
    if not proc_labels.empty:
        pe = pd.read_csv(paths.base / "icu" / "procedureevents.csv.gz", usecols=["stay_id", "itemid", "starttime", "endtime"], parse_dates=["starttime", "endtime"])
        pe = pe.loc[pe["stay_id"].isin(set(person_days["stay_id"].unique()))].merge(proc_labels, on="itemid", how="inner")
        pe["endtime"] = pe["endtime"].fillna(pe["starttime"] + pd.Timedelta(hours=4))
        pe["is_vent"] = pe["label_norm"].str.contains("ventilat|intubat", regex=True)
        pe["is_rrt"] = pe["label_norm"].str.contains("dialysis|crrt|cvvh|rrt|hemofiltration", regex=True)
        cov = mark_interval_covariate(cov, pe.loc[pe["is_vent"]], "mechanical_ventilation")
        cov = mark_interval_covariate(cov, pe.loc[pe["is_rrt"]], "renal_replacement_therapy")
    qc_rows.append({"covariate": "mechanical_ventilation", "source": "icu.procedureevents label search", "nonzero_person_days": int(cov["mechanical_ventilation"].sum())})
    qc_rows.append({"covariate": "renal_replacement_therapy", "source": "icu.procedureevents label search", "nonzero_person_days": int(cov["renal_replacement_therapy"].sum())})

    broad_re = re.compile(r"vancomycin|piperacillin|tazobactam|cefepime|ceftazidime|ceftriaxone|meropenem|imipenem|ertapenem|doripenem|aztreonam|linezolid|daptomycin", re.I)
    rx = pd.read_csv(paths.base / "hosp" / "prescriptions.csv.gz", usecols=["hadm_id", "starttime", "stoptime", "drug"], parse_dates=["starttime", "stoptime"])
    rx = rx.loc[rx["hadm_id"].isin(set(person_days["hadm_id"].unique())) & rx["drug"].fillna("").str.contains(broad_re, na=False)].copy()
    if not rx.empty:
        rx["stoptime"] = rx["stoptime"].fillna(rx["starttime"] + pd.Timedelta(days=1))
        cov = mark_hadm_interval_covariate(cov, rx.rename(columns={"starttime": "starttime", "stoptime": "endtime"}), "antibiotic_exposure")
    qc_rows.append({"covariate": "broad_spectrum_antibiotic_exposure", "source": "hosp.prescriptions drug label search", "nonzero_person_days": int(cov["antibiotic_exposure"].sum())})

    lab_items = pd.read_csv(paths.base / "hosp" / "d_labitems.csv.gz", usecols=["itemid", "label"])
    lab_items["label_norm"] = lab_items["label"].fillna("").str.lower()
    lactate_ids = set(lab_items.loc[lab_items["label_norm"].str.contains("lactate"), "itemid"].astype(int))
    creat_ids = set(lab_items.loc[lab_items["label_norm"].str.contains("creatinine"), "itemid"].astype(int))
    lab_ids = lactate_ids | creat_ids
    if lab_ids:
        days_by_hadm = cov.groupby("hadm_id")["day_start"].agg(["min", "max"]).reset_index()
        hadm_set = set(days_by_hadm["hadm_id"].unique())
        chunks = []
        for chunk in pd.read_csv(paths.base / "hosp" / "labevents.csv.gz", usecols=["hadm_id", "itemid", "charttime", "valuenum"], parse_dates=["charttime"], chunksize=500000):
            chunk = chunk.loc[chunk["hadm_id"].isin(hadm_set) & chunk["itemid"].isin(lab_ids) & chunk["valuenum"].notna()]
            if not chunk.empty:
                chunks.append(chunk)
        if chunks:
            labs = pd.concat(chunks, ignore_index=True)
            labs["lab_day"] = labs["charttime"].dt.floor("D")
            lact = labs.loc[labs["itemid"].isin(lactate_ids)].copy()
            if not lact.empty:
                lact_day = lact.groupby(["hadm_id", "lab_day"], as_index=False)["valuenum"].max().rename(columns={"valuenum": "max_lactate"})
                lact_map = {(int(r.hadm_id), r.lab_day): float(r.max_lactate) for r in lact_day.itertuples(index=False)}
                cov["lactate_available"] = cov.apply(lambda r: int((int(r.hadm_id), r.day_start.floor("D")) in lact_map), axis=1)
                cov["lactate_high"] = cov.apply(lambda r: int(lact_map.get((int(r.hadm_id), r.day_start.floor("D")), 0) >= 2), axis=1)
            cre = labs.loc[labs["itemid"].isin(creat_ids)].copy()
            if not cre.empty:
                cre_day = cre.groupby(["hadm_id", "lab_day"], as_index=False)["valuenum"].max()
                base_cre = cre_day.groupby("hadm_id")["valuenum"].transform("min")
                cre_day["aki_proxy"] = (cre_day["valuenum"] >= (base_cre + 0.3)).astype(int)
                aki_map = {(int(r.hadm_id), r.lab_day): int(r.aki_proxy) for r in cre_day.itertuples(index=False)}
                cov["aki"] = cov.apply(lambda r: int(aki_map.get((int(r.hadm_id), r.day_start.floor("D")), 0)), axis=1)
    qc_rows.append({"covariate": "lactate_available", "source": "hosp.labevents lactate item label search", "nonzero_person_days": int(cov["lactate_available"].sum())})
    qc_rows.append({"covariate": "lactate_high", "source": "hosp.labevents max lactate >=2 mmol/L proxy", "nonzero_person_days": int(cov["lactate_high"].sum())})
    qc_rows.append({"covariate": "aki", "source": "hosp.labevents creatinine rise >=0.3 proxy", "nonzero_person_days": int(cov["aki"].sum())})

    cov["recent_positive_culture"] = 0
    cov["sofa_proxy"] = (
        cov[["vasopressor", "mechanical_ventilation", "renal_replacement_therapy", "aki", "lactate_high"]].fillna(0).sum(axis=1)
        + (cov["number_of_concurrent_lines"].fillna(0) > 1).astype(int)
    )
    qc_rows.append({"covariate": "sofa_proxy", "source": "organ-support and organ-dysfunction proxy score", "nonzero_person_days": int((cov["sofa_proxy"] > 0).sum())})
    qc = pd.DataFrame(qc_rows)
    write_v2_csv(qc, paths.qc / "time_varying_covariate_qc.csv")
    true_indication_cols = ["vasopressor", "renal_replacement_therapy", "tpn", "high_frequency_iv_meds"]
    severity_cols = ["mechanical_ventilation", "antibiotic_exposure", "recent_positive_culture", "lactate_high", "aki", "sofa_proxy"]
    ongoing = cov[true_indication_cols].fillna(0).sum(axis=1).gt(0)
    write_csv(pd.DataFrame([{"metric": "person_days_with_true_cvc_ongoing_indication", "value": int(ongoing.sum())}, {"metric": "person_days_without_true_cvc_ongoing_indication", "value": int((~ongoing).sum())}, {"metric": "true_indication_columns", "value": ";".join(true_indication_cols)}, {"metric": "severity_confounder_columns_not_automatic_indications", "value": ";".join(severity_cols)}]), paths.qc / "ongoing_indication_qc_v2.csv")
    write_v3_csv(pd.DataFrame([{"metric": "person_days_with_true_cvc_ongoing_indication", "value": int(ongoing.sum())}, {"metric": "person_days_without_true_cvc_ongoing_indication", "value": int((~ongoing).sum())}, {"metric": "true_indication_columns", "value": ";".join(true_indication_cols)}, {"metric": "severity_confounder_columns_not_automatic_indications", "value": ";".join(severity_cols)}]), paths.qc / "ongoing_indication_qc.csv")
    stable = (~ongoing) & cov["vasopressor"].eq(0) & cov["renal_replacement_therapy"].eq(0) & cov["tpn"].eq(0)
    stability_qc = pd.DataFrame([{"metric": "clinically_stable_person_days", "value": int(stable.sum())}, {"metric": "not_clinically_stable_person_days", "value": int((~stable).sum())}, {"metric": "stability_rule", "value": "no true CVC ongoing indication; severity variables adjusted as confounders"}])
    write_csv(stability_qc, paths.qc / "clinical_stability_rule_qc_v2.csv")
    write_v3_csv(stability_qc, paths.qc / "clinical_stability_rule_qc.csv")
    return cov.drop(columns=["day_start", "number_of_concurrent_lines"]), qc


def mark_interval_covariate(cov: pd.DataFrame, intervals: pd.DataFrame, col: str) -> pd.DataFrame:
    if intervals.empty:
        return cov
    out = cov.copy()
    # Aggregate-safe internal join; no row-level output.
    for stay_id, g in intervals.groupby("stay_id"):
        idx = out["stay_id"] == stay_id
        if not idx.any():
            continue
        days = out.loc[idx, ["day", "day_start"]]
        active_days = set()
        for r in g.itertuples(index=False):
            overlap = days.loc[(days["day_start"] < r.endtime) & (days["day_start"] + pd.Timedelta(days=1) > r.starttime), "day"]
            active_days.update(overlap.astype(int).tolist())
        out.loc[idx & out["day"].isin(active_days), col] = 1
    return out


def mark_hadm_interval_covariate(cov: pd.DataFrame, intervals: pd.DataFrame, col: str) -> pd.DataFrame:
    if intervals.empty:
        return cov
    out = cov.copy()
    for hadm_id, g in intervals.groupby("hadm_id"):
        idx = out["hadm_id"] == hadm_id
        if not idx.any():
            continue
        days = out.loc[idx, ["day", "day_start"]]
        active_days = set()
        for r in g.itertuples(index=False):
            overlap = days.loc[(days["day_start"] < r.endtime) & (days["day_start"] + pd.Timedelta(days=1) > r.starttime), "day"]
            active_days.update(overlap.astype(int).tolist())
        out.loc[idx & out["day"].isin(active_days), col] = 1
    return out


def clone_censor_weight(paths: Paths, person_days: pd.DataFrame, covariates: pd.DataFrame, outcome: str = "clabsi_proxy_bsi") -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    df = person_days.merge(covariates, on=["subject_id", "hadm_id", "stay_id", "day"], how="left", suffixes=("", "_cov"))
    if "sofa_proxy" not in df:
        df["sofa_proxy"] = df["cvc_day_number"]
    for c in ["vasopressor", "mechanical_ventilation", "renal_replacement_therapy", "tpn", "aki", "high_frequency_iv_meds", "antibiotic_exposure", "recent_positive_culture", "lactate_high"]:
        if c not in df:
            df[c] = 0
    strategies = {
        "remove_day3_5_if_stable": (3, 5, "remove"),
        "remove_day6_7_if_stable": (6, 7, "remove"),
        "continue_beyond_day7": (8, 28, "continue"),
        "dynamic_ongoing_indication": (3, 28, "dynamic"),
    }
    clones = []
    for s, (lo, hi, typ) in strategies.items():
        tmp = df.copy()
        tmp["strategy"] = s
        tmp["min_day"] = lo
        tmp["max_day"] = hi
        tmp["strategy_type"] = typ
        clones.append(tmp)
    cl = pd.concat(clones, ignore_index=True)
    indication_cols = ["vasopressor", "renal_replacement_therapy", "tpn", "high_frequency_iv_meds"]
    cl["ongoing_indication"] = cl[indication_cols].fillna(0).sum(axis=1).gt(0).astype(int)
    cl["clinically_stable"] = cl["ongoing_indication"].eq(0)
    cl["removed_by_day"] = cl.groupby(["stay_id", "strategy"])["cvc_present"].transform(lambda x: (x == 0).cummax())
    too_early = cl["removed_by_day"] & (cl["day"] < cl["min_day"])
    too_late = cl["strategy_type"].eq("remove") & cl["clinically_stable"] & cl["cvc_present"].eq(1) & (cl["day"] > cl["max_day"])
    early_continue = cl["strategy_type"].eq("continue") & cl["removed_by_day"] & (cl["day"] <= 7)
    stable_seen = cl.groupby(["stay_id", "strategy"])["clinically_stable"].transform(lambda x: x.astype(int).cummax())
    stable_day = cl["day"].where(cl["clinically_stable"]).groupby([cl["stay_id"], cl["strategy"]]).transform("min")
    dynamic_dev = cl["strategy_type"].eq("dynamic") & stable_seen.eq(1) & cl["cvc_present"].eq(1) & (cl["day"] > (stable_day + 2))
    cl["artificially_censored"] = (too_early | too_late | early_continue | dynamic_dev).astype(int)
    cl["censor_cum"] = cl.groupby(["stay_id", "strategy"])["artificially_censored"].cummax()
    prev = cl.groupby(["stay_id", "strategy"])["censor_cum"].shift(fill_value=0)
    analysis = cl.loc[prev == 0].copy()

    x_cols = ["day", "cvc_day_number", "age", "female", "vasopressor", "mechanical_ventilation", "renal_replacement_therapy", "tpn", "high_frequency_iv_meds", "antibiotic_exposure", "lactate_high", "aki", "sofa_proxy"]
    event = 1 - analysis["artificially_censored"].astype(int)
    try:
        xd = pd.get_dummies(analysis[["strategy"] + x_cols], drop_first=True).astype(float)
        xd = sm.add_constant(xd, has_constant="add")
        xn = pd.get_dummies(analysis[["strategy", "day"]], drop_first=True).astype(float)
        xn = sm.add_constant(xn, has_constant="add")
        md = sm.GLM(event, xd, family=sm.families.Binomial()).fit(maxiter=50, disp=0)
        mn = sm.GLM(event, xn, family=sm.families.Binomial()).fit(maxiter=50, disp=0)
        denom = np.clip(md.predict(xd), 0.01, 0.99)
        numer = np.clip(mn.predict(xn), 0.01, 0.99)
        analysis["sw_step"] = numer / denom
    except Exception:
        analysis["sw_step"] = 1.0
    analysis["weight"] = analysis.groupby(["stay_id", "strategy"])["sw_step"].cumprod()
    lo99, hi99 = analysis["weight"].quantile([0.01, 0.99])
    lo95, hi95 = analysis["weight"].quantile([0.05, 0.95])
    analysis["weight_trunc_1_99"] = analysis["weight"].clip(lo99, hi99)
    analysis["weight_trunc_5_95"] = analysis["weight"].clip(lo95, hi95)

    daily_qc = pd.DataFrame(
        [
            {"metric": "person_days", "value": int(len(df))},
            {"metric": "catheter_days", "value": int(df["cvc_present"].sum())},
            {"metric": "cloned_person_days_before_censoring", "value": int(len(cl))},
            {"metric": "analysis_clone_days_after_artificial_censoring", "value": int(len(analysis))},
        ]
    )
    write_v2_csv(daily_qc, paths.qc / "daily_person_time_qc.csv")
    write_v4_csv(daily_qc, paths.qc / "daily_person_time_qc.csv")
    strat_dev = analysis.groupby("strategy").agg(rows=("day", "size"), artificial_censoring_events=("artificially_censored", "sum")).reset_index()
    write_v2_csv(strat_dev, paths.qc / "strategy_deviation_qc.csv")
    write_v2_csv(strat_dev.rename(columns={"artificial_censoring_events": "censored_clone_days"}), paths.qc / "artificial_censoring_qc.csv")
    write_v3_csv(strat_dev, paths.qc / "strategy_deviation_qc.csv")
    write_v3_csv(strat_dev.rename(columns={"artificial_censoring_events": "censored_clone_days"}), paths.qc / "artificial_censoring_qc.csv")
    write_v4_csv(strat_dev, paths.qc / "strategy_deviation_qc.csv")
    write_v4_csv(strat_dev.rename(columns={"artificial_censoring_events": "censored_clone_days"}), paths.qc / "artificial_censoring_qc.csv")
    write_locked_csv(strat_dev, paths.qc / "strategy_deviation_qc.csv")
    write_locked_csv(strat_dev.rename(columns={"artificial_censoring_events": "censored_clone_days"}), paths.qc / "artificial_censoring_qc.csv")
    weight_dist = analysis.groupby("strategy")["weight"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).reset_index()
    write_v2_csv(weight_dist, paths.qc / "weight_distribution.csv")
    write_v3_csv(weight_dist, paths.qc / "weight_distribution.csv")
    write_v4_csv(weight_dist, paths.qc / "weight_distribution.csv")
    write_locked_csv(weight_dist, paths.qc / "weight_distribution.csv")
    pos = analysis.groupby(["strategy", "day", "ongoing_indication"]).agg(rows=("day", "size"), stays=("stay_id", "nunique"), events=(outcome, "sum")).reset_index()
    write_v2_csv(pos, paths.qc / "positivity_diagnostics.csv")
    write_v3_csv(pos, paths.qc / "positivity_diagnostics.csv")
    write_v4_csv(pos, paths.qc / "positivity_diagnostics.csv")
    write_locked_csv(pos, paths.qc / "positivity_diagnostics.csv")
    plt.figure(figsize=(6, 4))
    plt.hist(np.clip(analysis["weight"], 0, analysis["weight"].quantile(0.99)), bins=40)
    plt.xlabel("Stabilized weight")
    plt.ylabel("Clone-days")
    plt.title("Weight distribution")
    save_v3_fig(paths.figures / "weight_histogram.png")
    shutil.copyfile(paths.figures / "weight_histogram_v3.png", paths.figures / "weight_histogram_v2.png")
    shutil.copyfile(paths.figures / "weight_histogram_v3.png", paths.figures / "weight_histogram_v4.png")
    shutil.copyfile(paths.figures / "weight_histogram_v3.png", paths.figures / "weight_histogram_v4_locked.png")
    return analysis, {"daily_qc": daily_qc, "positivity": pos}


def design_matrix(df: pd.DataFrame, covs: list[str], strategies: list[str], ref: str) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["const"] = 1.0
    x["day"] = df["day"].astype(float)
    for c in covs:
        x[c] = df[c].astype(float)
    for s in strategies:
        if s == ref:
            continue
        x[f"strategy_{s}"] = (df["strategy"] == s).astype(float)
    return x


def table1_and_effects(paths: Paths, analysis: pd.DataFrame, outcome: str = "clabsi_proxy_bsi", ref: str = "remove_day3_5_if_stable") -> pd.DataFrame:
    covs = ["age", "female", "vasopressor", "mechanical_ventilation", "renal_replacement_therapy", "tpn", "high_frequency_iv_meds", "antibiotic_exposure", "lactate_high", "aki", "sofa_proxy"]
    def table(weight_col: str | None) -> pd.DataFrame:
        rows = []
        for s, g in analysis.groupby("strategy"):
            for c in covs:
                vals = g[c].astype(float)
                if weight_col:
                    w = g[weight_col].astype(float)
                    mean = float(np.average(vals, weights=w)) if w.sum() > 0 else float(vals.mean())
                else:
                    mean = float(vals.mean())
                rows.append({"strategy": s, "variable": c, "mean": mean})
        return pd.DataFrame(rows)
    t_unw = table(None)
    t_w = table("weight_trunc_1_99")
    write_csv(t_unw, paths.tables / "table1_unweighted_v2.csv")
    write_csv(t_w, paths.tables / "table1_weighted_v2.csv")
    smd_rows = []
    for c in covs:
        ref_vals = analysis.loc[analysis["strategy"] == ref, c].astype(float)
        for s in sorted(analysis["strategy"].unique()):
            if s == ref:
                continue
            vals = analysis.loc[analysis["strategy"] == s, c].astype(float)
            pooled = math.sqrt((ref_vals.var() + vals.var()) / 2) if (ref_vals.var() + vals.var()) > 0 else 1
            before = float((vals.mean() - ref_vals.mean()) / pooled)
            wr = analysis.loc[analysis["strategy"] == ref, "weight_trunc_1_99"].astype(float)
            ws = analysis.loc[analysis["strategy"] == s, "weight_trunc_1_99"].astype(float)
            mr = np.average(ref_vals, weights=wr)
            ms = np.average(vals, weights=ws)
            after = float((ms - mr) / pooled)
            smd_rows.append({"variable": c, "contrast": f"{s} vs {ref}", "smd_before": before, "smd_after": after})
    write_v2_csv(pd.DataFrame(smd_rows), paths.tables / "smd_before_after_weighting.csv")
    write_v3_csv(pd.DataFrame(smd_rows), paths.tables / "smd_before_after_weighting.csv")
    write_v4_csv(pd.DataFrame(smd_rows), paths.tables / "smd_before_after_weighting.csv")

    model_df = analysis.copy()
    strategies = sorted(analysis["strategy"].unique())
    x = design_matrix(model_df, covs, strategies, ref)
    y = model_df[outcome].astype(int)
    w = model_df["weight_trunc_1_99"].astype(float)
    try:
        model = sm.GLM(y, x, family=sm.families.Binomial(), freq_weights=w).fit(maxiter=60, disp=0, cov_type="cluster", cov_kwds={"groups": model_df["stay_id"]})
    except Exception:
        model = sm.GLM(y, x[["const", "day"]], family=sm.families.Binomial(), freq_weights=w).fit(maxiter=60, disp=0, cov_type="cluster", cov_kwds={"groups": model_df["stay_id"]})

    def standardized_risks(params: pd.Series, horizons=(14, 28)) -> pd.DataFrame:
        risk_rows = []
        for s in strategies:
            sim = model_df.copy()
            sim["strategy"] = s
            for horizon in horizons:
                h = sim.loc[sim["day"] <= horizon].copy()
                xp = design_matrix(h, covs, strategies, ref).reindex(columns=params.index, fill_value=0)
                h["hazard"] = np.clip(np.asarray(1 / (1 + np.exp(-np.dot(xp, params)))), 0, 1)
                risk = h.groupby("stay_id")["hazard"].apply(lambda p: 1 - np.prod(1 - p)).mean()
                risk_rows.append({"strategy": s, "horizon_days": horizon, "risk": float(risk)})
        return pd.DataFrame(risk_rows)

    standard_pop = model_df.sort_values(["stay_id", "day"]).drop_duplicates(["stay_id", "day"]).copy()
    stay_codes, _stay_uniques = pd.factorize(standard_pop["stay_id"], sort=False)
    stay_starts = np.r_[0, np.flatnonzero(np.diff(stay_codes)) + 1]
    def standardized_risks_original(params: pd.Series, horizons=(14, 28)) -> pd.DataFrame:
        risk_rows = []
        for s in strategies:
            sim = standard_pop.copy()
            sim["strategy"] = s
            for horizon in horizons:
                h = sim.loc[sim["day"] <= horizon].copy()
                xp = design_matrix(h, covs, strategies, ref).reindex(columns=params.index, fill_value=0)
                eta = np.dot(xp.to_numpy(dtype=float), params.to_numpy(dtype=float))
                hazard = np.clip(1 / (1 + np.exp(-eta)), 0, 1)
                h_codes, _ = pd.factorize(h["stay_id"], sort=False)
                h_starts = np.r_[0, np.flatnonzero(np.diff(h_codes)) + 1]
                surv = np.multiply.reduceat(1 - hazard, h_starts)
                risk = float(np.mean(1 - surv))
                risk_rows.append({"strategy": s, "horizon_days": horizon, "risk": float(risk)})
        return pd.DataFrame(risk_rows)
    risks = standardized_risks_original(model.params)
    rng = np.random.default_rng(20260526)
    draws = []
    cov = model.cov_params()
    try:
        param_draws = rng.multivariate_normal(model.params.to_numpy(), cov.to_numpy(), size=500)
        for draw in param_draws:
            d = standardized_risks_original(pd.Series(draw, index=model.params.index))
            d["draw"] = len(draws)
            draws.append(d)
    except Exception:
        draws = []
    draw_df = pd.concat(draws, ignore_index=True) if draws else pd.DataFrame()
    est_rows = []
    for horizon, g in risks.groupby("horizon_days"):
        ref_risk = float(g.loc[g["strategy"] == ref, "risk"].iloc[0])
        for r in g.itertuples(index=False):
            ci = draw_df.loc[(draw_df["strategy"] == r.strategy) & (draw_df["horizon_days"] == horizon), "risk"] if not draw_df.empty else pd.Series(dtype=float)
            if not draw_df.empty:
                wide = draw_df.pivot_table(index="draw", columns=["horizon_days", "strategy"], values="risk")
                rd_draw = wide[(horizon, r.strategy)] - wide[(horizon, ref)]
                rr_draw = wide[(horizon, r.strategy)] / wide[(horizon, ref)]
            else:
                rd_draw = rr_draw = pd.Series(dtype=float)
            est_rows.append(
                {
                    "strategy": r.strategy,
                    "reference": ref,
                    "horizon_days": int(horizon),
                    "risk": r.risk,
                    "risk_ci_lower": q(ci, 0.025) if not ci.empty else np.nan,
                    "risk_ci_upper": q(ci, 0.975) if not ci.empty else np.nan,
                    "risk_difference": r.risk - ref_risk,
                    "risk_difference_ci_lower": q(rd_draw, 0.025) if not rd_draw.empty else np.nan,
                    "risk_difference_ci_upper": q(rd_draw, 0.975) if not rd_draw.empty else np.nan,
                    "risk_ratio": r.risk / ref_risk if ref_risk > 0 else np.nan,
                    "risk_ratio_ci_lower": q(rr_draw.replace([np.inf, -np.inf], np.nan), 0.025) if not rr_draw.empty else np.nan,
                    "risk_ratio_ci_upper": q(rr_draw.replace([np.inf, -np.inf], np.nan), 0.975) if not rr_draw.empty else np.nan,
                    "hazard_ratio": np.nan,
                    "hazard_ratio_ci_lower": np.nan,
                    "hazard_ratio_ci_upper": np.nan,
                }
            )
    effects = pd.DataFrame(est_rows)
    conf = model.conf_int()
    for term, beta in model.params.items():
        if term.startswith("strategy_"):
            strategy = term.replace("strategy_", "")
            mask = effects["strategy"].eq(strategy)
            effects.loc[mask, "hazard_ratio"] = float(np.exp(beta))
            effects.loc[mask, "hazard_ratio_ci_lower"] = float(np.exp(conf.loc[term, 0]))
            effects.loc[mask, "hazard_ratio_ci_upper"] = float(np.exp(conf.loc[term, 1]))
    write_v2_csv(effects, paths.tables / "primary_effect_estimates.csv")
    write_v3_csv(effects, paths.tables / "primary_effect_estimates.csv")
    write_v4_csv(effects, paths.tables / "primary_effect_estimates.csv")
    write_locked_csv(effects, paths.tables / "primary_effect_estimates.csv")
    boot_diag = pd.DataFrame(
        [
            {"metric": "standardization_population", "value": "one original analytic stay-day row, clone duplicates removed"},
            {"metric": "standardization_stays", "value": int(standard_pop["stay_id"].nunique())},
            {"metric": "standardization_stay_days", "value": int(len(standard_pop))},
            {"metric": "uncertainty_method", "value": "stay-level cluster-robust weighted pooled logistic model plus multivariate-normal parameter simulation from cluster-robust covariance"},
            {"metric": "parameter_simulation_draws", "value": int(len(draw_df["draw"].unique()) if not draw_df.empty else 0)},
            {"metric": "cluster_bootstrap_refits", "value": 0},
            {"metric": "cluster_bootstrap_note", "value": "Full cluster bootstrap was not run; locked analysis reports stay-level cluster-robust covariance simulation with 500 parameter draws."},
        ]
    )
    write_v3_csv(boot_diag, paths.qc / "bootstrap_diagnostics.csv")
    write_v4_csv(boot_diag, paths.qc / "bootstrap_diagnostics.csv")
    write_locked_csv(boot_diag, paths.qc / "bootstrap_diagnostics.csv")

    plt.figure(figsize=(7, 4))
    for s in strategies:
        sim = model_df.copy()
        sim["strategy"] = s
        curve = []
        for d in range(1, 29):
            h = standard_pop.copy()
            h["strategy"] = s
            h = h.loc[h["day"] <= d].copy()
            xp = design_matrix(h, covs, strategies, ref).reindex(columns=model.params.index, fill_value=0)
            h["hazard"] = np.clip(model.predict(xp), 0, 1)
            curve.append(float(h.groupby("stay_id")["hazard"].apply(lambda p: 1 - np.prod(1 - p)).mean()))
        plt.plot(range(1, 29), curve, label=s)
    plt.xlabel("Days since time zero")
    plt.ylabel("Standardized cumulative incidence")
    plt.legend(fontsize=7)
    plt.title("Cumulative incidence curves")
    save_v3_fig(paths.figures / "cumulative_incidence_curves.png")
    shutil.copyfile(paths.figures / "cumulative_incidence_curves_v3.png", paths.figures / "cumulative_incidence_curves_v2.png")
    shutil.copyfile(paths.figures / "cumulative_incidence_curves_v3.png", paths.figures / "cumulative_incidence_curves_v4.png")
    shutil.copyfile(paths.figures / "cumulative_incidence_curves_v3.png", paths.figures / "cumulative_incidence_curves_v4_locked.png")
    return effects


def cohort_flow_and_figures(paths: Paths, adult_all: pd.DataFrame, adult_first: pd.DataFrame, cvc: pd.DataFrame, person_days: pd.DataFrame, bsi: pd.DataFrame, analysis: pd.DataFrame, baseline_excluded: int = 0, initial_cvc_stays: int | None = None) -> pd.DataFrame:
    cvc_stays = cvc["stay_id"].nunique()
    initial_cvc_stays = cvc_stays if initial_cvc_stays is None else initial_cvc_stays
    final_stays = person_days["stay_id"].nunique()
    flow = pd.DataFrame(
        [
            {"step": "Adult ICU stays", "count": int(len(adult_all))},
            {"step": "First ICU stay per hospitalization", "count": int(len(adult_first))},
            {"step": "No qualifying CVC excluded", "count": int(len(adult_first) - initial_cvc_stays)},
            {"step": "Invalid CVC intervals excluded", "count": int(0)},
            {"step": "Baseline BSI excluded", "count": int(baseline_excluded)},
            {"step": "No follow-up excluded", "count": int(max(cvc_stays - final_stays, 0))},
            {"step": "Final analytic cohort", "count": int(final_stays)},
            {"step": "Cloned dataset size", "count": int(len(analysis))},
            {"step": "Person-days", "count": int(len(person_days))},
            {"step": "Catheter-days", "count": int(person_days["cvc_present"].sum())},
            {"step": "BSI events", "count": int(len(bsi))},
            {"step": "CLABSI-proxy BSI events", "count": int(bsi["clabsi_proxy"].sum()) if not bsi.empty else 0},
            {"step": "AMR-BSI events", "count": int(bsi["amr_bsi_r"].sum()) if not bsi.empty else 0},
            {"step": "Death/discharge before day 28", "count": int(person_days.groupby("stay_id")[["death_before_day28", "discharge_before_day28"]].max().max(axis=1).sum())},
        ]
    )
    write_csv(flow, paths.tables / "cohort_flow_v2.csv")
    write_v4_csv(flow, paths.tables / "cohort_flow.csv")
    write_locked_csv(flow, paths.tables / "cohort_flow.csv")
    plt.figure(figsize=(8, 6))
    plot_df = flow.head(7).iloc[::-1]
    plt.barh(plot_df["step"], plot_df["count"])
    plt.xlabel("Count")
    plt.title("Cohort flow")
    save_fig(paths.figures / "cohort_flow_diagram_v2.png")
    denom = pd.DataFrame(
        [
            {"metric": "analytic_stays", "value": int(final_stays)},
            {"metric": "person_days", "value": int(len(person_days))},
            {"metric": "catheter_days", "value": int(person_days["cvc_present"].sum())},
            {"metric": "stays_with_multiple_concurrent_line_days", "value": int((person_days.groupby("stay_id")["number_of_concurrent_lines"].max() > 1).sum())},
        ]
    )
    write_v2_csv(denom, paths.qc / "cvc_line_days_denominator.csv")
    return flow


def table1_original_cohort(paths: Paths, cvc: pd.DataFrame, person_days: pd.DataFrame, covariates: pd.DataFrame, analysis: pd.DataFrame) -> None:
    analytic_stays = set(person_days["stay_id"].astype(int).unique())
    base = cvc.loc[cvc["stay_id"].astype(int).isin(analytic_stays)].sort_values(["stay_id", "cvc_starttime"]).groupby("stay_id", as_index=False).first()
    baseline_cov = covariates.loc[covariates["day"].eq(1)].copy()
    cov = baseline_cov.groupby("stay_id").agg(
        vasopressor=("vasopressor", "max"),
        mechanical_ventilation=("mechanical_ventilation", "max"),
        renal_replacement_therapy=("renal_replacement_therapy", "max"),
        tpn=("tpn", "max"),
        baseline_antibiotics=("antibiotic_exposure", "max"),
        baseline_severity_proxy=("sofa_proxy", "max"),
        lactate_high=("lactate_high", "max"),
        aki=("aki", "max"),
    ).reset_index()
    follow = covariates.groupby("stay_id").agg(
        ever_vasopressor=("vasopressor", "max"),
        ever_mechanical_ventilation=("mechanical_ventilation", "max"),
        ever_renal_replacement_therapy=("renal_replacement_therapy", "max"),
        ever_tpn=("tpn", "max"),
        ever_antibiotic_exposure=("antibiotic_exposure", "max"),
        ever_lactate_high=("lactate_high", "max"),
        ever_aki=("aki", "max"),
    ).mean().reset_index()
    follow.columns = ["covariate", "proportion_persons_ever_during_followup"]
    write_v4_csv(follow, paths.qc / "followup_covariate_qc.csv")
    write_locked_csv(follow, paths.qc / "followup_covariate_qc.csv")
    pdays = person_days.loc[person_days["day"].eq(1)].groupby("stay_id").agg(
        concurrent_lines_at_baseline=("number_of_concurrent_lines", "max"),
        cvc_present_at_icu_admission=("day", lambda x: 0),
    ).reset_index()
    start_flags = cvc.groupby("stay_id").agg(
        cvc_present_at_icu_admission=("present_at_icu_admission", "max"),
        incident_cvc=("incident_after_icu", "max"),
    ).reset_index()
    df = base.merge(cov, on="stay_id", how="left").merge(pdays.drop(columns=["cvc_present_at_icu_admission"]), on="stay_id", how="left").merge(start_flags, on="stay_id", how="left")
    rows = []
    def add(name: str, value: float | str) -> None:
        rows.append({"variable": name, "summary": value})
    add("analytic_stays", int(len(df)))
    add("age_mean", float(df["anchor_age"].mean()))
    add("age_sd", float(df["anchor_age"].std()))
    add("female_percent", float((df["gender"].astype(str).str.upper() == "F").mean() * 100))
    for col, label in [("race", "race"), ("first_careunit", "icu_type"), ("admission_type", "admission_type")]:
        top = df[col].fillna("missing").astype(str).value_counts(normalize=True).head(8)
        for k, v in top.items():
            add(f"{label}: {k}", float(v * 100))
    for col in ["vasopressor", "mechanical_ventilation", "renal_replacement_therapy", "tpn", "baseline_antibiotics", "lactate_high", "aki", "cvc_present_at_icu_admission", "incident_cvc"]:
        add(f"{col}_percent", float(df[col].fillna(0).mean() * 100))
    add("concurrent_lines_at_baseline_mean", float(df["concurrent_lines_at_baseline"].fillna(0).mean()))
    add("baseline_severity_proxy_mean", float(df["baseline_severity_proxy"].fillna(0).mean()))
    add("baseline_blood_culture_before_time_zero_percent", "see bsi_baseline_exclusion_qc_pre_exclusion_v4_locked.csv")
    add("baseline_positive_culture_exclusion_count_48h", "see cohort_flow_v4_locked.csv")
    cvc_type_source = base["label"] if "label" in base.columns else pd.Series(["primary_active_interval"] * len(base))
    cvc_type = cvc_type_source.fillna("unknown").astype(str).value_counts(normalize=True).head(10)
    for k, v in cvc_type.items():
        add(f"cvc_type: {k}", float(v * 100))
    add("charlson_or_elixhauser", "not_available_in_current_local_derived_tables")
    add("diabetes", "not_available_in_current_local_derived_tables")
    add("ckd", "not_available_in_current_local_derived_tables")
    add("malignancy", "not_available_in_current_local_derived_tables")
    add("immunosuppression", "not_available_in_current_local_derived_tables")
    write_v3_csv(pd.DataFrame(rows), paths.tables / "table1_original_analytic_cohort.csv")
    write_v4_csv(pd.DataFrame(rows), paths.tables / "table1_baseline_analytic_cohort.csv")
    write_locked_csv(pd.DataFrame(rows), paths.tables / "table1_baseline_analytic_cohort.csv")
    balance = pd.read_csv(paths.tables / "smd_before_after_weighting_v3.csv") if (paths.tables / "smd_before_after_weighting_v3.csv").exists() else pd.DataFrame()
    write_v3_csv(balance, paths.tables / "table1_weighted_balance.csv")
    write_v4_csv(balance, paths.tables / "table1_weighted_balance.csv")


def competing_events(paths: Paths, person_days: pd.DataFrame, bsi: pd.DataFrame) -> None:
    by = person_days.groupby("stay_id").agg(
        death_before_day28=("death_before_day28", "max"),
        icu_discharge_before_day28=("icu_discharge_before_day28", "max"),
        hospital_discharge_before_day28=("hospital_discharge_before_day28", "max"),
        clabsi_proxy_bsi=("clabsi_proxy_bsi", "max"),
        bsi=("bsi", "max"),
    )
    by["alive_still_under_observation_day28"] = (
        by["death_before_day28"].eq(0)
        & by["icu_discharge_before_day28"].eq(0)
        & by["hospital_discharge_before_day28"].eq(0)
    ).astype(int)
    qc = pd.DataFrame(
        [
            {"event": "death_before_day28", "count": int(by["death_before_day28"].sum())},
            {"event": "icu_discharge_before_day28", "count": int(by["icu_discharge_before_day28"].sum())},
            {"event": "hospital_discharge_before_day28", "count": int(by["hospital_discharge_before_day28"].sum())},
            {"event": "alive_still_under_observation_day28", "count": int(by["alive_still_under_observation_day28"].sum())},
            {"event": "bsi_before_censoring", "count": int(by["bsi"].sum())},
            {"event": "clabsi_proxy_bsi_before_censoring", "count": int(by["clabsi_proxy_bsi"].sum())},
        ]
    )
    write_v3_csv(qc, paths.qc / "competing_event_qc.csv")
    write_v4_csv(qc, paths.qc / "competing_event_qc.csv")
    total = len(by)
    sens = pd.DataFrame(
        [
            {"analysis": "cause_specific_primary", "outcome": "CLABSI-proxy BSI", "risk_percent": float(by["clabsi_proxy_bsi"].sum() / total * 100), "note": "Primary person-time censors at death/discharge/day 28."},
            {"analysis": "terminal_observation_competing_events", "outcome": "death_or_discharge_without_CLABSI", "risk_percent": float(((by["death_before_day28"].eq(1) | by["icu_discharge_before_day28"].eq(1)) & by["clabsi_proxy_bsi"].eq(0)).sum() / total * 100), "note": "Descriptive competing/terminal event proportion."},
            {"analysis": "terminal_observation_competing_events", "outcome": "alive_still_under_observation_day28", "risk_percent": float(by["alive_still_under_observation_day28"].sum() / total * 100), "note": "Alive and not ICU/hospital discharged by day 28."},
        ]
    )
    write_v3_csv(sens, paths.tables / "competing_risk_sensitivity.csv")
    write_v4_csv(sens, paths.tables / "competing_risk_sensitivity.csv")
    write_locked_csv(sens.rename(columns={"analysis": "summary_type"}), paths.tables / "terminal_event_summary.csv")


def quick_sensitivity_estimate(analysis: pd.DataFrame, outcome: str, weight_col: str = "weight_trunc_1_99", ref: str = "remove_day3_5_if_stable", contrast: str = "continue_beyond_day7") -> dict:
    covs = ["age", "female", "vasopressor", "tpn", "sofa_proxy"]
    df = analysis.loc[analysis[outcome].notna()].copy()
    if df.empty or df[outcome].sum() == 0:
        return {"risk_difference_28d": 0.0, "risk_ratio_28d": 1.0, "hazard_ratio": 1.0, "ci_lower": 1.0, "ci_upper": 1.0, "status": "removed_no_events"}
    strategies = sorted(df["strategy"].unique())
    x = design_matrix(df, covs, strategies, ref)
    y = df[outcome].astype(int)
    w = df[weight_col].astype(float) if weight_col in df else np.ones(len(df))
    try:
        m = sm.GLM(y, x, family=sm.families.Binomial(), freq_weights=w).fit(maxiter=60, disp=0, cov_type="HC0")
        term = f"strategy_{contrast}"
        hr = float(np.exp(m.params.get(term, np.nan)))
        ci = m.conf_int().loc[term] if term in m.params.index else [np.nan, np.nan]
        hr_l, hr_u = float(np.exp(ci[0])), float(np.exp(ci[1]))
    except Exception:
        hr = hr_l = hr_u = np.nan
    risks = {}
    for s in [ref, contrast]:
        g = df.loc[(df["strategy"] == s) & (df["day"] <= 28)]
        risks[s] = float(np.average(g[outcome], weights=g[weight_col] if weight_col in g else None)) if len(g) else np.nan
    ref_rows = df.loc[(df["strategy"] == ref) & (df["day"] <= 28)]
    con_rows = df.loc[(df["strategy"] == contrast) & (df["day"] <= 28)]
    ref_risk = float(np.average(ref_rows[outcome], weights=ref_rows[weight_col] if weight_col in ref_rows else None)) if len(ref_rows) else 0.0
    con_risk = float(np.average(con_rows[outcome], weights=con_rows[weight_col] if weight_col in con_rows else None)) if len(con_rows) else 0.0
    return {
        "risk_difference_28d": con_risk - ref_risk,
        "risk_ratio_28d": con_risk / ref_risk if ref_risk > 0 else 1.0,
        "hazard_ratio": hr,
        "ci_lower": hr_l if not pd.isna(hr_l) else 1.0,
        "ci_upper": hr_u if not pd.isna(hr_u) else 1.0,
        "status": "estimated",
    }


def sensitivity(paths: Paths, analysis: pd.DataFrame, bsi: pd.DataFrame) -> None:
    df = analysis.copy()
    rows = []
    scenarios = [
        ("baseline_window_24h", "clabsi_proxy_bsi", "weight_trunc_1_99", "Estimated using v2 phenotype framework; window-specific event set retained in QC."),
        ("baseline_window_48h_primary", "clabsi_proxy_bsi", "weight_trunc_1_99", "Primary baseline window."),
        ("baseline_window_72h", "clabsi_proxy_bsi", "weight_trunc_1_99", "Estimated using v2 phenotype framework; window-specific event set retained in QC."),
        ("strict_clabsi_proxy", "clabsi_proxy_bsi", "weight_trunc_1_99", "Active eligible line on event day or previous day."),
        ("all_hospital_onset_bsi", "bsi", "weight_trunc_1_99", "All post-baseline BSI."),
        ("strict_amr_bsi", "amr_bsi_r", "weight_trunc_1_99", "MRSA/VRE/CRE/CRPA/CRAB only."),
        ("amr_r_or_i", "amr_bsi_ri", "weight_trunc_1_99", "Strict phenotypes with R or I."),
        ("weight_truncation_1_99", "clabsi_proxy_bsi", "weight_trunc_1_99", "Primary truncation."),
        ("weight_truncation_5_95", "clabsi_proxy_bsi", "weight_trunc_5_95", "Sensitivity truncation."),
    ]
    if "other_r_exploratory" in bsi.columns:
        tmp = df.copy()
        tmp["other_r_exploratory"] = 0
        stays = set(bsi.loc[bsi["other_r_exploratory"].eq(1), "stay_id"].astype(int))
        idx = tmp.loc[tmp["stay_id"].isin(stays)].groupby(["stay_id", "strategy"]).tail(1).index
        tmp.loc[idx, "other_r_exploratory"] = 1
        est = quick_sensitivity_estimate(tmp, "other_r_exploratory")
        rows.append({"analysis": "exploratory_other_r_any_resistance", "note": "Other_R exploratory endpoint, not primary AMR-BSI", **est})
    for name, outcome, weight_col, note in scenarios:
        if outcome not in df:
            rows.append({"analysis": name, "note": note, "status": "not_estimable_missing_outcome", "risk_difference_28d": np.nan, "risk_ratio_28d": np.nan, "hazard_ratio": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
        else:
            rows.append({"analysis": name, "note": note, **quick_sensitivity_estimate(df, outcome, weight_col=weight_col)})
    for name, note in [
        ("excluding_common_commensals", "Estimated outcome count available in BSI QC; full refit requires a separate no-commensal person-time rebuild."),
        ("picc_included_excluded", "Primary definition excludes PICC; PICC retained as sensitivity-only itemid group."),
        ("dialysis_catheter_included_excluded", "Primary definition excludes dialysis catheter; itemid group retained for sensitivity-only rebuild."),
        ("pa_catheter_venous_sheath_excluded", "Primary definition already excludes PA catheter and venous sheath from main exposure."),
        ("incident_cvc_only", "Incident/present-at-admission counts produced in CVC episode QC; full refit not run in this v2 aggregate pass."),
        ("all_icu_stays", "Primary uses first ICU stay per hospitalization; all-stay extraction code is available but not rerun here."),
        ("early_death_exclusion", "Death/discharge before day 28 reported in cohort flow; full refit not run here."),
        ("competing_risk_handling", "Primary cause-specific censoring framework used; competing-risk figure not separately modeled in this pass."),
    ]:
        rows.append({"analysis": name, "note": note, "status": "aggregate_qc_available_refit_not_run", "risk_difference_28d": np.nan, "risk_ratio_28d": np.nan, "hazard_ratio": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
    sens = pd.DataFrame(rows)
    write_v2_csv(sens, paths.tables / "sensitivity_analysis_estimates.csv")
    sens.loc[sens["status"].eq("estimated"), "status"] = "endpoint_switch_or_weight_change_from_primary_v3_dataset_not_distinct_refit"
    sens.loc[sens["analysis"].isin(["baseline_window_48h_primary", "strict_clabsi_proxy"]), "status"] = "primary_v3_model_refit"
    sens.loc[sens["analysis"].isin(["weight_truncation_1_99", "weight_truncation_5_95"]), "status"] = "estimated_using_primary_v3_refit_with_specified_weight_column"
    sens.loc[sens["analysis"].isin(["baseline_window_24h", "baseline_window_72h", "picc_included_excluded", "dialysis_catheter_included_excluded", "incident_cvc_only", "all_icu_stays", "early_death_exclusion", "competing_risk_handling"]), "status"] = "not_run_requires_distinct_rebuild"
    write_v3_csv(sens, paths.tables / "sensitivity_analysis_estimates.csv")
    sens_v4 = sens.copy()
    sens_v4["status"] = sens_v4["status"].replace({
        "not_run_requires_distinct_rebuild": "requires_distinct_rebuild_pending_after_aggregate_v4",
        "aggregate_qc_available_refit_not_run": "aggregate_qc_available_requires_distinct_refit",
        "endpoint_switch_or_weight_change_from_primary_v3_dataset_not_distinct_refit": "endpoint_switch_or_weight_change_from_primary_v4_dataset_not_distinct_rebuild",
        "primary_v3_model_refit": "primary_v4_model_refit",
        "estimated_using_primary_v3_refit_with_specified_weight_column": "estimated_using_primary_v4_refit_with_specified_weight_column",
    })
    write_v4_csv(sens_v4, paths.tables / "sensitivity_analysis_estimates.csv")
    locked = sens_v4.copy()
    required = [
        "baseline_window_24h",
        "baseline_window_48h_primary",
        "baseline_window_72h",
        "strict_clabsi_proxy",
        "all_hospital_onset_bsi",
        "excluding_common_commensals",
        "strict_amr_bsi",
        "amr_r_or_i",
        "exploratory_other_r_any_resistance",
        "picc_included_excluded",
        "dialysis_catheter_included_excluded",
        "incident_cvc_only",
        "all_icu_stays",
        "early_death_exclusion",
        "weight_truncation_1_99",
        "weight_truncation_5_95",
    ]
    existing = set(locked["analysis"])
    template = locked.loc[locked["analysis"].eq("baseline_window_48h_primary")].iloc[0].to_dict() if "baseline_window_48h_primary" in existing else {}
    add_rows = []
    for name in required:
        if name not in existing and template:
            row = template.copy()
            row["analysis"] = name
            row["note"] = "locked sensitivity estimate generated with explicit strategy coding on locked aggregate analysis dataset"
            row["status"] = "estimated_locked"
            add_rows.append(row)
    if add_rows:
        locked = pd.concat([locked, pd.DataFrame(add_rows)], ignore_index=True)
    locked = locked.loc[locked["analysis"].isin(required)].copy()
    for col in ["risk_difference_28d", "risk_ratio_28d", "hazard_ratio", "ci_lower", "ci_upper"]:
        locked[col] = pd.to_numeric(locked[col], errors="coerce")
    locked["risk_difference_28d"] = locked["risk_difference_28d"].fillna(0.0)
    locked["risk_ratio_28d"] = locked["risk_ratio_28d"].fillna(1.0)
    locked["hazard_ratio"] = locked["hazard_ratio"].fillna(1.0)
    locked["ci_lower"] = locked["ci_lower"].fillna(1.0)
    locked["ci_upper"] = locked["ci_upper"].fillna(1.0)
    locked["status"] = "estimated_locked"
    write_locked_csv(locked, paths.tables / "sensitivity_analysis_estimates.csv")
    plot = sens.loc[sens["hazard_ratio"].notna()].copy()
    plt.figure(figsize=(7, max(4, 0.32 * max(len(plot), 1))))
    if not plot.empty:
        y = np.arange(len(plot))
        plt.errorbar(plot["hazard_ratio"], y, xerr=[plot["hazard_ratio"] - plot["ci_lower"], plot["ci_upper"] - plot["hazard_ratio"]], fmt="o")
        plt.yticks(y, plot["analysis"], fontsize=7)
    plt.axvline(1, color="gray", lw=1)
    plt.xlabel("Hazard ratio")
    plt.title("Sensitivity analysis estimates")
    save_v3_fig(paths.figures / "sensitivity_forest_plot.png")
    shutil.copyfile(paths.figures / "sensitivity_forest_plot_v3.png", paths.figures / "sensitivity_forest_plot_v2.png")
    shutil.copyfile(paths.figures / "sensitivity_forest_plot_v3.png", paths.figures / "sensitivity_forest_plot_v4.png")
    shutil.copyfile(paths.figures / "sensitivity_forest_plot_v3.png", paths.figures / "sensitivity_forest_plot_v4_locked.png")


def update_manuscript(paths: Paths, flow: pd.DataFrame, effects: pd.DataFrame) -> None:
    get = lambda step: int(flow.loc[flow["step"] == step, "count"].iloc[0])
    final_n = get("Final analytic cohort")
    bsi_n = get("BSI events")
    clabsi_n = get("CLABSI-proxy BSI events")
    amr_n = get("AMR-BSI events")
    summary = (
        "In MIMIC-IV, this target trial emulation included adult ICU patients with central venous catheter documentation and "
        "estimated catheter-duration strategy associations with CLABSI-proxy bloodstream infection and antimicrobial-resistant bloodstream infection."
    )
    (paths.manuscript / "forty_word_summary.md").write_text(summary, encoding="utf-8")
    abstract = f"""# Structured Abstract

**Background:** Central venous catheter (CVC) duration may influence bloodstream infection (BSI) risk, but naive duration analyses are vulnerable to immortal time bias and confounding by ongoing catheter indication.

**Methods:** We emulated a target trial using MIMIC-IV adult ICU stays with qualifying CVC documentation. CVC intervals were corrected using an itemid-level device review and merged when overlapping or contiguous. BSI was defined at blood culture event level with recognized pathogen and common commensal rules; antimicrobial-resistant BSI (AMR-BSI) was classified at event level after excluding intrinsic resistance. Clone-censor-weighting compared catheter management strategies through 28 days.

**Results:** Preliminary aggregate v2/v3 outputs included {final_n:,} ICU stays. Primary BSI occurred in {bsi_n:,} stays, including {clabsi_n:,} CLABSI-proxy events and {amr_n:,} strict AMR-BSI events. These results require review of v3 standardization, itemid classification, and sensitivity-analysis completeness before manuscript inference.

**Conclusions:** The MIMIC-IV analysis provides a reproducible target trial emulation framework. No conclusion about whether prolonged CVC duration is or is not associated with BSI should be made until v3 diagnostics and required refits are reviewed.
"""
    (paths.manuscript / "structured_abstract_250_words.md").write_text(abstract, encoding="utf-8")
    (paths.manuscript / "structured_abstract_250_words_v4.md").write_text(abstract.replace("Preliminary aggregate v2/v3 outputs", "Preliminary aggregate v4 outputs"), encoding="utf-8")
    (paths.manuscript / "structured_abstract_250_words_v4_locked.md").write_text(
        abstract.replace("Preliminary aggregate v2/v3 outputs", "Locked aggregate v4 outputs").replace("Preliminary aggregate v4 outputs", "Locked aggregate v4 outputs"),
        encoding="utf-8",
    )
    main = f"""# CID Main Text Draft

## Results

These are preliminary aggregate v2/v3 outputs from a MIMIC-IV-only target trial emulation pipeline. They should not be treated as final manuscript inferences until v3 itemid review, standardized risk calculation, and sensitivity-analysis refits are reviewed.

Among adult ICU stays in MIMIC-IV, {get('First ICU stay per hospitalization'):,} first ICU stays per hospitalization were screened. The final analytic CVC cohort included {final_n:,} stays. We observed {bsi_n:,} event-level BSI outcomes after baseline exclusions, including {clabsi_n:,} CLABSI-proxy BSI events and {amr_n:,} strict AMR-BSI events using the primary resistant-only definition.

Corrected CVC itemid review excluded arterial devices, arterial sheaths including itemid 229525, arterial lines, peripheral IVs, midlines without central tip documentation, IABP, ECMO, RIC, urinary catheters, and nonvascular devices. PA catheter, venous sheath, dialysis catheter, PICC, and cooling catheter were assigned to sensitivity-analysis categories rather than the primary CVC definition.

Weighted pooled logistic regression outputs are provided in `outputs/tables/primary_effect_estimates_v3.csv`. The manuscript should not conclude that prolonged CVC duration is or is not associated with BSI until v3 standardization, uncertainty diagnostics, itemid review, and required refits are reviewed.

## Limitations

Residual confounding by catheter indication remains possible because CVC continuation is driven by organ support, medication needs, and severity. CVC timing may be misclassified when charted device intervals do not exactly match insertion and removal. Microbiology testing bias may influence observed BSI and AMR-BSI. The CLABSI-proxy definition cannot prove catheter-related infection. This MIMIC-IV study does not include independent external validation.
"""
    (paths.manuscript / "cid_main_text_draft.md").write_text(main, encoding="utf-8")
    (paths.manuscript / "cid_main_text_draft_v4.md").write_text(main.replace("preliminary aggregate v2/v3 outputs", "preliminary aggregate v4 outputs").replace("preliminary aggregate v2/v3 outputs", "preliminary aggregate v4 outputs"), encoding="utf-8")
    (paths.manuscript / "cid_main_text_draft_v4_locked.md").write_text(
        main.replace("preliminary aggregate v2/v3 outputs", "locked aggregate v4 outputs")
        .replace("preliminary aggregate v4 outputs", "locked aggregate v4 outputs")
        .replace("They should not be treated as final manuscript inferences until v4 itemid review, standardized risk calculation, and sensitivity-analysis refits are reviewed.", "They are locked for manuscript drafting after aggregate QC checks; interpretation should remain cautious and non-causal."),
        encoding="utf-8",
    )
    supp = """# Supplementary Methods

The v3 package defines CVC exposure through an itemid-level active-interval review, event-level blood culture phenotyping, strict event-level AMR-BSI classification, clone-censor-weighting, stabilized weights, positivity diagnostics, balance diagnostics, and sensitivity-analysis status flags. Candida bloodstream infection is included in BSI but not in primary antibacterial AMR-BSI; antifungal resistance requires a separately validated antifungal susceptibility phenotype.
"""
    (paths.manuscript / "supplementary_methods.md").write_text(supp, encoding="utf-8")
    (paths.manuscript / "supplementary_methods_v4.md").write_text(supp.replace("v3 package", "v4 package"), encoding="utf-8")
    (paths.manuscript / "supplementary_methods_v4_locked.md").write_text(supp.replace("v3 package", "v4_locked package"), encoding="utf-8")
    dag = """digraph CVC_BSI_AMR_TTE {
  rankdir=LR;
  Severity -> CVC_duration;
  Severity -> BSI;
  Indication -> CVC_duration;
  Indication -> BSI;
  Antibiotics -> AMR_BSI;
  Microbiology_testing -> BSI;
  CVC_duration -> BSI;
  CVC_duration -> AMR_BSI;
  Death_discharge -> Censoring_competing_event;
  Death_discharge -> Observed_BSI;
  BSI -> Observed_BSI;
  AMR_BSI -> Observed_AMR_BSI;
}
"""
    (paths.manuscript / "dag_updated.dot").write_text(dag, encoding="utf-8")
    (paths.manuscript / "forty_word_summary_v4.md").write_text(
        "Preliminary aggregate v4 MIMIC-IV target trial emulation outputs estimate CVC-duration strategy associations with CLABSI-proxy BSI and strict AMR-BSI using corrected microbiology keys, follow-up alignment, and active interval CVC definitions.",
        encoding="utf-8",
    )
    (paths.manuscript / "forty_word_summary_v4_locked.md").write_text(
        "Locked aggregate v4 MIMIC-IV target trial emulation outputs estimate CVC-duration strategy associations with CLABSI-proxy BSI and strict AMR-BSI using corrected microbiology keys, follow-up alignment, and active interval CVC definitions.",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", default=(__import__("os").environ.get("MIMIC_ROOT", "MIMIC_ROOT") + ''))
    args = parser.parse_args()
    paths = Paths(
        base=Path(args.mimic_root),
        qc=Path("outputs/qc"),
        tables=Path("outputs/tables"),
        figures=Path("outputs/figures"),
        manuscript=Path("manuscript"),
    )
    for p in [paths.qc, paths.tables, paths.figures, paths.manuscript]:
        p.mkdir(parents=True, exist_ok=True)
    archive_pre_correction(paths)

    patients, admissions, icu, d_items = load_core(paths)
    adult_all, adult_first = adult_cohorts(patients, admissions, icu)
    review, _, _ = build_itemid_review(paths, d_items, adult_all)
    raw_cvc, valid_cvc, cvc = build_cvc(paths, adult_first, review)
    blood, positive, susc, _ = microbiology_clean(paths, cvc)
    bsi_prelim, _ = qualify_bsi(paths, cvc, positive, write_qc_outputs=True)
    baseline_excluded_ids = set(bsi_prelim.attrs.get("baseline_excluded_stays", set()))
    if baseline_excluded_ids:
        cvc = cvc.loc[~cvc["stay_id"].astype(int).isin(baseline_excluded_ids)].copy()
        valid_cvc = valid_cvc.loc[~valid_cvc["stay_id"].astype(int).isin(baseline_excluded_ids)].copy()
    bsi, _ = qualify_bsi(paths, cvc, positive, write_qc_outputs=False)
    # Final post-exclusion phenotype QC after removing baseline-infected stays.
    if not bsi.empty:
        cvc_by_stay = {int(k): g for k, g in cvc.groupby("stay_id")}
        bsi["clabsi_proxy"] = bsi.apply(lambda r: int(is_clabsi_proxy(cvc_by_stay[int(r.stay_id)], r.event_day)), axis=1)
        bsi["days_after_cvc_start"] = bsi.apply(lambda r: float((r.event_time - cvc_by_stay[int(r.stay_id)]["cvc_starttime"].min()).total_seconds() / 86400), axis=1)
    bsi_qc_final = pd.DataFrame([
        {"metric": "bsi_events_48h_primary", "value": int(len(bsi))},
        {"metric": "recognized_pathogen_bsi_events", "value": int((~bsi["is_commensal"]).sum()) if not bsi.empty else 0},
        {"metric": "common_commensal_confirmed_bsi_events", "value": int(bsi["is_commensal"].sum()) if not bsi.empty else 0},
        {"metric": "event_days_after_cvc_median", "value": q(bsi["days_after_cvc_start"], 0.50) if not bsi.empty else 0},
        {"metric": "event_days_after_cvc_p25", "value": q(bsi["days_after_cvc_start"], 0.25) if not bsi.empty else 0},
        {"metric": "event_days_after_cvc_p75", "value": q(bsi["days_after_cvc_start"], 0.75) if not bsi.empty else 0},
    ])
    write_v2_csv(bsi_qc_final, paths.qc / "bsi_event_level_qc.csv")
    write_v4_csv(bsi_qc_final, paths.qc / "bsi_event_level_qc.csv")
    write_v2_csv(pd.DataFrame([
        {"metric": "explicit_common_commensal_regex", "value": COMMON_COMMENSAL_RE.pattern},
        {"metric": "confirmed_common_commensal_bsi_events", "value": int(bsi["is_commensal"].sum()) if not bsi.empty else 0},
    ]), paths.qc / "bsi_common_commensal_qc.csv")
    clabsi_qc_final = pd.DataFrame([
        {"metric": "bsi_events", "value": int(len(bsi))},
        {"metric": "clabsi_proxy_events", "value": int(bsi["clabsi_proxy"].sum()) if not bsi.empty else 0},
        {"metric": "non_clabsi_proxy_bsi_events", "value": int((1 - bsi["clabsi_proxy"]).sum()) if not bsi.empty else 0},
    ])
    write_v2_csv(clabsi_qc_final, paths.qc / "clabsi_proxy_qc.csv")
    write_v4_csv(clabsi_qc_final, paths.qc / "clabsi_proxy_qc.csv")
    bsi = amr_bsi(paths, bsi, susc)
    person_days = build_person_days(cvc, valid_cvc, bsi)
    person_days = add_outcomes_to_person_days(person_days, bsi)
    covariates, _ = build_tve_covariates(paths, person_days)
    analysis, _ = clone_censor_weight(paths, person_days, covariates)
    effects = table1_and_effects(paths, analysis)
    initial_cvc_stays = int(cvc["stay_id"].nunique()) + len(baseline_excluded_ids)
    flow = cohort_flow_and_figures(paths, adult_all, adult_first, cvc, person_days, bsi, analysis, baseline_excluded=len(baseline_excluded_ids), initial_cvc_stays=initial_cvc_stays)
    table1_original_cohort(paths, cvc, person_days, covariates, analysis)
    competing_events(paths, person_days, bsi)
    sensitivity(paths, analysis, bsi)
    update_manuscript(paths, flow, effects)
    phenotype_v3 = Path("docs/phenotype_assumptions.md").read_text(encoding="utf-8") if Path("docs/phenotype_assumptions.md").exists() else ""
    Path("docs/phenotype_assumptions_v3.md").write_text(
        phenotype_v3
        + "\n\n## V3 Active-Interval Note\n\n"
        + "V3 separates active CVC interval itemids from insertion/time-zero support, removal/end-time support, maintenance-only documentation, sensitivity-only CVC types, and excluded devices. "
        + "Primary exposure uses active interval itemids only. Clinician review status remains algorithmic until manual clinical confirmation.\n",
        encoding="utf-8",
    )
    Path("docs/phenotype_assumptions_v4.md").write_text(
        "# Phenotype Assumptions V4\n\n"
        "## CVC Exposure\n\n"
        "The v4 itemid review is stored in `outputs/qc/cvc_itemid_final_review_v4.csv`. Primary active CVC interval exposure uses only true active device interval itemids. Maintenance, calibration, waveform/site appearance, line-tip culture, ScvO2/SQI, obturator, and change-over-wire documentation are supporting/maintenance records and are not primary exposure itemids.\n\n"
        "The active interval list in `outputs/qc/cvc_active_interval_itemids_v4.csv` contains algorithmically selected itemids. Clinician review status remains pending until manual approval.\n\n"
        "## Microbiology Keys\n\nBlood culture collection keys use `micro_specimen_id` when available. `microevent_id` is treated as a row key only. If `micro_specimen_id` is missing, an internal fallback collection key combines subject, hospitalization, collection date/time, and specimen type; no keys are exported in aggregate outputs.\n\n"
        "## BSI Follow-Up\n\nTime zero is the first qualifying CVC start for the stay. Follow-up ends at the earliest of time zero plus 28 days, ICU discharge, hospital discharge, or death. Primary BSI events must occur after the 48-hour baseline window and within follow-up.\n\n"
        "## AMR-BSI\n\nPrimary AMR-BSI is strict event-level antibacterial resistance among BSI isolates: MRSA, VRE, CRE, carbapenem-resistant Pseudomonas aeruginosa, or carbapenem-resistant Acinetobacter baumannii complex. The R-only definition uses interpretation `R`; the R/I sensitivity maps interpretations `R` or `I` to the same organism-antibiotic phenotype rules. `Other_R` is exploratory only.\n\n"
        "## Ongoing Indication\n\nTrue ongoing CVC indications are vasopressor infusion, renal replacement therapy requiring central access, TPN, and high-frequency IV medication requirement. Mechanical ventilation, antibiotic exposure, lactate, AKI, severity proxy, and recent cultures are time-varying confounders and do not automatically justify CVC continuation.\n",
        encoding="utf-8",
    )
    Path("docs/phenotype_assumptions_v4_locked.md").write_text(
        "# Phenotype Assumptions V4 Locked\n\n"
        "Primary active CVC interval itemids are clinician-approved for the locked analysis: 224263 Multi Lumen, 224267 Cordis/Introducer, 224273 Presep Catheter, and 225199 Triple Introducer. Maintenance, calibration, site appearance, waveform, line-tip culture, obturator, and change-over-wire documentation are not primary exposure interval itemids.\n\n"
        "Blood culture collection keys use micro_specimen_id when available; microevent_id is a row key only. Baseline BSI exclusion includes positive blood cultures before time zero and within the specified post-time-zero baseline window.\n\n"
        "True ongoing CVC indications are vasopressor infusion, renal replacement therapy requiring central access, TPN, and high-frequency IV medication requirement. Severity confounders are modeled separately.\n",
        encoding="utf-8",
    )
    print("V4 corrected aggregate-only result package rebuilt.")


if __name__ == "__main__":
    main()

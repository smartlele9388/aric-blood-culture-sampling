"""Write aggregate source/code/output provenance after the main raw run finishes."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from extract_day3_from_raw import digest

SMALL_RAW = ["hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/d_labitems.csv.gz",
             "icu/icustays.csv.gz", "icu/d_items.csv.gz"]
EXTRACTION_RAW = {"icu/procedureevents.csv.gz": "procedureevents.csv.gz",
                  "hosp/microbiologyevents.csv.gz": "microbiologyevents.csv.gz",
                  "hosp/diagnoses_icd.csv.gz": "diagnoses_icd.csv.gz"}
MAIN_RAW = {"hosp/labevents.csv.gz": "labevents.csv.gz",
            "icu/inputevents.csv.gz": "inputevents.csv.gz",
            "hosp/prescriptions.csv.gz": "prescriptions.csv.gz"}
SCRIPTS = [
    "scripts/rebuild_locked_mimic_package_v4.py",
    "scripts/run_host_vulnerability_day3_landmark.py",
    "scripts/run_host_vulnerability_indication_phase1_locked.py",
    "scripts/run_host_vulnerability_indication_phase2_adjusted.py",
    "src/analysis/host_vulnerability_indication.py",
    "src/analysis/host_vulnerability_advanced.py",
    "ARIC_revision_20260920/code/rebuild_labs.py",
    "ARIC_revision_20260920/code/rebuild_treatments.py",
    "ARIC_revision_20260920/code/run_reanalysis.py",
    "ARIC_revision_20260920/association_audit_20261003/code/coding_rules_final.py",
    "ARIC_revision_20260920/route_A/code/run_sampling_analysis.py",
]


def stat(path: Path, with_hash: bool = True):
    data = {"path": str(path), "bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()}
    if with_hash:
        data["sha256"] = digest(path)
    return data


def share_safe_count(value):
    if isinstance(value, int) and 0 < value < 11:
        return "1-10"
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raw, root = args.raw_root.resolve(), args.output_root.resolve()
    aggregate, secure = root / "aggregate", root / "secure_derived_inputs"
    extraction = json.loads((aggregate / "raw_extraction_checkpoint.json").read_text(encoding="utf-8"))
    main_ckpt = json.loads((aggregate / "main_risk_checkpoint.json").read_text(encoding="utf-8"))
    source = {}
    for name in SMALL_RAW:
        source[name] = stat(raw / name)
    for name, key in EXTRACTION_RAW.items():
        source[name] = stat(raw / name, False)
        source[name]["sha256"] = extraction["source_files_sha256"][key]
    for name, key in MAIN_RAW.items():
        source[name] = stat(raw / name, False)
        source[name]["sha256"] = main_ckpt["input_hashes"][key]
    project = Path(__file__).resolve().parent
    code = {name: stat(project / name) for name in SCRIPTS}
    for name in ["extract_day3_from_raw.py", "build_main_from_raw.py", "derive_competing_exits.py", "fit_main_cox_survival.R",
                 "fit_main_sampling.py", "create_lineage_manifest.py", "run.py"]:
        code[name] = stat(project / name)
    restricted_outputs = {path.name: stat(path) for path in sorted(secure.glob("*.csv.gz"))}
    aggregate_outputs = {path.name: stat(path) for path in sorted(aggregate.glob("*.csv"))}
    manifest = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "raw_root": str(raw), "output_root": str(root),
        "stage_provenance": {"main_raw_extraction_and_risk": "completed from original MIMIC-IV tables",
                             "competing_exits": "derived later from raw-built risk_final_day3 and landmark_cohort_day3; no additional raw-table scan",
                             "portable_fit_only_check": "refit from existing restricted stage outputs; not a second full raw execution"},
        "patient_level_outputs": "local secure_derived_inputs only; DO NOT distribute",
        "source_files": source, "code_files": code,
        "restricted_output_file_metadata": restricted_outputs,
        "aggregate_output_file_metadata": aggregate_outputs,
        "counts": {"extraction": {k:share_safe_count(v) for k,v in extraction.items()
                                   if isinstance(v,(int,float)) and k != "relevant_cancelled_only_specimens"},
                   "main_risk": {k:share_safe_count(v) for k,v in main_ckpt.items()
                                 if isinstance(v,(int,float)) and k != "cancelled_only_relevant_specimens"}},
    }
    (aggregate / "lineage_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Lineage manifest complete: {len(source)} raw files, {len(code)} code files, {len(restricted_outputs)} restricted outputs", flush=True)


if __name__ == "__main__":
    main()

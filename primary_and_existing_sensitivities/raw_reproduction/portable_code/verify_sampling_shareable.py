"""Audit the sharing views against the actual completed sampling execution."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.summary_output_root.resolve()
    shared = root / "shareable"
    local_qc = json.loads((root / "sampling_behavior_qc.json").read_text(encoding="utf-8"))
    public_qc = json.loads((shared / "sampling_behavior_qc.json").read_text(encoding="utf-8"))
    comparison = json.loads((root / "raw_stage_vs_first_run_comparison.json").read_text(encoding="utf-8"))
    test_log = (root / "sampling_portable_tests.log").read_text(encoding="utf-8-sig")
    checks = {
        "all_four_local_aggregate_CSVs_match_first_run": comparison["all_fields_agree"],
        "fourteen_tests_recorded_passed": "14 passed" in test_log,
        "patient_rows_not_exported": local_qc["patient_rows_exported"] is False,
        "already_corrected_score_not_incremented": local_qc["score_input"] == "risk_final_day3 already corrected; no delta applied",
        "raw_files_not_read": local_qc["raw_files_read_by_this_script"] is False,
        "cancellation_only_count_omitted_from_public_QC": "source_checked_cancelled_only_rows" not in public_qc,
        "whole_exit_cause_map_omitted_from_public_QC": "exit_cause_counts_all_stays" not in public_qc,
        "execution_adapter_hash_matches_current_source": local_qc["source_hashes_sha256"]["sampling_behavior_adapter"] == digest(Path(__file__).parent / "sampling_behavior_from_raw.py"),
    }
    forbidden_patient_columns = {"subject_id", "hadm_id", "stay_id", "collection_key", "event_time", "charttime", "first_sample_time", "landmark_time"}
    csv_summaries = {}
    for filename in comparison["files"]:
        frame = pd.read_csv(shared / filename)
        checks[f"no_patient_fields:{filename}"] = not bool(forbidden_patient_columns & set(frame.columns))
        csv_summaries[filename] = {"rows": len(frame), "columns": len(frame.columns), "sha256": digest(shared / filename)}
        if filename == "sampling_behavior_first_sampling_cif.csv":
            checks["both_unsampled_exit_columns_omitted_from_public_CIF"] = not bool({
                "administratively_censored_unsampled_by_horizon", "non_sampling_competing_exits_by_horizon"
            } & set(frame.columns))
    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "actual_execution_started_at": local_qc["started_at"],
              "actual_execution_completed_at": local_qc["completed_at"],
              "sharing_ready_files": csv_summaries,
              "scope": "aggregate all-field comparison, execution provenance, targeted tests and sharing exclusions; no new clinical inference"}
    (root / "sampling_public_view_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (shared / "sampling_public_view_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("Sampling sharing-view gate failed")


if __name__ == "__main__":
    main()

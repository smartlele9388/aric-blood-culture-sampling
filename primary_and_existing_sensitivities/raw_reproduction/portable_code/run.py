"""Portable single-entry CLI for the ARIC raw MIMIC-IV day-3 main analysis.

`full` executes raw extraction, feature/risk construction, one Poisson model,
one trusted R Cox model, and provenance. `fit-only` is an explicit checkpoint
smoke test using already-existing restricted stage outputs; it is not a raw
data reproduction and must never be reported as one.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CODE = Path(__file__).resolve().parent


def execute(label: str, command: list[str], logs: Path) -> None:
    log = logs / f"{label}.log"
    print(f"Starting {label}; log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=CODE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"{label} failed with exit code {code}; see {log}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Authorized MIMIC-IV 3.1 directory containing hosp/ and icu/")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New local work directory; patient-level files remain under secure_derived_inputs/")
    parser.add_argument("--rscript", type=Path, default=None,
                        help="Rscript executable with survival installed; defaults to PATH")
    parser.add_argument("--mode", choices=["full", "fit-only"], default="full")
    parser.add_argument("--check-paths-only", action="store_true",
                        help="Validate code/raw/output paths without reading raw tables or creating output")
    args = parser.parse_args()
    raw, out = args.raw_root.resolve(), args.output_root.resolve()
    if not (raw / "hosp/diagnoses_icd.csv.gz").is_file() or not (raw / "icu/procedureevents.csv.gz").is_file():
        parser.error("--raw-root does not contain expected authorized MIMIC-IV hosp/ and icu/ tables")
    if out == raw or out.is_relative_to(raw) or out == CODE or out.is_relative_to(CODE):
        parser.error("--output-root must be outside raw data and portable code directories")
    if args.check_paths_only:
        required = ["extract_day3_from_raw.py", "build_main_from_raw.py", "derive_competing_exits.py",
                    "fit_main_sampling.py", "fit_main_cox_survival.R", "create_lineage_manifest.py",
                    "scripts/rebuild_locked_mimic_package_v4.py",
                    "scripts/run_host_vulnerability_day3_landmark.py",
                    "ARIC_revision_20260920/code/run_reanalysis.py",
                    "ARIC_revision_20260920/route_A/code/run_sampling_analysis.py"]
        missing = [name for name in required if not (CODE / name).is_file()]
        if missing:
            parser.error(f"portable code tree incomplete: {missing}")
        build_source = (CODE / "build_main_from_raw.py").read_text(encoding="utf-8")
        if 'digest(PROJECT / "extract_day3_from_raw.py")' not in build_source:
            parser.error("portable build source hash does not resolve within code tree")
        print(json.dumps({"code_root": str(CODE), "raw_root": str(raw),
                          "output_root": str(out), "extract_code_hash_target": str(CODE / "extract_day3_from_raw.py"),
                          "required_code_files": len(required), "output_created": out.exists()}, indent=2), flush=True)
        return
    rscript = args.rscript or shutil.which("Rscript")
    if not rscript or not Path(rscript).is_file():
        parser.error("Rscript not found; provide --rscript PATH with survival installed")
    secure, aggregate, logs = out / "secure_derived_inputs", out / "aggregate", out / "logs"
    for directory in (secure, aggregate, logs):
        directory.mkdir(parents=True, exist_ok=True)
    extraction_ckpt = aggregate / "raw_extraction_checkpoint.json"
    main_ckpt = aggregate / "main_risk_checkpoint.json"
    if args.mode == "full" and (extraction_ckpt.exists() or main_ckpt.exists()):
        parser.error("full mode requires a new output directory without prior stage checkpoints; use fit-only only for existing verified inputs")
    if args.mode == "fit-only" and not (main_ckpt.exists() and
        (secure / "risk_final_day3.csv.gz").exists() and
        (secure / "sampling_intervals_final_day3.csv.gz").exists()):
        parser.error("fit-only requires existing main checkpoint, risk and sampling interval files")
    started = datetime.now().astimezone().isoformat()
    if args.mode == "full":
        execute("01_raw_extraction", [sys.executable, str(CODE / "extract_day3_from_raw.py"),
                "--raw-root", str(raw), "--output-root", str(out)], logs)
        execute("02_build_main_risk", [sys.executable, str(CODE / "build_main_from_raw.py"),
                "--raw-root", str(raw), "--output-root", str(out)], logs)
    execute("02b_competing_exits", [sys.executable, str(CODE / "derive_competing_exits.py"),
            "--output-root", str(out)], logs)
    execute("03_primary_sampling_poisson", [sys.executable, str(CODE / "fit_main_sampling.py"),
            "--output-root", str(out)], logs)
    execute("04_primary_cox_R", [str(rscript), str(CODE / "fit_main_cox_survival.R"), str(out)], logs)
    execute("05_lineage_manifest", [sys.executable, str(CODE / "create_lineage_manifest.py"),
            "--raw-root", str(raw), "--output-root", str(out)], logs)
    record = {"mode": args.mode, "started_at": started,
              "completed_at": datetime.now().astimezone().isoformat(),
              "raw_root": str(raw), "output_root": str(out), "code_root": str(CODE),
              "interpretation": "complete raw-source execution" if args.mode == "full" else
                                "fit-only checkpoint verification; not a raw-source execution",
              "stage_logs": [str(x) for x in sorted(logs.glob("*.log"))]}
    (aggregate / f"portable_pipeline_run_{args.mode.replace('-', '_')}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(f"Pipeline {args.mode} complete; aggregate outputs: {aggregate}", flush=True)


if __name__ == "__main__":
    main()

"""Fit the one primary patient-clustered Poisson sampling model from raw-built risk."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
source = PROJECT / "ARIC_revision_20260920/route_A/code/run_sampling_analysis.py"
spec = importlib.util.spec_from_file_location("aric_primary_sampling", source)
sampling = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sampling
spec.loader.exec_module(sampling)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    secure, aggregate = root / "secure_derived_inputs", root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    sampling.OUT = aggregate
    x = pd.read_csv(secure / "sampling_intervals_final_day3.csv.gz",
                    parse_dates=["day_start", "stop_datetime"])
    summary = sampling.validate_risk(x)
    effects, diagnostics = sampling.fit_poisson(
        x, "raw_reproduced_sampling_primary_specimens", "collection_count", sampling.PRIMARY
    )
    diagnostics["model_formula"] = sampling.PRIMARY
    diagnostics["offset"] = "log(actual interval duration_days)"
    diagnostics["cluster"] = "subject_id"
    (aggregate / "raw_reproduced_sampling_primary_diagnostics.json").write_text(
        json.dumps({"risk_summary": summary, "model": diagnostics}, indent=2), encoding="utf-8"
    )
    print(f"Main sampling Poisson complete; {len(x):,} intervals, {int(x.collection_count.sum()):,} specimens, {len(effects)} coefficients", flush=True)


if __name__ == "__main__":
    main()

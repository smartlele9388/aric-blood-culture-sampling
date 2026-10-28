"""Write targeted R PH sensitivity manifest; no patient rows exported.

The default reproducible sensitivity is the finite lactate-availability fit.
Pass ``--include-unresolved`` only to inspect the exploratory AKI/labs time
interactions, which were not identifiable in this risk set.
"""
import csv
import sys
from pathlib import Path

from prepare_r_batch_manifest import BASE, DELTA, HERE, MAIN, RISK

if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--include-unresolved"):
    raise SystemExit("Usage: python prepare_ph_manifest.py [--include-unresolved]")
specs = [("ph_lactate_time_final_r", MAIN + " + ph_lactate_available", "ph_lactate_time")]
if len(sys.argv) == 2:
    specs += [
        ("ph_aki_time_final_r", MAIN + " + ph_aki_assessable", "ph_aki_time"),
        ("ph_labs_time_final_r", MAIN + " + ph_lactate_available + ph_aki_assessable", "ph_labs_time"),
    ]
path = HERE / "ph_batch_manifest_local.tsv"
with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model_name", "input_path", "rhs", "score_delta_path", "transform"], delimiter="\t")
    writer.writeheader()
    for name, rhs, transform in specs:
        writer.writerow({"model_name": name, "input_path": str(RISK), "rhs": rhs,
                         "score_delta_path": str(DELTA), "transform": transform})
print(path)

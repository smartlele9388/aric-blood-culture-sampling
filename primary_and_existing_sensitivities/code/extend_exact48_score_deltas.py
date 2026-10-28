"""Fill the two exact-48-hour-only stays in the secure score-delta cache.

No patient-level data leave ``secure_derived_inputs``. The public audit file
contains counts and hashes only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path((__import__("os").environ.get("ARIC_PROJECT_ROOT", "ARIC_PROJECT_ROOT") + ''))
NEW = ROOT / "ARIC_revision_20261009"
SECURE = NEW / "secure_derived_inputs"
OUT = NEW / "source_audit"
OLD = ROOT / "ARIC_revision_20260920"
CODING = OLD / "association_audit_20261003" / "code" / "coding_rules_final.py"
ORIGINAL_CACHE = SECURE / "final_score_deltas_stay.csv.gz"
UNION_CACHE = SECURE / "final_score_deltas_stay_day3_exact48_union.csv.gz"
EXACT_RISK = OLD / "secure_derived_inputs" / "revised_risk_exact48h.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    spec = importlib.util.spec_from_file_location("coding_rules_final", CODING)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import frozen final coding rule")
    coding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coding)

    existing = pd.read_csv(ORIGINAL_CACHE)
    required = ["stay_id", "delta_cancer", "delta_meta", "delta_score"]
    assert list(existing.columns) == required
    assert existing.stay_id.is_unique
    assert existing.delta_score.eq(existing.delta_cancer + existing.delta_meta).all()

    risk = pd.read_csv(
        EXACT_RISK,
        usecols=["subject_id", "hadm_id", "stay_id", "cancer", "metastatic_cancer", "event", "exposure_vulnerability_burden_score", "exposure_vulnerability_category"],
    )
    original_flags = risk[["subject_id", "hadm_id", "stay_id", "cancer", "metastatic_cancer"]].drop_duplicates()
    assert original_flags.stay_id.is_unique, "Frozen cancer flags vary within stay"
    missing = original_flags.loc[~original_flags.stay_id.isin(existing.stay_id)]
    assert len(missing) == 2, f"Expected two exact-48h-only stays, got {len(missing)}"
    reconstructed = coding.stay_flag_deltas(missing)
    extra = reconstructed[required].copy()
    assert len(extra) == 2 and extra.stay_id.is_unique
    assert extra.delta_score.eq(extra.delta_cancer + extra.delta_meta).all()

    union = pd.concat([existing, extra], ignore_index=True)
    assert union.stay_id.is_unique
    assert len(union) == len(existing) + 2
    assert set(original_flags.stay_id).issubset(set(union.stay_id))
    assert union.set_index("stay_id").loc[existing.stay_id, required[1:]].reset_index(drop=True).equals(existing[required[1:]].reset_index(drop=True))
    union.to_csv(UNION_CACHE, index=False, compression="gzip")

    extra_risk = risk.loc[risk.stay_id.isin(extra.stay_id)].copy()
    corrected = coding.correct_frame(extra_risk, extra)
    summary = {
        "source_coding_rule": str(CODING),
        "original_day3_cache": str(ORIGINAL_CACHE),
        "new_secure_union_cache": str(UNION_CACHE),
        "exact48_risk": str(EXACT_RISK),
        "day3_cache_stays": int(len(existing)),
        "exact48_stays": int(original_flags.stay_id.nunique()),
        "exact48_stays_missing_from_day3_cache": int(len(extra)),
        "exact48_extra_risk_intervals": int(len(extra_risk)),
        "exact48_extra_events": int(extra_risk.event.sum()),
        "extra_delta_cancer_nonzero_stays": int(extra.delta_cancer.ne(0).sum()),
        "extra_delta_metastatic_nonzero_stays": int(extra.delta_meta.ne(0).sum()),
        "extra_delta_score_nonzero_stays": int(extra.delta_score.ne(0).sum()),
        "extra_exact48_intervals_changing_category": int(extra_risk.exposure_vulnerability_category.astype(str).ne(corrected.exposure_vulnerability_category.astype(str)).sum()),
        "union_cache_stays": int(len(union)),
        "input_day3_cache_sha256": sha256(ORIGINAL_CACHE),
        "input_exact48_risk_sha256": sha256(EXACT_RISK),
        "output_union_cache_sha256": sha256(UNION_CACHE),
        "note": "No patient IDs, diagnosis rows or event times in this public summary. Union cache remains in local restricted directory.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exact48_delta_extension_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["exact48_stays_missing_from_day3_cache", "exact48_extra_risk_intervals", "exact48_extra_events", "extra_delta_score_nonzero_stays", "extra_exact48_intervals_changing_category", "new_secure_union_cache"]}))


if __name__ == "__main__":
    main()

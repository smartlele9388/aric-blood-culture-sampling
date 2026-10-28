"""Re-export aggregate Poisson covariance archives without pickled object labels."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "ARIC_revision_20260920"
SOURCES = [
    BASE / "route_A/results",
    BASE / "association_audit_20261003/corrected_models_final",
    ROOT / "ARIC_revision_20261009/analysis_core/results",
]
OUT = HERE / "safe_poisson_covariance_npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for source in SOURCES:
        for npz in sorted(source.glob("sampling*_covariance.npz")):
            stem = npz.name.removesuffix("_covariance.npz")
            csv = source / (stem + "_coefficients.csv")
            if not csv.exists():
                raise FileNotFoundError(csv)
            table = pd.read_csv(csv)
            terms = np.array(table.term.astype(str).tolist(), dtype=np.str_)
            with np.load(npz, allow_pickle=False) as original:
                beta = original["coefficients"].astype(float)
                covariance = original["covariance"].astype(float)
            assert len(terms) == len(beta) == covariance.shape[0] == covariance.shape[1]
            assert np.allclose(beta, table.log_rate_ratio.to_numpy(dtype=float), atol=1e-12)
            outfile = OUT / (source.parent.name + "__" + source.name + "__" + npz.name)
            np.savez_compressed(outfile, terms=terms, coefficients=beta, covariance=covariance)
            with np.load(outfile, allow_pickle=False) as verified:
                assert verified["terms"].tolist() == terms.tolist()
                assert np.array_equal(verified["coefficients"], beta)
                assert np.array_equal(verified["covariance"], covariance)
            rows.append({"model": stem, "source_csv": str(csv), "source_npz": str(npz),
                         "source_sha256": sha256(npz), "safe_npz": str(outfile),
                         "safe_sha256": sha256(outfile), "terms": len(terms),
                         "numeric_arrays_unchanged": True, "allow_pickle_required_to_read_safe_archive": False})
    pd.DataFrame(rows).to_csv(HERE / "safe_poisson_covariance_manifest.csv", index=False)
    print(json.dumps({"archives": len(rows), "all_allow_pickle_false": True}, indent=2))


if __name__ == "__main__":
    main()

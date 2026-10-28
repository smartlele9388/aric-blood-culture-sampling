"""No-scan path smoke tests for the portable ARIC main pipeline."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


CODE = Path(__file__).resolve().parents[1]


def test_new_output_directory_is_not_created_during_path_check():
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        raw = tmp / "raw"
        (raw / "hosp").mkdir(parents=True)
        (raw / "icu").mkdir(parents=True)
        (raw / "hosp/diagnoses_icd.csv.gz").touch()
        (raw / "icu/procedureevents.csv.gz").touch()
        output = tmp / "new-output"
        command = [sys.executable, str(CODE / "run.py"), "--mode", "full",
                   "--check-paths-only", "--raw-root", str(raw),
                   "--output-root", str(output)]
        result = subprocess.run(command, cwd=tmp, text=True, capture_output=True, check=True)
        record = json.loads(result.stdout)
        assert not output.exists()
        assert record["output_created"] is False
        assert Path(record["extract_code_hash_target"]).resolve() == CODE / "extract_day3_from_raw.py"
        assert record["required_code_files"] >= 10


def test_imports_resolve_only_inside_portable_tree():
    # The working directory is an unrelated temporary directory, not the old project.
    with tempfile.TemporaryDirectory() as temporary:
        code = (
            "import sys;from pathlib import Path;"
            f"p=Path({str(CODE)!r});sys.path.insert(0,str(p));"
            "import extract_day3_from_raw as e,build_main_from_raw as b,fit_main_sampling as f;"
            "from scripts import rebuild_locked_mimic_package_v4 as l;"
            "from src.analysis import host_vulnerability_indication as s;"
            "assert all(Path(m.__file__).resolve().is_relative_to(p) for m in (e,b,f,l,s));"
            "print('portable imports passed')"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=temporary,
                                text=True, capture_output=True, check=True)
        assert "portable imports passed" in result.stdout

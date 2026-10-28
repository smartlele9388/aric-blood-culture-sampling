"""Synthetic unit fixtures only; no research observations or estimates are invented."""
from pathlib import Path
import importlib.util
import tempfile

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1] / "sampling_behavior_from_raw.py"
spec = importlib.util.spec_from_file_location("portable_sampling_behavior", CODE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_baseline_is_first_interval_and_corrected_score_not_incremented_again():
    times = pd.to_datetime(["2000-01-02", "2000-01-01", "2000-01-01", "2000-01-01"])
    risk = pd.DataFrame({"stay_id": [1, 1, 2, 3], "subject_id": [10, 10, 20, 30],
                         "hadm_id": [100, 100, 200, 300], "day_start": times,
                         "landmark_time": pd.Timestamp("2000-01-01"),
                         "exposure_vulnerability_burden_score": [8, 2, 1, 4],
                         "delta_score": [1, 1, 1, 1]})
    baseline = m.freeze_baseline_groups(risk)
    assert baseline.baseline_score.tolist() == [2, 1, 4]
    assert baseline.baseline_group.tolist() == ["moderate", "low", "high"]
    assert baseline.stay_id.is_unique


def test_missing_baseline_score_rejected():
    risk = pd.DataFrame({"stay_id": [1], "subject_id": [10], "hadm_id": [100],
                         "day_start": [pd.Timestamp("2000-01-01")],
                         "landmark_time": [pd.Timestamp("2000-01-01")],
                         "exposure_vulnerability_burden_score": [np.nan]})
    with pytest.raises(RuntimeError, match="missing, nonfinite"):
        m.freeze_baseline_groups(risk)


def test_terminal_sampling_precedes_same_time_exit_and_administrative_censor():
    stay = pd.DataFrame({"sampled": [True, False, False], "first_sample_day": [1, np.nan, np.nan],
                         "followup_days": [1, 1, 28], "cause": [1, 2, 0]})
    curve = m.aalen_johansen_first_sampling(stay, (1, 3, 28))
    np.testing.assert_allclose(list(curve.values()), [1 / 3] * 3, atol=1e-15)


def test_aalen_johansen_handles_censoring_rather_than_always_using_direct_fraction():
    # This pre-28 censoring fixture is rejected by main's cohort gate, but the
    # independent AJ routine must still handle censoring correctly.
    stay = pd.DataFrame({"sampled": [True, False, False], "first_sample_day": [2, np.nan, np.nan],
                         "followup_days": [2, 1, 1], "cause": [1, 2, 0]})
    assert m.aalen_johansen_first_sampling(stay, (2,))[2] == pytest.approx(2 / 3)


def test_all_field_comparison_detects_changed_value_but_allows_float_roundoff():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prior, current = root / "prior", root / "current"
        prior.mkdir(); current.mkdir()
        frame = pd.DataFrame({"group": ["synthetic_only"], "count": [100], "fraction": [.2]})
        for filename in m.SUMMARY_FILES:
            frame.to_csv(prior / filename, index=False)
            rounded = frame.copy()
            rounded["fraction"] = np.nextafter(.2, 1)
            rounded.to_csv(current / filename, index=False)
        assert m.reconcile_summaries(prior, current)["all_fields_agree"]
        changed = frame.copy(); changed["count"] = 101
        changed.to_csv(current / m.SUMMARY_FILES[0], index=False)
        assert not m.reconcile_summaries(prior, current)["all_fields_agree"]

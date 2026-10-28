"""Synthetic unit tests only: no research records are fabricated or modified."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT = Path(__file__).resolve().parents[1] / 'code/run_sampling_analysis.py'
spec = importlib.util.spec_from_file_location('sampling_analysis', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def risk():
    start = pd.Timestamp('2000-01-01')
    return pd.DataFrame({
        'subject_id': [1, 1], 'hadm_id': [11, 11], 'stay_id': [111, 111],
        'day_start': [start, start + pd.Timedelta(days=1)],
        'stop_datetime': [start + pd.Timedelta(days=1), start + pd.Timedelta(days=1.5)],
        't_start': [0., 1.], 't_stop': [1., 1.5], 'duration_days': [1., .5],
        'event': [0, 1],
    })


def collections(times, subjects=None, admissions=None):
    n = len(times)
    return pd.DataFrame({'collection_key': ['synthetic_test_' + str(i) for i in range(n)],
                         'subject_id': subjects if subjects is not None else [1] * n,
                         'hadm_id': admissions if admissions is not None else [11] * n,
                         'event_time': pd.to_datetime(times, format='mixed'), 'positive': [1] * n})


class BoundaryTests(unittest.TestCase):
    def test_internal_boundary_forward_final_event_included(self):
        c = collections(['2000-01-01', '2000-01-02', '2000-01-02 12:00:00'])
        a = m.assign_collections(risk(), c)
        self.assertEqual(a.risk_row.tolist(), [0, 1, 1])
        self.assertEqual(a.collection_key.nunique(), 3)

    def test_outside_window_and_wrong_linkage_not_assigned(self):
        c = collections(['1999-12-31 23:59:59', '2000-01-02 12:00:01',
                         '2000-01-01 12:00:00', '2000-01-01 12:00:00'],
                        subjects=[1, 1, 2, 1], admissions=[11, 11, 11, 22])
        self.assertEqual(len(m.assign_collections(risk(), c)), 0)

    def test_no_cross_stay_or_admission_assignment(self):
        r = risk()
        other = r.assign(subject_id=2, hadm_id=22, stay_id=222)
        r = pd.concat([r, other], ignore_index=True)
        c = collections(['2000-01-01 12:00:00'] * 2, subjects=[1, 2], admissions=[11, 22])
        self.assertEqual(m.assign_collections(r, c).risk_row.tolist(), [0, 2])

    def test_duplicate_keys_rejected(self):
        c = collections(['2000-01-01'] * 2)
        c['collection_key'] = 'synthetic_duplicate'
        with self.assertRaisesRegex(ValueError, 'globally unique'):
            m.assign_collections(risk(), c)

    def test_overlap_and_touching_stays_rejected(self):
        for hours in (24, 36):
            r = risk()
            other = r.iloc[:1].copy().assign(stay_id=112, event=0)
            other['day_start'] = pd.Timestamp('2000-01-01') + pd.Timedelta(hours=hours)
            other['stop_datetime'] = other.day_start + pd.Timedelta(days=1)
            other['t_start'] = 0.; other['t_stop'] = 1.; other['duration_days'] = 1.
            with self.assertRaisesRegex(ValueError, 'overlap or touch'):
                m.validate_risk(pd.concat([r, other], ignore_index=True))

    def test_actual_duration_and_timeband_mismatch_rejected(self):
        r = risk(); r.loc[1, 'duration_days'] = 1.
        with self.assertRaisesRegex(ValueError, 'datetime duration'):
            m.validate_risk(r)
        r = risk().iloc[:1].copy()
        r.loc[0, 'stop_datetime'] = r.day_start.iloc[0] + pd.Timedelta(days=3)
        r.loc[0, 'duration_days'] = 3.; r.loc[0, 't_stop'] = 3.
        with self.assertRaisesRegex(ValueError, 'time-band boundary'):
            m.validate_risk(r)

    def test_timestamp_occasion_collapses_specimens_not_positive_status(self):
        c = collections(['2000-01-01 12:00:00'] * 2 + ['2000-01-02 12:00:00'])
        c['positive'] = [0, 1, 0]
        occ = m.collapse_occasions(m.assign_collections(risk(), c))
        self.assertEqual(occ.specimens.tolist(), [2, 1])
        self.assertEqual(occ.positive.tolist(), [1, 0])

    def test_all_rows_decide_cancelled_only_and_negative_is_retained(self):
        raw = pd.DataFrame({
            'subject_id': [1] * 6, 'hadm_id': [11] * 6,
            'micro_specimen_id': [1, 1, 2, 2, 3, 4],
            'chartdate': ['2000-01-01'] * 6, 'charttime': ['2000-01-01 12:00:00'] * 6,
            'spec_type_desc': ['BLOOD CULTURE'] * 6,
            'org_name': ['CANCELLED', 'cancelled', 'cancelled', 'Escherichia coli', 'NO GROWTH', None],
        })
        prior = collections(['2000-01-01 12:00:00'] * 4)
        prior['collection_key'] = ['specimen:' + str(i) for i in range(1, 5)]
        prior['positive'] = [0, 1, 0, 0]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); raw_path = d / 'synthetic_raw.csv'; raw.to_csv(raw_path, index=False)
            with patch.object(m, 'RAW', raw_path), patch.object(m, 'OUT', d), patch.object(m, 'SEC', d):
                master, audit = m.raw_collection_audit(risk(), prior, {'synthetic_test': 'only'})
        self.assertEqual(master.has_non_cancelled.tolist(), [0, 1, 1, 1])
        self.assertEqual(master.positive.tolist(), [0, 1, 0, 0])
        self.assertEqual(audit['relevant_cancelled_only_specimens'], 1)
        self.assertEqual(audit['relevant_mixed_cancelled_and_noncancelled'], 1)

    def test_poisson_cluster_covariance_matches_independent_sandwich(self):
        # Synthetic repeated observations with unequal exposure: validate robust
        # covariance and offset handling against the explicit score sandwich.
        group = np.repeat(np.arange(12), 4)
        z = np.tile([0., 0., 1., 1.], 12)
        days = np.tile([.2, .8, .5, 1.], 12)
        y = np.tile([0., 1., 1., 2.], 12) + np.repeat(np.arange(12) % 3, 4)
        design = np.column_stack([np.ones(len(y)), z])
        model = sm.GLM(y, design, family=sm.families.Poisson(), offset=np.log(days))
        fit = model.fit(cov_type='cluster', cov_kwds={'groups': group, 'use_correction': True})
        score = design * (y - fit.fittedvalues)[:, None]
        cluster_scores = np.vstack([score[group == g].sum(axis=0) for g in np.unique(group)])
        bread = np.linalg.inv(design.T @ (fit.fittedvalues[:, None] * design))
        correction = 12 / 11 * (len(y) - 1) / (len(y) - 2)
        covariance = correction * bread @ cluster_scores.T @ cluster_scores @ bread
        np.testing.assert_allclose(fit.cov_params(), covariance, rtol=1e-8, atol=1e-12)
        expected_rate = [y[z == level].sum() / days[z == level].sum() for level in [0, 1]]
        self.assertAlmostEqual(np.exp(fit.params[1]), expected_rate[1] / expected_rate[0], places=8)


if __name__ == '__main__':
    unittest.main(verbosity=2)

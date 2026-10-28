"""Small fabricated boundary fixtures for software checks, not research simulations."""
import unittest
from validate_cohort import audit

def fixtures():
    result=[]
    for index,(status,time) in enumerate([(1,1.0),(2,2.0),(3,3.0),(4,4.0),(0,28.0)]):
        sampled = index in (0,2)
        result.append(dict(subject_id=str(index+1),hadm_id=str(index+11),stay_id=str(index+101),
            baseline_group=["low","moderate","high","low","high"][index],
            age_per_10y=0,female=index%2,icu_group="medical",admission_group="emergency",
            bsi_time_days=time,bsi_status=status,
            sampling_time_days=(time if index==0 else 0 if index==2 else time),
            sampling_status=(1 if sampled else status),specimens=int(sampled),
            occasions=int(sampled),risk_intervals=1,risk_duration_days=time))
    return result

class BoundaryTests(unittest.TestCase):
    def test_zero_time_sampling_and_terminal_tie_are_kept(self):
        out=audit(fixtures(),"structural")
        self.assertTrue(out["passed"],out["errors"])
        self.assertEqual(out["summary"]["zero_time_first_samples"],1)
        self.assertEqual(out["summary"]["first_sampling_at_terminal_BSI"],1)
    def test_bsi_requires_sampling(self):
        rows=fixtures(); rows[0].update(sampling_status=1,specimens=0,occasions=0)
        self.assertFalse(audit(rows,"structural")["passed"])
    def test_competing_exit_cannot_be_recoded_as_early_admin_censoring(self):
        rows=fixtures(); rows[1].update(bsi_status=0,sampling_status=0)
        self.assertFalse(audit(rows,"structural")["passed"])
    def test_duplicate_admission_is_rejected(self):
        rows=fixtures(); rows[1]["hadm_id"]=rows[0]["hadm_id"]
        self.assertFalse(audit(rows,"structural")["passed"])
    def test_repeat_admission_by_one_patient_is_retained(self):
        rows=fixtures(); rows[1]["subject_id"]=rows[0]["subject_id"]
        out=audit(rows,"structural")
        self.assertTrue(out["passed"],out["errors"])
        self.assertEqual(out["summary"]["patients"],4)
        self.assertEqual(out["summary"]["stays"],5)
    def test_no_post_exit_sample(self):
        rows=fixtures(); rows[0]["sampling_time_days"]=1.1
        self.assertFalse(audit(rows,"structural")["passed"])
    def test_structural_fixture_cannot_pass_real_cohort_gate(self):
        self.assertFalse(audit(fixtures(),"S0")["passed"])
    def test_centered_adult_age_and_mismatched_age_coding(self):
        rows=fixtures(); rows[0].update(age_per_10y=-4.2, age=18)
        self.assertTrue(audit(rows,"structural")["passed"])
        rows[0]["age_per_10y"]=1.8
        self.assertFalse(audit(rows,"structural")["passed"])
    def test_admission_age_does_not_redefine_anchor_age_eligibility(self):
        rows=fixtures(); rows[0].update(age_per_10y=-4.3,age=17)
        self.assertTrue(audit(rows,"structural")["passed"])
    def test_missing_input_reports_include_scope(self):
        for rows in ([],[{}]):
            result=audit(rows,"structural")
            self.assertFalse(result["passed"])
            self.assertIn("scope",result)

if __name__ == "__main__": unittest.main()

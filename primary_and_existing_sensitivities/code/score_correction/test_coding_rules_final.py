import unittest
import numpy as np
import pandas as pd
from coding_rules_final import classify,correct_frame

class TestFinalCoding(unittest.TestCase):
    def test_version_aware_neoplasms(self):
        cases=[('199.0',9,(1,1)),('199.1',9,(1,0)),('C80.0',10,(1,1)),('C80.1',10,(1,0)),
               ('C7B.1',10,(1,1)),('209.70',9,(1,1)),('209.40',9,(0,0)),('203.00',9,(1,0)),
               ('173.3',9,(1,0)),('D3A.0',10,(0,0))]
        for code,version,expected in cases:
            with self.subTest(code=code):self.assertEqual(classify(code,version),expected)

    def test_score_and_category_invariants(self):
        delta=pd.DataFrame({'stay_id':[11,12],'delta_cancer':[1,0],'delta_meta':[0,-1],'delta_score':[1,-1]})
        frame=pd.DataFrame({'stay_id':[11,12],'cancer':[0,1],'metastatic_cancer':[0,1],
             'corrected_chronic_score':[1,2],'corrected_score':[1,2],'lag_corrected_score':[1,4],
             'exposure_vulnerability_burden_score':[1,4],'exposure_vulnerability_category':['low','high'],
             'vmod':[0,0],'vhigh':[0,1]})
        out=correct_frame(frame,delta)
        self.assertEqual(out.exposure_vulnerability_burden_score.tolist(),[2,3])
        self.assertEqual(out.exposure_vulnerability_category.astype(str).tolist(),['moderate','moderate'])
        self.assertEqual(out.vmod.tolist(),[1,1]);self.assertEqual(out.vhigh.tolist(),[0,0])
        self.assertEqual(out.cancer.tolist(),[1,1]);self.assertEqual(out.metastatic_cancer.tolist(),[0,0])

if __name__=='__main__':unittest.main()

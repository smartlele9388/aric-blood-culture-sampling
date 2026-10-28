"""Check primary design/timing fields against the prior correctly scored risk.

The earlier risk cache predates ICD correction, so apply the frozen prior delta
before comparing the score. This reads only existing local stage inputs, does
not fit models or scan raw tables, and exports aggregate mismatch counts only.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    root=a.output_root.resolve();workspace=root.parents[1]
    new_path=root/'secure_derived_inputs/risk_final_day3.csv.gz'
    old_path=workspace/'ARIC_revision_20260920/secure_derived_inputs/revised_risk_day3.csv.gz'
    delta_path=workspace/'ARIC_revision_20261009/secure_derived_inputs/final_score_deltas_stay.csv.gz'
    rules=root/'portable_code/ARIC_revision_20260920/association_audit_20261003/code/coding_rules_final.py'
    spec=importlib.util.spec_from_file_location('frozen_coding_comparison',rules)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=module.correct_frame(pd.read_csv(old_path),pd.read_csv(delta_path))
    new=pd.read_csv(new_path)
    keys=['stay_id','t_start']
    fields=['subject_id','hadm_id','t_stop','event','duration_days','landmark_time','day_start','stop_datetime',
            'exposure_vulnerability_burden_score','exposure_vulnerability_category','exposure_indication_present',
            'age_per_10y','female','icu_group','admission_group','multiple_line','cvc_type_group',
            'exposure_antibiotic_exposure','exposure_lactate_available','exposure_aki_assessable',
            'exposure_mechanical_ventilation','lag_aki','lag_lactate_high']
    joined=new[keys+fields].merge(old[keys+fields],on=keys,how='outer',suffixes=('_new','_prior'),
                                  validate='one_to_one',indicator=True)
    matched=joined._merge.eq('both')
    mismatches={}
    for name in fields:
        l=joined.loc[matched,name+'_new'];r=joined.loc[matched,name+'_prior']
        if name in ('landmark_time','day_start','stop_datetime'):
            bad=pd.to_datetime(l).ne(pd.to_datetime(r))
        elif pd.api.types.is_numeric_dtype(l) and pd.api.types.is_numeric_dtype(r):
            bad=~np.isclose(l,r,rtol=0,atol=1e-10,equal_nan=True)
        else: bad=l.astype('string').fillna('<missing>').ne(r.astype('string').fillna('<missing>'))
        n=int(bad.sum());mismatches[name]=n if n==0 or n>=11 else '<11 suppressed'
    result={'comparison':'raw-built final primary design/timing fields versus prior risk after frozen ICD delta',
            'new_intervals':len(new),'prior_intervals':len(old),'matched_risk_keys':int(matched.sum()),
            'new_only_keys':int(joined._merge.eq('left_only').sum()),
            'prior_only_keys':int(joined._merge.eq('right_only').sum()),
            'field_disagreements':mismatches,
            'all_primary_design_and_timing_fields_match':all(v==0 for v in mismatches.values()),
            'source_hashes':{'new_risk':sha(new_path),'prior_risk':sha(old_path),'prior_final_delta':sha(delta_path),
                             'frozen_scoring_rules':sha(rules),'comparison_code':sha(Path(__file__))},
            'privacy':'No patient identifiers, dates or rows exported'}
    (root/'aggregate/primary_design_timing_reconciliation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

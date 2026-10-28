"""Remove heterogeneous immune-code score point; export only aggregate/model output."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'ARIC_revision_20260920';ROUTE=BASE/'route_A'
OUT=HERE/'results';SEC=HERE.parent/'secure_derived_inputs'
OUT.mkdir(exist_ok=True);SEC.mkdir(exist_ok=True)
sys.path.insert(0,str(BASE/'association_audit_20261003/code'))
from coding_rules_final import correct_frame

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def remove_point(frame):
    result=frame.copy();flag=result.immunosuppression.astype(int)
    for col in ['corrected_chronic_score','corrected_score','lag_corrected_score',
                'exposure_vulnerability_burden_score','documented_chronic_count']:
        if col in result:result[col]=result[col].astype(int)-flag
    score=result.exposure_vulnerability_burden_score
    assert score.ge(0).all()
    result['exposure_vulnerability_category']=pd.Categorical(
        np.where(score<=1,'low',np.where(score<=3,'moderate','high')),
        categories=['low','moderate','high'])
    for col,val in [('vmod','moderate'),('vhigh','high')]:
        if col in result:result[col]=result.exposure_vulnerability_category.eq(val).astype(int)
    return result

def main():
    d=pd.read_csv(SEC/'final_score_deltas_stay.csv.gz')
    r=pd.read_csv(BASE/'secure_derived_inputs/revised_risk_day3.csv.gz')
    s=pd.read_csv(ROUTE/'secure_derived_inputs/sampling_intervals_day3.csv.gz')
    r=correct_frame(r,d);s=correct_frame(s,d)
    assert len(r)==len(s)==60272 and r.stay_id.eq(s.stay_id).all()
    r2=remove_point(r);s2=remove_point(s)
    changed=r.exposure_vulnerability_category.astype(str).ne(r2.exposure_vulnerability_category.astype(str))
    trans=pd.DataFrame({'historical_corrected':r.exposure_vulnerability_category.astype(str),
                        'without_immune_flag':r2.exposure_vulnerability_category.astype(str),'event':r.event})
    trans.groupby(['historical_corrected','without_immune_flag'],as_index=False).agg(
        intervals=('event','size'),events=('event','sum')).to_csv(OUT/'immune_flag_category_transitions.csv',index=False)
    summary={'immune_code_positive_stays':int(r.loc[r.immunosuppression.eq(1),'stay_id'].nunique()),
             'immune_code_positive_intervals':int(r.immunosuppression.eq(1).sum()),
             'category_changed_intervals':int(changed.sum()),
             'category_changed_stays':int(r.loc[changed,'stay_id'].nunique()),
             'events_in_changed_category_intervals':int(r.loc[changed,'event'].sum()),
             'intervals_unchanged':len(r),'events_unchanged':int(r.event.sum()),'patient_rows_exported':False}
    (OUT/'immune_flag_sensitivity_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Local file required for independent Efron-cluster Cox refit; excluded from all delivery staging.
    r2.to_csv(SEC/'immune_flag_excluded_day3_risk.csv.gz',index=False)
    sampling=module(ROUTE/'code/run_sampling_analysis.py','sampling_original');sampling.OUT=OUT
    _,diag=sampling.fit_poisson(s2,'sampling_without_immune_flag','collection_count',sampling.PRIMARY)
    pd.DataFrame([diag]).to_csv(OUT/'immune_flag_sampling_diagnostics.csv',index=False)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

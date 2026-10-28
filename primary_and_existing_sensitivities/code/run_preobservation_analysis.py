"""Pre-landmark and prior-window acute-only associations; model-level output."""
from pathlib import Path
import importlib.util,hashlib,json,platform,sys
import numpy as np
import pandas as pd
import scipy,statsmodels

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'ARIC_revision_20260920'
ROUTE=BASE/'route_A'
OUT=HERE/'results';OUT.mkdir(parents=True,exist_ok=True)
SEC=HERE.parent/'secure_derived_inputs';SEC.mkdir(parents=True,exist_ok=True)

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def freeze_at_landmark(frame,fields):
    first=frame.sort_values(['stay_id','t_start','t_stop']).drop_duplicates('stay_id').set_index('stay_id')
    if not np.allclose(first.t_start,0,atol=1e-8):
        raise ValueError('Some stays have no risk interval at the landmark')
    result=frame.copy()
    for field in fields:
        result[field]=result.stay_id.map(first[field])
        if result[field].isna().any():raise ValueError('Baseline field missing: '+field)
    return result

def main():
    rev=load_module(BASE/'code/run_reanalysis.py','frozen_revision');rev.OUT=OUT
    sampling=load_module(ROUTE/'code/run_sampling_analysis.py','frozen_sampling');sampling.OUT=OUT
    risk_path=BASE/'secure_derived_inputs/revised_risk_day3.csv.gz'
    sample_path=ROUTE/'secure_derived_inputs/sampling_intervals_day3.csv.gz'
    r=pd.read_csv(risk_path)
    s=pd.read_csv(sample_path)
    assert (len(r),r.stay_id.nunique(),r.subject_id.nunique(),int(r.event.sum()))==(60272,13863,12757,268)
    assert len(s)==len(r) and s.event.sum()==r.event.sum() and np.isclose(s.duration_days.sum(),r.duration_days.sum())
    for frame in [r,s]:
        frame['exposure_creatinine_rise_documented']=frame.lag_aki.eq(1).astype(int)
        frame['exposure_lactate_high_documented']=frame.lag_lactate_high.eq(1).astype(int)
    s['exposure_mechanical_ventilation']=s.prior_ventilation.astype(int)
    # The first eligible risk interval corresponds to the day-3 landmark.
    fields=['exposure_mechanical_ventilation','exposure_creatinine_rise_documented',
            'exposure_lactate_high_documented','exposure_indication_present','multiple_line',
            'exposure_antibiotic_exposure','exposure_lactate_available','exposure_aki_assessable']
    baseline_r=freeze_at_landmark(r,fields)
    baseline_s=freeze_at_landmark(s,fields)
    # These patient-level model frames stay local and are excluded from delivery.
    baseline_r.to_csv(SEC/'preobservation_landmark_day3_risk.csv.gz',index=False)
    r.to_csv(SEC/'preobservation_daily_prior_window_day3_risk.csv.gz',index=False)
    for frame in [baseline_s,s]:
        frame['prior_ventilation']=frame.exposure_mechanical_ventilation
        frame['prior_creatinine_rise']=frame.exposure_creatinine_rise_documented
        frame['prior_high_lactate']=frame.exposure_lactate_high_documented
    cox_rhs=('exposure_mechanical_ventilation + exposure_creatinine_rise_documented + '
             'exposure_lactate_high_documented + exposure_indication_present + '+rev.ADJUST)
    poisson_rhs=('prior_ventilation + prior_creatinine_rise + prior_high_lactate + '
                 'exposure_indication_present + '+sampling.ADJUST+' + '+sampling.TIME)
    checks=[]
    for name,frame in [('landmark_only',baseline_r),('daily_prior_window',r)]:
        fit,diag=rev.fit(frame,'bsi_acute_'+name,cox_rhs)
        diag['formula']=cox_rhs;diag['information_timing']=name
        checks.append(diag)
    pchecks=[]
    for name,frame in [('landmark_only',baseline_s),('daily_prior_window',s)]:
        for kind,outcome in [('specimens','collection_count'),('occasions','occasion_count')]:
            _,diag=sampling.fit_poisson(frame,'sampling_acute_'+name+'_'+kind,outcome,poisson_rhs)
            diag['formula']=poisson_rhs;diag['information_timing']=name
            pchecks.append(diag)
    pd.DataFrame(checks).to_csv(OUT/'cox_model_diagnostics.csv',index=False)
    pd.DataFrame(pchecks).to_csv(OUT/'sampling_model_diagnostics.csv',index=False)
    summary={'question':'Association of documented pre-landmark acute treatment and laboratory state with sampling and detected proxy BSI, without any discharge-diagnosis flag',
             'risk_intervals':len(r),'stays':int(r.stay_id.nunique()),'patients':int(r.subject_id.nunique()),
             'BSI_events':int(r.event.sum()),'risk_days':float(r.duration_days.sum()),
             'specimens':int(s.collection_count.sum()),'sampling_occasions':int(s.occasion_count.sum()),
             'baseline_exposure_fields':fields,'cox_formula':cox_rhs,'poisson_formula':poisson_rhs,
             'source_hashes':{'risk_day3':sha(risk_path),'sampling_intervals_day3':sha(sample_path),
                              'run_reanalysis':sha(BASE/'code/run_reanalysis.py'),
                              'run_sampling_analysis':sha(ROUTE/'code/run_sampling_analysis.py'),
                              'this_script':sha(Path(__file__))},
             'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                         'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},
             'patient_rows_exported':False,'original_inputs_modified':False}
    (OUT/'preobservation_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'counts':{k:summary[k] for k in ['risk_intervals','stays','patients','BSI_events','risk_days','specimens','sampling_occasions']},
                      'cox_models':[d['model'] for d in checks],
                      'sampling_models':[d['model'] for d in pchecks]},indent=2))

if __name__=='__main__':main()

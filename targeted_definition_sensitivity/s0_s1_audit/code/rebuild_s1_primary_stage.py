"""Rebuild complete S1 input from candidates and verified full-window features.

Common full day windows must match the existing raw-built stage exactly before
their features may be reused. Newly eligible windows are extracted from raw
sources once. S1 intervals, terminal times and collections are rebuilt, never
copied from the old event-truncated risk table.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def verify_windows(old,new):
    keys=['stay_id','local_day'];old=old.sort_values(keys).reset_index(drop=True);new=new.sort_values(keys).reset_index(drop=True)
    assert old[keys].equals(new[keys]),'Common stay/window keys changed: cached features cannot be reused'
    rows={}
    for col in old:
        if col not in new:raise AssertionError('Rebuilt windows missing original field '+col)
        eq=old[col].eq(new[col])|(old[col].isna()&new[col].isna());rows[col]=int((~eq).sum())
    if any(rows.values()):raise AssertionError('Shared full day windows differ; do not reuse stale covariates '+json.dumps(rows))
    return {'shared_full_day_windows':len(old),'shared_stays':old.stay_id.nunique(),'all_fields_exactly_equal':True,'field_disagreements':rows}


def join_feature(oldpath,newpath,secure,name,days):
    old=pd.read_csv(oldpath);new=pd.read_csv(newpath)
    if set(old.columns)!=set(new.columns):raise ValueError('Feature schema changed '+name)
    combo=pd.concat([old,new[old.columns]],ignore_index=True)
    assert not combo.duplicated(['stay_id','local_day']).any()
    keys=days[['stay_id','local_day']].merge(combo[['stay_id','local_day']],on=['stay_id','local_day'],how='outer',indicator=True,validate='one_to_one')
    assert keys._merge.eq('both').all(),'Feature windows not exactly covering S1'
    combo.to_csv(secure/name,index=False,compression='gzip')
    return {'old_full_window_rows_reused':len(old),'new_window_rows_extracted':len(new),'S1_window_rows':len(combo)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--prior-stage',type=Path,required=True)
    p.add_argument('--baseline-code',type=Path,required=True);p.add_argument('--audit-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args();raw=args.raw_root.resolve();prior=args.prior_stage.resolve();audit=args.audit_root.resolve();root=args.output_root.resolve()
    if root.is_relative_to(raw) or root.is_relative_to(prior):raise ValueError('Output must be new isolated stage')
    secure=root/'secure_derived_inputs';agg=root/'aggregate';code=root/'code/portable_baseline'
    if (agg/'S1_primary_input_checkpoint.json').exists():raise RuntimeError('Refusing to overwrite a completed stage')
    for d in [secure,agg,root/'logs']:d.mkdir(parents=True,exist_ok=True)
    if not code.exists():shutil.copytree(args.baseline_code,code,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    sys.path.insert(0,str(code))
    build=module('S1_original_build',code/'build_main_from_raw.py')
    from scripts import run_host_vulnerability_day3_landmark as legacy
    from scripts import rebuild_locked_mimic_package_v4 as locked
    from src.analysis.host_vulnerability_indication import CHRONIC_CONDITIONS,prepare_host_indication_person_days,derive_chronic_vulnerability
    started=datetime.now().astimezone().isoformat();prsecure=prior/'secure_derived_inputs';ausec=audit/'secure_derived_inputs'
    cohort=pd.read_csv(ausec/'S1_cohort.csv.gz',parse_dates=['landmark_time','index_interval_start','index_interval_end','intime','outtime','dischtime','deathtime','followup_end'])
    oldcohort=pd.read_csv(prsecure/'landmark_cohort_day3.csv.gz')
    common=set(oldcohort.stay_id);newids=set(cohort.stay_id)-common
    assert common<=set(cohort.stay_id),'S1 cannot remove original S0 eligible stays under this tightening'
    cvc=pd.read_csv(prsecure/'merged_cvc_episodes_day3.csv.gz',parse_dates=['cvc_starttime','cvc_endtime','intime','outtime','dischtime','deathtime'])
    valid=pd.read_csv(prsecure/'valid_cvc_records_day3.csv.gz',parse_dates=['starttime','endtime'])
    days=legacy.build_landmark_days('day3_calendar_landmark',cohort,cvc,valid)
    olddays=pd.read_csv(prsecure/'day_windows_day3.csv.gz',parse_dates=['day_start','day_end','landmark_time','followup_end'])
    windowcheck=verify_windows(olddays,days.loc[days.stay_id.isin(common)])
    (agg/'S1_shared_full_window_reconciliation.json').write_text(json.dumps(windowcheck,indent=2),encoding='utf-8')
    print(f'All common full day-window keys/fields exactly match; rebuilding complete S1 risk inputs',flush=True)
    build.secure_write(days,secure/'day_windows_day3.csv.gz',secure)
    newdays=days.loc[days.stay_id.isin(newids)].copy()
    addroot=root/'newly_eligible_only';addsec=addroot/'secure_derived_inputs';addagg=addroot/'aggregate'
    addsec.mkdir(parents=True,exist_ok=True);addagg.mkdir(parents=True,exist_ok=True)
    if newdays.empty:raise RuntimeError('This implementation expects the actually observed nonempty newly eligible set')
    labqc=build.laboratory(newdays,raw,addsec,addagg)
    treatmentqc=build.treatments(newdays,raw,addsec,addagg)
    antibioticqc=build.antibiotics(newdays,raw,addsec)
    featureqc={}
    for name in ['corrected_features_day3.csv.gz','corrected_treatments_day3.csv.gz','antibiotic_exposure_day3.csv.gz']:
        featureqc[name]=join_feature(prsecure/name,addsec/name,secure,name,days)
    # Read diagnoses once for new stays, derive legacy flags and the same frozen
    # version-aware corrected cancer flags; copy verified old deltas unchanged.
    newhadms=set(cohort.loc[cohort.stay_id.isin(newids),'hadm_id']);pieces=[];n_dx=0
    for ch in pd.read_csv(raw/'hosp/diagnoses_icd.csv.gz',usecols=['subject_id','hadm_id','icd_code','icd_version'],chunksize=500000):
        n_dx+=len(ch);z=ch.loc[ch.hadm_id.isin(newhadms)].copy()
        if len(z):pieces.append(z)
    dx=pd.concat(pieces,ignore_index=True);build.secure_write(dx,addsec/'diagnoses_newly_eligible.csv.gz',addsec)
    newchronic=derive_chronic_vulnerability(dx);oldchronic=pd.read_csv(prsecure/'legacy_chronic_flags_day3.csv.gz')
    chronic=pd.concat([oldchronic,newchronic[oldchronic.columns]],ignore_index=True)
    assert not chronic.duplicated(['subject_id','hadm_id']).any()
    build.secure_write(chronic,secure/'legacy_chronic_flags_day3.csv.gz',secure)
    build.CODING.RAW_DIAGNOSES=addsec/'diagnoses_newly_eligible.csv.gz'
    newflags=cohort.loc[cohort.stay_id.isin(newids),['subject_id','hadm_id','stay_id']].merge(newchronic[['subject_id','hadm_id','cancer','metastatic_cancer']],on=['subject_id','hadm_id'],how='left',validate='one_to_one')
    newflags[['cancer','metastatic_cancer']]=newflags[['cancer','metastatic_cancer']].fillna(0).astype(int)
    delta_new=build.CODING.stay_flag_deltas(newflags)
    delta_old=pd.read_csv(prsecure/'coding_delta_day3.csv.gz');delta=pd.concat([delta_old,delta_new[delta_old.columns]],ignore_index=True)
    assert delta.stay_id.is_unique and set(delta.stay_id)==set(cohort.stay_id)
    build.secure_write(delta,secure/'coding_delta_day3.csv.gz',secure)
    features=pd.read_csv(secure/'corrected_features_day3.csv.gz');treatment=pd.read_csv(secure/'corrected_treatments_day3.csv.gz');antibiotic=pd.read_csv(secure/'antibiotic_exposure_day3.csv.gz')
    state=days.merge(features,on=['stay_id','local_day'],validate='one_to_one',suffixes=('','_revised'))
    if 'age_revised' in state:state['age']=state.age_revised;state=state.drop(columns='age_revised')
    state=state.merge(treatment,on=['stay_id','local_day'],validate='one_to_one').merge(antibiotic,on=['stay_id','local_day'],validate='one_to_one')
    state['high_frequency_iv_meds']=0;state['recent_positive_culture']=0;state['sofa_proxy']=0
    state['cvc_type_group']=build.cvc_type(state,valid);state=prepare_host_indication_person_days(state,chronic)
    firstall=pd.read_csv(ausec/'S1_first_all_bsi.csv.gz',parse_dates=['event_time']);first=pd.read_csv(ausec/'S1_first_line_timed_bsi.csv.gz',parse_dates=['event_time'])
    state,outcomeqc=legacy.add_outcomes(state,firstall,first)
    for name,f in [('landmark_cohort_day3',cohort),('patient_day_states_day3',state),('first_all_bsi_day3',firstall),('clabsi_events_day3',first)]:build.secure_write(f,secure/(name+'.csv.gz'),secure)
    outcomeqc.to_csv(agg/'outcome_assignment_qc.csv',index=False)
    build.REANALYSIS.ORIGINAL=secure;build.REANALYSIS.SEC=secure;build.REANALYSIS.OUT=agg
    d,lm,ev=build.REANALYSIS.load('day3');risk,end,zero=build.REANALYSIS.build_risk(d,lm,ev)
    build.secure_write(risk,secure/'risk_before_icd_correction_day3.csv.gz',secure)
    risk=build.CODING.correct_frame(risk,delta)
    for target,acute in [('chronic_vulnerability_score',None),('vulnerability_burden_score','acute_vulnerability_score'),('vulnerability_burden_score_full_sensitivity','acute_vulnerability_score_full_sensitivity')]:
        if target in risk:risk[target]=risk[CHRONIC_CONDITIONS].sum(axis=1).astype(int)+(0 if acute is None else risk[acute].astype(int))
    assert risk.corrected_chronic_score.eq(risk[CHRONIC_CONDITIONS].sum(axis=1)).all()
    build.secure_write(risk,secure/'risk_final_day3.csv.gz',secure);summary=build.SAMPLING.validate_risk(risk)
    blood=pd.read_csv(ausec/'candidate_blood_results_raw.csv.gz',parse_dates=['charttime','event_time'])
    positive=blood.loc[~blood.invalid_org & blood.org_norm.notna() & blood.org_norm.ne('')].copy()
    sourceqc=module('S1_original_extract',code/'extract_day3_from_raw.py').save_collection_master(blood,positive,cohort,secure)
    collections=pd.read_csv(secure/'culture_collections_raw_day3.csv.gz',parse_dates=['event_time','charttime'])
    assigned=build.SAMPLING.assign_collections(risk,collections.loc[collections.has_non_cancelled.eq(1)].copy())
    sampling,occasions=build.SAMPLING.add_outcomes(risk,assigned);build.secure_write(sampling,secure/'sampling_intervals_final_day3.csv.gz',secure)
    # Newly reconstructed risk endpoints must exactly agree with independent S1
    # terminal selection; only risk-day summation permits floating-point tolerance.
    terminals=pd.read_csv(ausec/'S1_terminal_times.csv.gz',parse_dates=['observed_end'])
    re=risk.groupby('stay_id',as_index=False).agg(observed_end=('stop_datetime','max'),event_eligible=('event','max'),risk_person_days=('duration_days','sum'))
    m=re.merge(terminals[['stay_id','observed_end','event_eligible','risk_person_days']],on='stay_id',validate='one_to_one',suffixes=('_risk','_audit'))
    assert m.observed_end_risk.eq(m.observed_end_audit).all() and m.event_eligible_risk.eq(m.event_eligible_audit).all()
    assert np.allclose(m.risk_person_days_risk,m.risk_person_days_audit,rtol=0,atol=1e-12)
    checkpoint={'status':'executed','definition':'S1 sensitivity; original S0 remains primary','started_at':started,'completed_at':datetime.now().astimezone().isoformat(),
        'window_reconciliation':windowcheck,'feature_reuse':featureqc,'new_stays':len(newids),'new_day_windows':len(newdays),
        'new_stay_raw_extraction':{'laboratory':labqc,'treatment':treatmentqc,'antibiotic':antibioticqc,'diagnosis_source_rows_scanned':n_dx},
        'risk_summary':summary,'collections':sourceqc,'noncancelled_specimens_in_risk':len(assigned),'sampling_occasions_in_risk':len(occasions),
        'zero_time_events_adjusted_to_one_second':zero,'full_rebuilt_terminal_inputs_match_S1_audit':True,
        'source_hashes':{str(f):sha(f) for f in [Path(__file__),prsecure/'merged_cvc_episodes_day3.csv.gz',prsecure/'valid_cvc_records_day3.csv.gz',prsecure/'day_windows_day3.csv.gz',prsecure/'corrected_features_day3.csv.gz',prsecure/'corrected_treatments_day3.csv.gz',prsecure/'antibiotic_exposure_day3.csv.gz',ausec/'S1_cohort.csv.gz',ausec/'S1_first_line_timed_bsi.csv.gz',ausec/'candidate_blood_results_raw.csv.gz']}}
    (agg/'S1_primary_input_checkpoint.json').write_text(json.dumps(checkpoint,indent=2,default=int),encoding='utf-8')
    print(json.dumps({'status':'executed','definition':'S1','risk_summary':summary,'noncancelled_specimens_in_risk':len(assigned),'sampling_occasions_in_risk':len(occasions)}),flush=True)


if __name__=='__main__':main()

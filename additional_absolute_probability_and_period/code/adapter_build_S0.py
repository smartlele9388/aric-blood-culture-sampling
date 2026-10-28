"""Read-only S0 stage adapter. Patient rows are written only to secure_derived_inputs."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, platform, shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def extract_functions(source, names, dest):
    """Freeze exact source functions, avoiding legacy imports with output side effects."""
    text=source.read_text(encoding='utf-8-sig'); tree=ast.parse(text)
    parts=[ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(parts)!=len(names):raise RuntimeError('Missing source functions')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text('"""Exact isolated functions from frozen original source; no entrypoint side effects."""\nimport numpy as np\nimport pandas as pd\n\n'+'\n\n'.join(parts)+'\n',encoding='utf-8')
    spec=importlib.util.spec_from_file_location(dest.stem,dest); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def main():
    a=argparse.ArgumentParser();a.add_argument('--stage-root',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True);a.add_argument('--provided-root',type=Path,required=True);args=a.parse_args()
    start=datetime.now().astimezone().isoformat();root=args.output_root;sec=root/'secure_derived_inputs';agg=root/'aggregate';code=root/'code'
    for d in(sec,agg,code,root/'logs'):d.mkdir(parents=True,exist_ok=True)
    src=args.stage_root/'secure_derived_inputs';portable=args.stage_root/'portable_code'
    sampling=portable/'ARIC_revision_20260920/route_A/code/run_sampling_analysis.py'
    mapsrc=portable/'src/analysis/host_vulnerability_advanced.py'
    frozen=extract_functions(sampling,{'validate_risk','assign_collections','collapse_occasions'},code/'source_baseline/frozen_sampling_functions.py')
    mapping=extract_functions(mapsrc,{'collapse_icu_type','collapse_admission_type'},code/'source_baseline/frozen_grouping_functions.py')
    sources={}
    def read(name,dates):
        p=src/name;d=pd.read_csv(p,parse_dates=dates)
        sources[name]={'path':str(p.resolve()),'sha256':sha(p),'rows':len(d),'columns':d.columns.tolist()};return d
    risk=read('risk_final_day3.csv.gz',['day_start','stop_datetime','landmark_time','observed_end','risk_end','index_interval_end','event_time']).reset_index(drop=True)
    landmark=read('landmark_cohort_day3.csv.gz',['landmark_time','index_interval_start','index_interval_end','followup_end','deathtime','outtime','dischtime'])
    master=read('culture_collections_raw_day3.csv.gz',['event_time','charttime'])
    sampling_rows=read('sampling_intervals_final_day3.csv.gz',['day_start'])
    terminal=read('competing_events_day3.csv.gz',['landmark_time','observed_end','risk_end','line_window_end','deathtime','outtime','dischtime','index_interval_end'])
    qc=frozen.validate_risk(risk)
    first=risk.sort_values(['stay_id','day_start']).drop_duplicates('stay_id').set_index('stay_id')
    last=risk.sort_values(['stay_id','day_start']).drop_duplicates('stay_id',keep='last').set_index('stay_id')
    lm=landmark.set_index('stay_id').reindex(first.index);term=terminal.set_index('stay_id').reindex(first.index)
    assert landmark.stay_id.is_unique and landmark.hadm_id.is_unique and terminal.stay_id.is_unique
    assert set(landmark.stay_id)==set(first.index)==set(terminal.stay_id)
    assert np.array_equal(first.subject_id,lm.subject_id) and np.array_equal(first.hadm_id,lm.hadm_id)
    assert np.array_equal(first.subject_id,term.subject_id)
    assert (first.t_start==0).all() and (first.day_start==lm.landmark_time).all()
    assert (last.stop_datetime==term.observed_end).all()
    assert (first.landmark_time==lm.index_interval_start.dt.floor('D')+pd.Timedelta(days=2)).all()
    assert (first.icu_group==lm.icu_type.map(mapping.collapse_icu_type)).all()
    assert (first.admission_group==lm.admission_type.map(mapping.collapse_admission_type)).all()
    assert np.allclose(first.age_per_10y,(first.age-60)/10,rtol=0,atol=1e-12)
    assert (first.female==lm.female).all()
    # Reconstruct exits separately from the terminal-stage classifications.
    lineend=(lm.index_interval_end-pd.Timedelta(nanoseconds=1)).dt.floor('D')+pd.Timedelta(days=2)
    riskend=pd.concat([lineend,lm.followup_end],axis=1).min(axis=1)
    eligible=first.event_time.notna() & first.event_time.ge(lm.landmark_time) & first.event_time.le(riskend) & first.event_time.lt(lineend)
    end=riskend.where(~eligible,first.event_time)
    cause=pd.Series(0,index=first.index)
    cause.loc[riskend.eq(lineend)]=4;cause.loc[riskend.eq(lm.outtime)|riskend.eq(lm.dischtime)]=3;cause.loc[riskend.eq(lm.deathtime)]=2;cause.loc[eligible]=1
    assert np.array_equal(cause,term.cause) and (end==term.observed_end).all()
    assert np.array_equal(eligible,first.event_eligible)
    assert (lineend==term.line_window_end).all() and (riskend==term.risk_end).all()
    assert not (eligible & first.event_time.eq(lm.landmark_time)).any()
    durations=(end-lm.landmark_time).dt.total_seconds()/86400
    assert np.allclose(durations,term.time_days,atol=1e-12,rtol=0)
    # Frozen allocator, with original row positions; no reliance on a previous allocation.
    assigned=frozen.assign_collections(risk,master.loc[master.has_non_cancelled.eq(1)].copy())
    assert assigned.charttime.notna().all() and (assigned.event_time==assigned.charttime).all()
    assigned['stay_id']=risk.stay_id.to_numpy()[assigned.risk_row.to_numpy()]
    occ=assigned.groupby(['stay_id','event_time'],as_index=False).agg(specimens=('collection_key','size'))
    sam=sampling_rows.set_index(['stay_id','day_start']).reindex(pd.MultiIndex.from_frame(risk[['stay_id','day_start']]))
    assert len(sam)==len(risk) and not sam.collection_count.isna().any()
    got_spec=assigned.groupby('risk_row').size().reindex(risk.index,fill_value=0)
    got_occ=frozen.collapse_occasions(assigned).groupby('risk_row').size().reindex(risk.index,fill_value=0)
    assert np.array_equal(got_spec,sam.collection_count) and np.array_equal(got_occ,sam.occasion_count)
    first_time=occ.groupby('stay_id').event_time.min().reindex(first.index)
    score=first.exposure_vulnerability_burden_score
    assert np.isfinite(score).all() and np.allclose(score,np.rint(score))
    implied=first.corrected_chronic_score+first.lag_mechanical_ventilation+first.lag_aki.eq(1)+first.lag_lactate_high.eq(1)
    assert np.array_equal(implied,score)
    group=np.where(score<=1,'low',np.where(score<=3,'moderate','high'))
    assert np.array_equal(group,term.baseline_group) and np.array_equal(group,first.exposure_vulnerability_category)
    frame=first[['subject_id','hadm_id','age','age_per_10y','female','icu_group','admission_group']].copy()
    frame['baseline_score']=score;frame['baseline_group']=group;frame['bsi_time_days']=durations;frame['bsi_status']=cause
    sampled=first_time.notna();frame['sampling_time_days']=((first_time-lm.landmark_time).dt.total_seconds()/86400).where(sampled,durations)
    frame['sampling_status']=cause.where(~sampled,1)
    frame['specimens']=assigned.groupby('stay_id').size().reindex(first.index,fill_value=0)
    frame['occasions']=occ.groupby('stay_id').size().reindex(first.index,fill_value=0)
    frame['risk_intervals']=risk.groupby('stay_id').size().reindex(first.index)
    frame['risk_duration_days']=risk.groupby('stay_id').duration_days.sum().reindex(first.index)
    assert frame[['age','age_per_10y','female','icu_group','admission_group']].notna().all().all()
    frame=frame.reset_index();out=sec/'cohort_standardization_S0.csv';frame.to_csv(out,index=False,float_format='%.17g')
    shutil.copy2(args.provided_root/'validate_cohort.py',code/'validate_cohort.py')
    shutil.copy2(args.provided_root/'test_validate_cohort.py',code/'test_validate_cohort.py')
    audit={'status':'completed','stage_counts':qc,'full_interval_specimen_count_mismatches':0,'full_interval_occasion_count_mismatches':0,'terminal_cause_or_time_mismatches':0,'source_model_group_mismatches':0,'source_score_composition_mismatches':0,'assigned_specimens':len(assigned),'exact_time_occasions':len(occ),'assigned_specimens_missing_exact_charttime':0,'assigned_event_time_charttime_mismatches':0,'time_zero_BSI_adjustments':0,'first_sampling_at_landmark':int((sampled&frame.set_index('stay_id').sampling_time_days.eq(0)).sum()),'first_sampling_at_terminal_BSI':int((first_time.eq(end)&cause.eq(1)).sum()),'first_sampling_at_other_terminal_exit':int((first_time.eq(end)&cause.ne(1)).sum()),'first_sample_after_exit':int(first_time.gt(end).sum()),'BSI_without_sample':int((cause.eq(1)&~sampled).sum()),'administrative_censoring_before_28_days':int((cause.eq(0)&durations.lt(28)).sum()),'sampling_rule':'Exact time zero retained. Valid first sampling overrides tied BSI or competing exit. Internal interval boundaries assign forward; final end included.','BSI_clock':'Collection time, retrospectively confirmed where required; not result-availability time.','exit_priority':'BSI > death > ICU/hospital discharge > line-attribution end > day28 administrative','grouping':'First final risk interval score, 0-1/2-3/>=4. Retrospective discharge-code burden plus already lagged acute markers.','demographic_source':'ICU first_careunit and hospital admission_type inherited from original landmark; no future transfers. Age=(age-60)/10.','completed_at':datetime.now().astimezone().isoformat()}
    (agg/'timing_and_tie_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    for p in [sampling,mapsrc,portable/'derive_competing_exits.py',portable/'ARIC_revision_20260920/code/run_reanalysis.py',portable/'scripts/run_host_vulnerability_day3_landmark.py',Path(__file__)]:
        sources[str(p.name)]={'path':str(p.resolve()),'sha256':sha(p)}
    (root/'input_provenance_local.json').write_text(json.dumps({'started_at':start,'completed_at':audit['completed_at'],'input_stage':'S0 raw-reproduced 2026-10-11','sources':sources,'adapter_output':{'path':str(out.resolve()),'sha256':sha(out)},'environment':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__}},indent=2),encoding='utf-8')
    public={k:{kk:vv for kk,vv in v.items() if kk!='path'} for k,v in sources.items()}
    (agg/'adapter_input_provenance_SHAREABLE.json').write_text(json.dumps({'sources':public,'adapter_output_sha256':sha(out),'patient_rows_exported':False},indent=2),encoding='utf-8')
    print(json.dumps({'status':'completed','stays':len(frame),'events':int(frame.bsi_status.eq(1).sum()),'specimens':int(frame.specimens.sum()),'occasions':int(frame.occasions.sum()),'output':'secure_derived_inputs/cohort_standardization_S0.csv'}),flush=True)

if __name__=='__main__':main()

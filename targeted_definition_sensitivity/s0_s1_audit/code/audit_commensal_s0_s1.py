"""Audit unchanged S0 and timestamp/specimen-distinct S1 over all CVC candidates.

Patient rows remain under secure_derived_inputs. No fit is run here: this stage
determines whether downstream cohort/risk rebuilding and fitting are triggered.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def write_secure(frame, secure, name):
    path = secure / (name + '.csv.gz')
    assert path.resolve().is_relative_to(secure.resolve())
    frame.to_csv(path, index=False, compression='gzip')


def read_blood(raw, admissions, locked):
    path = raw / 'hosp/microbiologyevents.csv.gz'
    header = list(pd.read_csv(path, nrows=0).columns)
    if 'micro_specimen_id' not in header or 'charttime' not in header:
        raise ValueError('S1 requires actual micro_specimen_id and charttime source columns')
    cols = ['subject_id','hadm_id','chartdate','charttime','spec_type_desc','org_name','ab_name','interpretation','micro_specimen_id']
    if 'microevent_id' in header: cols.append('microevent_id')
    keep = []; n = 0; retained = 0
    for i, chunk in enumerate(pd.read_csv(path, usecols=cols, chunksize=500000, low_memory=False), 1):
        n += len(chunk)
        chunk = chunk.loc[chunk.hadm_id.isin(admissions) & chunk.spec_type_desc.fillna('').astype(str).str.contains('blood', case=False)].copy()
        if len(chunk): keep.append(chunk); retained += len(chunk)
        if i % 5 == 0: print(f'Microbiology rows scanned {n:,}; CVC-admission blood rows retained {retained:,}', flush=True)
    blood = pd.concat(keep, ignore_index=True)
    for col in ['chartdate','charttime']: blood[col] = pd.to_datetime(blood[col], errors='coerce')
    blood['event_time'] = blood.charttime.fillna(blood.chartdate)
    blood = blood.loc[blood.event_time.notna()].copy()
    blood['org_norm'] = blood.org_name.map(locked.norm)
    blood['invalid_org'] = blood.org_name.map(locked.invalid_organism_label)
    fallback = blood.subject_id.astype(str)+'|'+blood.hadm_id.astype(str)+'|'+blood.event_time.astype(str)+'|'+blood.spec_type_desc.fillna('').astype(str)
    blood['collection_key'] = np.where(blood.micro_specimen_id.notna(), 'specimen:'+blood.micro_specimen_id.astype('Int64').astype(str), 'fallback:'+fallback)
    blood['isolate_key'] = blood.collection_key+'|'+blood.org_norm
    positive = blood.loc[~blood.invalid_org & blood.org_norm.ne('')].copy()
    susc = positive.loc[positive.interpretation.fillna('').astype(str).str.strip().ne('')].copy()
    return blood, positive, susc, {'source_rows_scanned': n, 'candidate_blood_rows_retained': len(blood), 'raw_header': header}


def flags(isolates):
    return isolates.groupby(['subject_id','hadm_id','stay_id','event_time','event_day'], as_index=False).agg(
        recognized_pathogen=('is_commensal', lambda x: int((~x.astype(bool)).any())),
        confirmed_common_commensal=('is_commensal',lambda x:int(x.astype(bool).any())),
        pathogen_count=('org_norm','nunique')).sort_values(['stay_id','event_time'])


def build_s1(positive, context, locked):
    linked = positive.merge(context[['subject_id','hadm_id','stay_id','admittime']].drop_duplicates('stay_id'), on=['subject_id','hadm_id'], how='inner')
    linked = linked.loc[linked.event_time >= linked.admittime].copy()
    linked['event_day'] = linked.event_time.dt.floor('D')
    linked['is_commensal'] = linked.org_name.fillna('').str.contains(locked.COMMON_COMMENSAL_RE, na=False)
    basecols = ['subject_id','hadm_id','stay_id','collection_key','isolate_key','event_time','event_day','org_norm','org_name','is_commensal']
    units = linked[basecols+['micro_specimen_id','charttime']].drop_duplicates()
    # Actual collection keys must map unambiguously to a source specimen/time.
    meta = units.groupby('collection_key').agg(n_id=('micro_specimen_id','nunique'), n_exact_time=('charttime','nunique'), n_event_time=('event_time','nunique'))
    ambiguous = meta.n_id.gt(1) | meta.n_exact_time.gt(1) | (meta.n_exact_time.eq(0)&meta.n_event_time.gt(1))
    if ambiguous.any(): raise ValueError('Ambiguous positive collection metadata; do not guess a confirmation timestamp')
    recognized = units.loc[~units.is_commensal, basecols].copy()
    comm = units.loc[units.is_commensal].copy()
    pairs = []; membership = []; s0_rows = []; s1_rows = []
    for keys, group in comm.groupby(['subject_id','hadm_id','stay_id','org_norm'],sort=False):
        byday = {day: g.drop_duplicates('collection_key').sort_values('collection_key') for day,g in group.groupby('event_day',sort=True)}
        valid0 = set(); valid1 = set()
        for day,g in byday.items():
            nextday = day+pd.Timedelta(days=1)
            pool = pd.concat([g,byday.get(nextday,g.iloc[:0])]).drop_duplicates('collection_key')
            good0 = len(pool)>=2; good1 = False
            for a,b in itertools.combinations(pool.itertuples(index=False),2):
                distinct_id = pd.notna(a.micro_specimen_id) and pd.notna(b.micro_specimen_id) and a.micro_specimen_id != b.micro_specimen_id
                exact_known = pd.notna(a.charttime) and pd.notna(b.charttime)
                distinct_time = exact_known and a.charttime != b.charttime
                accepted = bool(distinct_id and distinct_time)
                good1 |= accepted
                pairs.append(dict(zip(['subject_id','hadm_id','stay_id','org_norm'],keys)) | {
                    'anchor_day':day,'collection_key_a':a.collection_key,'collection_key_b':b.collection_key,
                    'micro_specimen_id_a':a.micro_specimen_id,'micro_specimen_id_b':b.micro_specimen_id,
                    'charttime_a':a.charttime,'charttime_b':b.charttime,'S0_valid_pair':True,'S1_valid_pair':accepted,
                    'distinct_actual_specimen_id':bool(distinct_id),'both_exact_times_present':bool(exact_known),'distinct_exact_time':bool(distinct_time)})
            for is_valid, valid in [(good0,valid0),(good1,valid1)]:
                if is_valid:
                    valid.add(day)
                    if nextday in byday: valid.add(nextday)
        if valid0: s0_rows.append(group.loc[group.event_day.isin(valid0),basecols])
        if valid1: s1_rows.append(group.loc[group.event_day.isin(valid1),basecols])
        for row in group.itertuples(index=False):
            membership.append({c:getattr(row,c) for c in basecols+['micro_specimen_id','charttime']} | {'S0_valid_day':row.event_day in valid0,'S1_valid_day':row.event_day in valid1})
    confirmed0 = pd.concat(s0_rows,ignore_index=True) if s0_rows else comm.iloc[:0][basecols]
    confirmed1 = pd.concat(s1_rows,ignore_index=True) if s1_rows else comm.iloc[:0][basecols]
    iso0 = pd.concat([recognized,confirmed0],ignore_index=True).drop_duplicates()
    iso1 = pd.concat([recognized,confirmed1],ignore_index=True).drop_duplicates()
    return flags(iso0),iso0,flags(iso1),iso1,pd.DataFrame(pairs),pd.DataFrame(membership),meta


def compare_frames(a,b,keys,cols):
    x = a[keys+cols].copy(); y=b[keys+cols].copy()
    if x.duplicated(keys).any() or y.duplicated(keys).any(): raise ValueError('Comparison key not unique')
    m=x.merge(y,on=keys,how='outer',suffixes=('_S0','_S1'),indicator=True)
    shared=m._merge.eq('both')
    different = pd.Series(False,index=m.index)
    field = {}
    for c in cols:
        eq=m[c+'_S0'].eq(m[c+'_S1']) | (m[c+'_S0'].isna()&m[c+'_S1'].isna())
        d=shared&~eq; different|=d; field[c]=int(d.sum())
    diff=m.loc[~shared|different].copy()
    return {'S0_rows':len(a),'S1_rows':len(b),'S0_only_rows':int(m._merge.eq('left_only').sum()),'S1_only_rows':int(m._merge.eq('right_only').sum()),'shared_changed_rows':int(different.sum()),'field_disagreements':field,'all_equal':len(diff)==0},diff


def terminal_times(cohort,events):
    # Same main-analysis attribution and tie/zero-time handling; not a refit.
    lm = cohort.copy()
    lm['line_window_end']=(lm.index_interval_end-pd.Timedelta(nanoseconds=1)).dt.floor('D')+pd.Timedelta(days=2)
    lm['risk_end']=pd.concat([lm.line_window_end,lm.followup_end],axis=1).min(axis=1)
    lm=lm.merge(events[['stay_id','event_time','recognized_pathogen','amr_bsi_r','amr_bsi_ri']],on='stay_id',how='left',validate='one_to_one')
    lm['event_eligible']=lm.event_time.notna()&(lm.event_time>=lm.landmark_time)&(lm.event_time<=lm.risk_end)&(lm.event_time<lm.line_window_end)
    lm['observed_end']=lm.risk_end.where(~lm.event_eligible,lm.event_time)
    zero=lm.event_eligible&lm.observed_end.eq(lm.landmark_time)
    lm.loc[zero,'observed_end']=lm.loc[zero,'landmark_time']+pd.Timedelta(seconds=1)
    lm['risk_person_days']=(lm.observed_end-lm.landmark_time).dt.total_seconds()/86400
    return lm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--stage-root',type=Path,required=True)
    p.add_argument('--baseline-code',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args(); raw=args.raw_root.resolve(); stage=args.stage_root.resolve(); base=args.baseline_code.resolve(); out=args.output_root.resolve()
    if out.is_relative_to(raw) or out.is_relative_to(stage): raise ValueError('Fresh output must be outside source data/stage')
    secure=out/'secure_derived_inputs'; aggregate=out/'aggregate'; secure.mkdir(parents=True,exist_ok=True);aggregate.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(base))
    from scripts import rebuild_locked_mimic_package_v4 as locked
    from scripts import run_host_vulnerability_day3_landmark as legacy
    # Prevent any imported helper QC from writing into a historical source tree.
    paths=locked.Paths(base=raw,qc=secure/'helper_qc',tables=secure/'helper_tables',figures=secure/'helper_figures',manuscript=secure/'helper_manuscript')
    for d in [paths.qc,paths.tables,paths.figures,paths.manuscript]: d.mkdir(parents=True,exist_ok=True)
    started=datetime.now().astimezone().isoformat()
    patients,admissions,icu,_=locked.load_core(paths)
    adult_all,adult_first=locked.adult_cohorts(patients,admissions,icu)
    cvc_path=stage/'secure_derived_inputs/merged_cvc_episodes_day3.csv.gz'
    cvc=pd.read_csv(cvc_path,parse_dates=['cvc_starttime','cvc_endtime','intime','outtime','dischtime','deathtime'])
    context=adult_first.loc[adult_first.stay_id.isin(set(cvc.stay_id))].copy()
    assert len(context)==cvc.stay_id.nunique()
    blood,positive,susc,sourceqc=read_blood(raw,set(context.hadm_id),locked)
    write_secure(blood,secure,'candidate_blood_results_raw')
    event0,iso0=legacy.qualify_all_bsi_events(positive,context)
    audit0,auditiso0,event1,iso1,pairs,membership,meta=build_s1(positive,context,locked)
    k=['stay_id','event_time']; cols=['recognized_pathogen','confirmed_common_commensal','pathogen_count']
    auditcheck,_=compare_frames(event0,audit0,k,cols)
    if not auditcheck['all_equal']: raise AssertionError('Independent day-membership S0 must exactly match legacy original S0')
    isocols=['subject_id','hadm_id','stay_id','collection_key','isolate_key','event_time','event_day','org_norm','org_name','is_commensal']
    iso_check,_=compare_frames(iso0,auditiso0,isocols,[])
    assert iso_check['all_equal']
    for n,f in [('S0_all_qualifying_event_times',event0),('S1_all_qualifying_event_times',event1),('S0_qualifying_isolates',iso0),('S1_qualifying_isolates',iso1),('commensal_pairs',pairs),('commensal_day_membership',membership)]:write_secure(f,secure,n)
    lm,lm_qc=legacy.make_landmarks(cvc,'day3_calendar_landmark')
    write_secure(lm,secure,'landmark_candidates')
    outputs={}; qcs=[]
    for label,event,iso in [('S0',event0,iso0),('S1',event1,iso1)]:
        cohort,allfirst,first,eventqc=legacy.select_scenario_events('day3_calendar_landmark',lm,cvc,event,iso,susc,paths)
        terminal=terminal_times(cohort,first)
        selected_iso=first.attrs.get('bsi_isolates',iso.iloc[:0])
        finals=terminal.loc[terminal.event_eligible,['stay_id','event_time']]
        final_iso=selected_iso.merge(finals,on=['stay_id','event_time'],how='inner')
        for n,f in [('cohort',cohort),('first_all_bsi',allfirst),('first_line_timed_bsi',first),('terminal_times',terminal),('final_event_isolates',final_iso)]:write_secure(f,secure,label+'_'+n)
        eventqc['definition']=label; qcs.append(eventqc)
        outputs[label]={'cohort':cohort,'first':first,'terminal':terminal,'iso':final_iso}
    comparison={}; localdiff=[]
    for label,a,b,keys,columns in [
        ('all_qualifying_event_times',event0,event1,k,cols),
        ('landmark_cohort',outputs['S0']['cohort'],outputs['S1']['cohort'],['stay_id'],['subject_id','hadm_id','landmark_time','index_interval_start','index_interval_end','followup_end']),
        ('first_line_timed_bsi',outputs['S0']['first'],outputs['S1']['first'],['stay_id'],['event_time','recognized_pathogen','confirmed_common_commensal','pathogen_count','org_name','amr_bsi_r','amr_bsi_ri']),
        ('final_terminal_times',outputs['S0']['terminal'],outputs['S1']['terminal'],['stay_id'],['landmark_time','risk_end','line_window_end','observed_end','event_eligible','risk_person_days']),
        ('final_event_isolates',outputs['S0']['iso'],outputs['S1']['iso'],isocols,[])]:
        comp,diff=compare_frames(a,b,keys,columns);comparison[label]=comp;write_secure(diff,secure,label+'_differences')
    priorcohort=pd.read_csv(stage/'secure_derived_inputs/landmark_cohort_day3.csv.gz',parse_dates=['landmark_time','index_interval_start','index_interval_end','followup_end'])
    priorfirst=pd.read_csv(stage/'secure_derived_inputs/clabsi_events_day3.csv.gz',parse_dates=['event_time'])
    comparison['S0_prior_cohort'],_=compare_frames(priorcohort,outputs['S0']['cohort'],['stay_id'],['subject_id','hadm_id','landmark_time','index_interval_start','index_interval_end','followup_end'])
    comparison['S0_prior_first_event'],_=compare_frames(priorfirst,outputs['S0']['first'],['stay_id'],['event_time','recognized_pathogen','confirmed_common_commensal','pathogen_count','org_name','amr_bsi_r','amr_bsi_ri'])
    priorrisk=pd.read_csv(stage/'secure_derived_inputs/risk_final_day3.csv.gz',parse_dates=['day_start','stop_datetime'])
    endprior=priorrisk.groupby('stay_id',as_index=False).agg(observed_end=('stop_datetime','max'),event_eligible=('event','max'),risk_person_days=('duration_days','sum'))
    comparison['S0_prior_risk_terminal'],_=compare_frames(endprior,outputs['S0']['terminal'],['stay_id'],['observed_end','event_eligible','risk_person_days'])
    pre0=event0.merge(lm[['stay_id','landmark_time']],on='stay_id');pre0=pre0.loc[pre0.event_time<pre0.landmark_time]
    pre1=event1.merge(lm[['stay_id','landmark_time']],on='stay_id');pre1=pre1.loc[pre1.event_time<pre1.landmark_time]
    qcs=pd.concat(qcs,ignore_index=True);qcs.to_csv(aggregate/'S0_S1_scenario_counts_local.csv',index=False)
    rows=[]
    for label,event,iso in [('S0',event0,iso0),('S1',event1,iso1)]:
        t=outputs[label]['terminal'];fi=outputs[label]['iso']
        metric={'CVC_candidate_stays':len(context),'landmark_candidates':len(lm),'qualifying_event_times_all_candidate_admissions':len(event),
            'mixed_qualifying_event_times_all_candidates':int((event.recognized_pathogen.eq(1)&event.confirmed_common_commensal.eq(1)).sum()),
            'prelandmark_qualifying_event_times':len(pre0 if label=='S0' else pre1),'prelandmark_excluded_stays':len(set((pre0 if label=='S0' else pre1).stay_id)),
            'final_cohort_stays':len(outputs[label]['cohort']),'final_cohort_patients':outputs[label]['cohort'].subject_id.nunique(),
            'first_line_timed_candidates':len(outputs[label]['first']),'final_first_BSI_events':int(t.event_eligible.sum()),'risk_person_days':float(t.risk_person_days.sum()),
            'commensal_final_event_times':fi.loc[fi.is_commensal,['stay_id','event_time']].drop_duplicates().shape[0],
            'mixed_final_event_times':int(outputs[label]['first'].merge(t.loc[t.event_eligible,['stay_id']],on='stay_id').eval('recognized_pathogen == 1 & confirmed_common_commensal == 1').sum())}
        rows.extend({'definition':label,'metric':name,'value':value} for name,value in metric.items())
    summary=pd.DataFrame(rows);summary.to_csv(aggregate/'S0_S1_counts_local.csv',index=False)
    pairqc={'anchor_day_pair_instances':len(pairs),'unique_specimen_pair_organism_stay_instances':len(pairs.drop_duplicates(['stay_id','org_norm','collection_key_a','collection_key_b'])),
        'S0_pair_instances':int(pairs.S0_valid_pair.sum()),'S1_pair_instances':int(pairs.S1_valid_pair.sum()),
        'S0_pair_instances_failing_S1':int((~pairs.S1_valid_pair).sum()),'same_exact_timestamp_pair_instances':int((pairs.both_exact_times_present&~pairs.distinct_exact_time).sum()),
        'pair_instances_missing_exact_time':int((~pairs.both_exact_times_present).sum()),'pair_instances_missing_or_same_actual_specimen_id':int((~pairs.distinct_actual_specimen_id).sum()),
        'candidate_positive_collection_keys_using_fallback':int(meta.index.to_series().str.startswith('fallback:').sum())}
    trigger=not(comparison['landmark_cohort']['all_equal'] and comparison['final_terminal_times']['all_equal'])
    inputs=[raw/'hosp/microbiologyevents.csv.gz',cvc_path,stage/'secure_derived_inputs/risk_final_day3.csv.gz',stage/'secure_derived_inputs/landmark_cohort_day3.csv.gz',stage/'secure_derived_inputs/clabsi_events_day3.csv.gz',base/'scripts/run_host_vulnerability_day3_landmark.py',base/'scripts/rebuild_locked_mimic_package_v4.py',Path(__file__)]
    report={'status':'executed','started_at':started,'completed_at':datetime.now().astimezone().isoformat(),
        'scope':'All source-backed CVC candidate admissions, all qualifying event times, mixed events and every prelandmark exclusion candidate; not restricted to 268 final endpoints',
        'S0':'Original distinct collection-key confirmation within same or consecutive calendar days; both calendar days retain all same-organism units',
        'S1':'All original S0 conditions and calendar-day membership retained; a supporting pair additionally requires two nonmissing different actual micro_specimen_id values and two nonmissing different exact charttime values. No added minimum elapsed-time interval',
        'not_proven':'Different source timestamps do not prove independent venipunctures; S1 is not a clinical gold standard',
        'source_qc':sourceqc,'pair_qc':pairqc,'legacy_S0_independent_reimplementation':auditcheck,'legacy_S0_isolate_reimplementation':iso_check,
        'comparisons':comparison,'downstream_risk_rebuild_triggered':trigger,'model_fit_status':'pending_if_triggered' if trigger else 'not_triggered_identical_cohort_and_terminal_inputs',
        'hashes':{str(f):sha(f) for f in inputs},'software':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__}}
    (aggregate/'S0_S1_audit_local.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    # Avoid printing granular clinical counts to the public conversational log.
    print(json.dumps({'status':'executed','S0_prior_cohort_exact':comparison['S0_prior_cohort']['all_equal'],'S0_prior_first_event_exact':comparison['S0_prior_first_event']['all_equal'],'risk_rebuild_triggered':trigger,'report':'aggregate/S0_S1_audit_local.json'}),flush=True)


if __name__=='__main__':main()

"""Aggregate the existing all-candidate S0/S1 audit; do not refit any S1 model."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--prior-audit',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();src=a.prior_audit/'secure_derived_inputs';out=a.output_root/'aggregate';hashes={}
    def load(name,dates):
        f=src/name;hashes[name]=sha(f);return pd.read_csv(f,parse_dates=dates)
    lm=load('landmark_candidates.csv.gz',['landmark_time']).set_index('stay_id')
    t0=load('S0_terminal_times.csv.gz',['landmark_time','observed_end','event_time']).set_index('stay_id')
    t1=load('S1_terminal_times.csv.gz',['landmark_time','observed_end','event_time']).set_index('stay_id')
    ev0=load('S0_all_qualifying_event_times.csv.gz',['event_time']);ev1=load('S1_all_qualifying_event_times.csv.gz',['event_time'])
    pre=[]
    for ev in(ev0,ev1):
        z=ev.merge(lm[['landmark_time']],left_on='stay_id',right_index=True);pre.append(set(z.loc[z.event_time.lt(z.landmark_time),'stay_id']))
    assert set(t0.index)==set(lm.index)-pre[0] and set(t1.index)==set(lm.index)-pre[1]
    common=t0.index.intersection(t1.index);new=t1.index.difference(t0.index);lost=t0.index.difference(t1.index);x=t0.loc[common];y=t1.loc[common]
    e0=x.event_eligible.astype(bool);e1=y.event_eligible.astype(bool)
    canceled=e0&~e1;postponed=e0&e1&y.event_time.gt(x.event_time);earlier=e0&e1&y.event_time.lt(x.event_time);newcommon=~e0&e1
    assert not earlier.any() and not newcommon.any()
    # Counts partition common-stay first event transitions and exact terminal extensions.
    unchanged=e0&e1&x.event_time.eq(y.event_time)
    assert int(unchanged.sum()+postponed.sum()+canceled.sum())==int(t0.event_eligible.sum())
    added_time=float(t1.loc[new].risk_person_days.sum());common_delta=float((y.risk_person_days-x.risk_person_days).sum());net=float(t1.risk_person_days.sum()-t0.risk_person_days.sum())
    assert abs(added_time+common_delta-net)<1e-9
    data=[]
    def add(metric,s0,s1,change,unit,scope):data.append({'metric':metric,'S0':s0,'S1':s1,'change':change,'unit':unit,'scope':scope})
    add('landmark_candidates',len(lm),len(lm),0,'stays','Before prior-infection exclusion')
    add('prelandmark_BSI_excluded_stays',len(pre[0]),len(pre[1]),len(pre[1])-len(pre[0]),'stays','All landmark candidates')
    add('eligible_stays',len(t0),len(t1),len(new)-len(lost),'stays','Final S0/S1 cohort')
    add('newly_eligible_stays',0,len(new),len(new),'stays','Only because prior-infection exclusion changes')
    add('S0_eligible_lost_under_S1',0,len(lost),len(lost),'stays','Membership removal')
    add('final_first_BSI_events',int(t0.event_eligible.sum()),int(t1.event_eligible.sum()),int(t1.event_eligible.sum()-t0.event_eligible.sum()),'events','Final risk windows')
    add('common_stays_first_BSI_unchanged',int(unchanged.sum()),int(unchanged.sum()),0,'events','Same first event time retained')
    add('common_stays_S0_first_BSI_no_S1_replacement',int(canceled.sum()),0,-int(canceled.sum()),'events','Original event invalid; followup rebuilt through non-BSI exit')
    add('common_stays_first_BSI_postponed',int(postponed.sum()),int(postponed.sum()),0,'events','Later qualifying S1 first event replaces former time')
    add('newly_eligible_stays_S1_BSI',0,int(t1.loc[new].event_eligible.sum()),int(t1.loc[new].event_eligible.sum()),'events','Added admissions followed under S1')
    add('common_stays_terminal_time_changed',0,int(x.observed_end.ne(y.observed_end).sum()),int(x.observed_end.ne(y.observed_end).sum()),'stays','Actual exits independently rebuilt')
    add('risk_person_days',float(t0.risk_person_days.sum()),float(t1.risk_person_days.sum()),net,'days','All final risk windows; tiny roundoff versus summed interval durations')
    add('risk_days_added_from_new_stays',0,added_time,added_time,'days','Newly eligible stays')
    add('risk_days_extension_in_common_stays',0,common_delta,common_delta,'days','Common-stay exit changes')
    pd.DataFrame(data).to_csv(out/'S0_S1_event_changes.csv',index=False,float_format='%.17g')
    pairs=load('commensal_pairs.csv.gz',[]).drop_duplicates(['stay_id','org_norm','collection_key_a','collection_key_b'])
    rejected=pairs[~pairs.S1_valid_pair];assert (rejected.both_exact_times_present&~rejected.distinct_exact_time).all()
    counts_path=a.prior_audit/'shareable/S0_S1_aggregate_counts.csv';counts=pd.read_csv(counts_path);hashes[counts_path.name]=sha(counts_path)
    candidate_n=int(float(counts.loc[counts.definition.eq('S0')&counts.metric.eq('CVC_candidate_stays'),'value'].iloc[0]))
    summary={'status':'completed_from_existing_audit_patient_tables','new_S1_models_fitted':False,'input_sha256':hashes,'all_prior_exclusion_changes_explain_new_eligibility':bool(set(new)==pre[0]-pre[1]),'candidate_population_for_commensal_pairs':candidate_n,'unique_S0_pair_count':len(pairs),'unique_S1_valid_pair_count':int(pairs.S1_valid_pair.sum()),'unique_same_exact_time_pair_rejections':len(rejected),'pair_counts_source':'Previously executed all-CVC-candidate audit; not denominator268finalBSI','S1_definition':'Original same/consecutive calendar-day confirmation retained, plus different actual specimen IDs and different nonmissing exact collection timestamps; no minimum elapsed time. Distinct timestamps do not prove independent venipunctures.','net_change_is_not_simple_event_deletion':True,'patient_rows_exported':False}
    (out/'S0_S1_event_changes_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'status':'completed','new_stays':len(new),'canceled_final_events':int(canceled.sum()),'postponed_first_events':int(postponed.sum()),'added_stay_events':int(t1.loc[new].event_eligible.sum()),'risk_time_change_days':net}),flush=True)

if __name__=='__main__':main()

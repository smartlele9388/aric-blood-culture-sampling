"""Independent aggregate check of S1 rebuild and model exports, without refitting."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from audit_baseline_and_period import sha, public_count, maxdiff


def table_check(old, new, keys):
    """Full outer coverage and field comparison; never export identifying rows."""
    assert not old.duplicated(keys).any() and not new.duplicated(keys).any()
    fields=[c for c in old if c not in keys]
    assert set(old)==set(new), 'Shared source schema changed'
    m=old.merge(new,on=keys,how='outer',suffixes=('_old','_new'),indicator=True,validate='one_to_one')
    assert m._merge.eq('both').all(), 'Shared source key coverage mismatch'
    diffs={}
    for c in fields:
        l,r=m[c+'_old'],m[c+'_new']
        if pd.api.types.is_numeric_dtype(l) and pd.api.types.is_numeric_dtype(r):
            neq=~np.isclose(l,r,rtol=0,atol=1e-10,equal_nan=True)
        else:
            neq=l.astype('string').fillna('<missing>').ne(r.astype('string').fillna('<missing>'))
        diffs[c]=int(np.sum(neq))
    assert not any(diffs.values()), 'Shared source field mismatch'
    return {'full_outer_key_count':len(m),'all_keys_match':True,'field_disagreements':diffs}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--s1-stage',type=Path,required=True)
    p.add_argument('--s0-stage',type=Path,required=True)
    p.add_argument('--audit-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args(); s1,s0,audit,out=[x.resolve() for x in [a.s1_stage,a.s0_stage,a.audit_root,a.output_root]]
    checkpoint=s1/'aggregate/S1_primary_input_checkpoint.json'
    if not checkpoint.exists(): raise RuntimeError('Complete S1 rebuild checkpoint required; do not inspect partially written input as a completed analysis')
    hash_record=s1/'aggregate/S1_shared_raw_source_hash_reconciliation.json'
    source_record=json.loads(hash_record.read_text())
    assert source_record['all_source_hashes_match'] and all(row['identical_raw_bytes'] and row['prior_sha256']==row['current_sha256'] for row in source_record['sources'])
    assert set(row['source'] for row in source_record['sources'])=={'icu/procedureevents.csv.gz','hosp/microbiologyevents.csv.gz','hosp/diagnoses_icd.csv.gz','hosp/labevents.csv.gz','icu/inputevents.csv.gz','hosp/prescriptions.csv.gz'}
    inputs={'S1_primary_input_checkpoint.json':checkpoint,'S1_rebuild_code':audit/'code/rebuild_s1_primary_stage.py','S1_shared_raw_source_hash_reconciliation.json':hash_record}
    sec,oldsec,asec=s1/'secure_derived_inputs',s0/'secure_derived_inputs',audit/'secure_derived_inputs'
    def read(path,dates=()):
        inputs[str(path.relative_to(path.parents[1]))+'_'+str(len(inputs))]=path
        return pd.read_csv(path,parse_dates=list(dates))
    risk=read(sec/'risk_final_day3.csv.gz',['day_start','stop_datetime','landmark_time'])
    samp=read(sec/'sampling_intervals_final_day3.csv.gz',['day_start','stop_datetime','landmark_time'])
    cohort=read(sec/'landmark_cohort_day3.csv.gz',['landmark_time','index_interval_end','followup_end'])
    first=read(sec/'clabsi_events_day3.csv.gz',['event_time'])
    terminal=read(asec/'S1_terminal_times.csv.gz',['observed_end'])
    oldcohort=read(oldsec/'landmark_cohort_day3.csv.gz')
    common=set(oldcohort.stay_id)
    assert common<=set(cohort.stay_id)
    windows=read(sec/'day_windows_day3.csv.gz')
    oldwindows=read(oldsec/'day_windows_day3.csv.gz')
    common_windows=windows.loc[windows.stay_id.isin(common)]
    reuse={'full_windows':table_check(oldwindows,common_windows,['stay_id','local_day'])}
    for name in ['corrected_features_day3.csv.gz','corrected_treatments_day3.csv.gz','antibiotic_exposure_day3.csv.gz']:
        f=read(sec/name);oldf=read(oldsec/name)
        reuse[name]=table_check(oldf,f.loc[f.stay_id.isin(common)],['stay_id','local_day'])
        newf=f.loc[~f.stay_id.isin(common)]
        extracted=read(s1/'newly_eligible_only/secure_derived_inputs'/name)
        reuse[name+'_newly_eligible_source_rows']=table_check(newf,extracted,['stay_id','local_day'])
    # Full outer key coverage against independent S1 terminal selection.
    final=risk.sort_values(['stay_id','t_start']).groupby('stay_id').tail(1)
    re=risk.groupby('stay_id',as_index=False).agg(observed_end=('stop_datetime','max'),event_eligible=('event','max'),risk_person_days=('duration_days','sum'))
    m=re.merge(terminal[['stay_id','observed_end','event_eligible','risk_person_days']],on='stay_id',how='outer',suffixes=('_risk','_audit'),indicator=True,validate='one_to_one')
    assert m._merge.eq('both').all() and set(risk.stay_id)==set(cohort.stay_id)
    independent_terminal_checks={'complete_cohort_risk_terminal_key_coverage':True,'matched_stays':len(m),
        'timestamp_disagreements':int(m.observed_end_risk.ne(m.observed_end_audit).sum()),
        'event_disagreements':int(m.event_eligible_risk.ne(m.event_eligible_audit).sum()),
        'per_stay_days_max_abs_difference':maxdiff(m.risk_person_days_risk,m.risk_person_days_audit)}
    assert independent_terminal_checks['timestamp_disagreements']==0 and independent_terminal_checks['event_disagreements']==0
    assert independent_terminal_checks['per_stay_days_max_abs_difference']<1e-10
    c=cohort.merge(first[['stay_id','event_time']],on='stay_id',how='left',validate='one_to_one')
    line_end=(c.index_interval_end-pd.Timedelta(nanoseconds=1)).dt.floor('D')+pd.Timedelta(days=2)
    risk_end=pd.concat([line_end,c.followup_end],axis=1).min(axis=1)
    valid=c.event_time.notna() & c.event_time.ge(c.landmark_time) & c.event_time.le(risk_end) & c.event_time.lt(line_end)
    end=risk_end.where(~valid,c.event_time)
    zero=valid & end.eq(c.landmark_time);end.loc[zero]+=pd.Timedelta(seconds=1)
    es=final[['stay_id','stop_datetime','event']].merge(pd.DataFrame({'stay_id':c.stay_id,'recomputed_end':end,'recomputed_event':valid.astype(int)}),on='stay_id',how='outer',indicator=True,validate='one_to_one')
    assert es._merge.eq('both').all()
    independent_terminal_checks['independently_recomputed_endpoint_disagreements']=int(es.stop_datetime.ne(es.recomputed_end).sum())
    independent_terminal_checks['independently_recomputed_event_disagreements']=int(es.event.ne(es.recomputed_event).sum())
    assert independent_terminal_checks['independently_recomputed_endpoint_disagreements']==0 and independent_terminal_checks['independently_recomputed_event_disagreements']==0
    required=['subject_id','hadm_id','stay_id','day_start','stop_datetime','t_start','t_stop','duration_days','event']
    assert not risk[required].isna().any().any()
    assert not risk.duplicated(['stay_id','t_start']).any() and risk.duration_days.gt(0).all()
    assert np.allclose((risk.stop_datetime-risk.day_start).dt.total_seconds()/86400,risk.duration_days,rtol=0,atol=1e-10)
    rr=risk.sort_values(['stay_id','t_start']);nextstart=rr.groupby('stay_id').t_start.shift(-1)
    assert np.allclose(nextstart[nextstart.notna()],rr.loc[nextstart.notna(),'t_stop'],rtol=0,atol=1e-10)
    assert rr.groupby('stay_id').event.sum().le(1).all() and int(rr.event.sum())==int(final.event.sum())
    assert risk[required].astype('string').equals(samp[required].astype('string'))
    # Reassign sampling independently; do not import the S1 assign function.
    collections=read(sec/'culture_collections_raw_day3.csv.gz',['event_time'])
    coll=collections.loc[collections.has_non_cancelled.eq(1)].reset_index(drop=True)
    assert coll.collection_key.is_unique
    assigned=np.full(len(coll),-1,dtype=np.int64);byadm=coll.groupby('hadm_id',sort=False).indices
    for _,g in risk.reset_index(drop=True).groupby('stay_id',sort=False):
        g=g.sort_values('day_start');ids=byadm.get(g.hadm_id.iloc[0],np.array([],dtype=int))
        if not len(ids):continue
        b=coll.iloc[ids];ts=b.event_time.astype('int64').to_numpy();starts=g.day_start.astype('int64').to_numpy();stops=g.stop_datetime.astype('int64').to_numpy()
        pos=np.searchsorted(starts,ts,side='right')-1;ok=pos>=0;ok[ok]&=ts[ok]<=stops[pos[ok]]
        ok&=b.event_time.notna().to_numpy()&b.subject_id.eq(g.subject_id.iloc[0]).to_numpy()
        assert not np.any(assigned[ids[ok]]>=0)
        assigned[ids[ok]]=g.index.to_numpy()[pos[ok]]
    ac=coll.loc[assigned>=0,['event_time']].copy();ac['risk_row']=assigned[assigned>=0]
    counts=ac.groupby('risk_row').size().reindex(range(len(risk)),fill_value=0)
    occasions=ac.drop_duplicates(['risk_row','event_time']).groupby('risk_row').size().reindex(range(len(risk)),fill_value=0)
    assert np.array_equal(counts.to_numpy(),samp.collection_count.to_numpy())
    assert np.array_equal(occasions.to_numpy(),samp.occasion_count.to_numpy())
    observed={'stays':int(risk.stay_id.nunique()),'patients':int(risk.subject_id.nunique()),'intervals':len(risk),'events':int(risk.event.sum()),'risk_person_days':float(risk.duration_days.sum()),'specimens':int(counts.sum()),'occasions':int(occasions.sum())}
    cp=json.loads(checkpoint.read_text());assert all(np.isclose(observed[k],cp['risk_summary'][v],rtol=0,atol=1e-9) for k,v in [('stays','stays'),('patients','patients'),('intervals','intervals'),('events','events'),('risk_person_days','risk_person_days')])
    assert observed['specimens']==cp['noncancelled_specimens_in_risk'] and observed['occasions']==cp['sampling_occasions_in_risk']
    # Check all available fit exports. Missing files are reported, not fabricated.
    model_checks={};ag=s1/'aggregate'
    for label,n,bcol,ecol in [('Cox','S1_main_cox_full_precision.csv','coefficient','hazard_ratio'),('Poisson','S1_sampling_primary_specimens_coefficients.csv','log_rate_ratio','rate_ratio')]:
        if not (ag/n).exists(): model_checks[label]={'status':'not yet available; no fit claimed by independent checker'};continue
        coef=read(ag/n)
        assert coef.intervals.eq(observed['intervals']).all() and coef.patients.eq(observed['patients']).all()
        if label=='Cox':
            vp=ag/'S1_main_cox_covariance.csv';vc=read(vp);cov=vc.set_index(vc.columns[0]).loc[coef.term,coef.term].to_numpy();z=norm.ppf(.975)
            assert coef.events.eq(observed['events']).all()
            fitdiag=read(ag/'S1_main_cox_diagnostics.csv').iloc[0].to_dict()
            assert fitdiag['fitted_rows']==observed['intervals'] and fitdiag['fitted_events']==observed['events']
            assert fitdiag['cluster']=='subject_id' and fitdiag['ties']=='efron' and fitdiag['n_parameters']==len(coef)
            ph=read(ag/'S1_main_cox_zph_log_event_time.csv').set_index('term')
            ph_note={'global_P':float(ph.loc['GLOBAL','p']),'lactate_availability_P':float(ph.loc['exposure_lactate_available','p']),'interpretation':'The global test does not establish proportional hazards for every term; lactate-availability diagnostic remains a reporting concern.'}
        else:
            vp=ag/'S1_sampling_primary_specimens_covariance.npz';inputs[vp.name]=vp;bundle=np.load(vp,allow_pickle=True)
            assert list(bundle['terms'])==coef.term.tolist() and np.allclose(bundle['coefficients'],coef[bcol],rtol=0,atol=1e-12)
            cov=bundle['covariance'];z=1.96
            assert coef.collections_or_occasions.eq(observed['specimens']).all()
            dp=ag/'S1_sampling_primary_diagnostics.json';inputs[dp.name]=dp;fitdiag=json.loads(dp.read_text())['model']
            assert fitdiag['intervals']==observed['intervals'] and fitdiag['patients']==observed['patients'] and fitdiag['outcome_count']==observed['specimens']
            assert fitdiag['cluster']=='subject_id' and fitdiag['offset']=='log(actual interval duration_days)'
            assert fitdiag['converged'] and fitdiag['design_rank']==len(coef) and fitdiag['information_rank']==len(coef) and fitdiag['max_abs_score']<.01
            ph_note=None
        beta=coef[bcol].to_numpy();se=np.sqrt(np.diag(cov))
        ds={'SE':maxdiff(se,coef.robust_se),'effect':maxdiff(np.exp(beta),coef[ecol]),'lower_CI':maxdiff(np.exp(beta-z*se),coef.ci_lower),'upper_CI':maxdiff(np.exp(beta+z*se),coef.ci_upper),'P_value':maxdiff(2*norm.sf(abs(beta/se)),coef.p_value)}
        assert max(ds.values())<1e-12 and np.isfinite(cov).all() and maxdiff(cov,cov.T)<1e-12
        assert np.linalg.matrix_rank(cov)==len(beta)
        model_checks[label]={'status':'executed full-output arithmetic and sample/configuration metadata verification; not refitted','parameters':len(beta),'covariance_rank':int(np.linalg.matrix_rank(cov)),'finite_covariance':True,'max_abs_arithmetic_differences':ds,'critical_value':float(z),'sample_and_fit_configuration':fitdiag,'PH_diagnostic_note':ph_note}
    out.mkdir(parents=True,exist_ok=True)
    result={'status':'executed','scope':'Independent code/read-only aggregate validation of S1 rebuild and any available model outputs; no raw scan or fitting','observed_S1_primary_inputs':observed,'raw_source_equality_record_check':{'status':'existing executed equality evidence read and internally checked; no new raw-table hash scan','sources':len(source_record['sources']),'all_prior_current_hash_pairs_match':True},'complete_original_full_window_reuse_verification':reuse,'independent_terminal_checks':independent_terminal_checks,'independent_sampling_per_interval_disagreements':{'specimens':0,'occasions':0},'model_output_verification':model_checks,'checked_at':datetime.now().astimezone().isoformat(),'source_sha256':{name:sha(path) for name,path in inputs.items()},'check_code_sha256':sha(Path(__file__)),'privacy':'No patient rows, dates, specimen keys or identifiers exported'}
    (out/'S1_independent_aggregate_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':'executed','S1_primary_counts':observed,'independent_terminal_checks':independent_terminal_checks,'sampling_disagreements':{'specimens':0,'occasions':0},'model_checks':model_checks},indent=2),flush=True)


if __name__=='__main__':main()

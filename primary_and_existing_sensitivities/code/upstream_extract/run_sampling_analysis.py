"""Route A sampling-process analysis. Source files are read-only; public output is aggregate.

Real risk intervals are inherited unchanged. Only collection-level technical corrections
documented before fitting are applied to the sampling outcome. No latent-infection model.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import warnings
import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm
import statsmodels
import statsmodels.api as sm

ROUTE = Path(__file__).resolve().parents[1]
BASE = ROUTE.parent
ROOT = BASE.parent
OUT = ROUTE / 'results'
SEC = ROUTE / 'secure_derived_inputs'
RAW = Path((__import__("os").environ.get("MIMIC_ROOT", "MIMIC_ROOT") + '/hosp/microbiologyevents.csv.gz'))
RISK = BASE / 'secure_derived_inputs/revised_risk_day3.csv.gz'
OLD_COLLECTIONS = BASE / 'secure_derived_inputs/culture_collections.csv.gz'
sys.path.insert(0, str(ROOT))
from scripts import rebuild_locked_mimic_package_v4 as source
from src.analysis.host_vulnerability_indication import CHRONIC_CONDITIONS, assert_aggregate_export

VULN = "C(exposure_vulnerability_category, Treatment(reference='low'))"
ADJUST = 'age_per_10y + female + C(icu_group) + C(admission_group) + multiple_line + C(cvc_type_group) + exposure_antibiotic_exposure + exposure_lactate_available + exposure_aki_assessable'
TIME = "C(time_band, Treatment(reference='0 to <2'))"
PRIMARY = f'{VULN} + exposure_indication_present + {ADJUST} + {TIME}'
COMPONENTS = 'documented_chronic_count + prior_ventilation + prior_creatinine_rise + prior_high_lactate'
COMPONENT = f'{COMPONENTS} + exposure_indication_present + {ADJUST} + {TIME}'
BANDS = ['0 to <2', '2 to <7', '7 to <14', '14 to 28']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(frame, name):
    assert_aggregate_export(frame)
    frame.to_csv(OUT / name, index=False)


def validate_risk(r):
    required = ['subject_id','hadm_id','stay_id','day_start','stop_datetime','t_start','t_stop','duration_days','event']
    if r[required].isna().any().any():
        raise ValueError('Required risk fields contain missing values')
    if r.duplicated(['stay_id','day_start']).any() or not r.duration_days.gt(0).all():
        raise ValueError('Duplicate interval start or nonpositive risk duration')
    if not np.allclose((r.stop_datetime-r.day_start).dt.total_seconds()/86400, r.duration_days):
        raise ValueError('Actual datetime duration differs from model duration')
    spans=[]
    for stay,g in r.groupby('stay_id', sort=False):
        g=g.sort_values('day_start')
        if g.subject_id.nunique()!=1 or g.hadm_id.nunique()!=1:
            raise ValueError('A stay maps to multiple patients or admissions')
        if len(g)>1 and not np.array_equal(g.day_start.to_numpy()[1:],g.stop_datetime.to_numpy()[:-1]):
            raise ValueError('Risk intervals overlap or have gaps within stay')
        if g.event.sum()>1 or (g.event.eq(1).any() and not g.event.iloc[-1]==1):
            raise ValueError('BSI must be a single terminal event')
        spans.append((stay,g.subject_id.iloc[0],g.hadm_id.iloc[0],g.day_start.iloc[0],g.stop_datetime.iloc[-1]))
    spans=pd.DataFrame(spans,columns=['stay_id','subject_id','hadm_id','start','stop'])
    if spans.groupby('hadm_id').subject_id.nunique().gt(1).any():
        raise ValueError('Hospital admission maps to multiple patients')
    # Endpoints are inclusive for terminal diagnostic collections: touching stays
    # would be ambiguous, so they cannot silently share a culture.
    for _,g in spans.groupby('hadm_id', sort=False):
        g=g.sort_values('start')
        if len(g)>1 and np.any(g.start.to_numpy()[1:]<=g.stop.to_numpy()[:-1]):
            raise ValueError('Hospital-to-stay risk windows overlap or touch')
    if not np.allclose(r.t_start,np.rint(r.t_start)) or r.t_start.min()<0 or r.t_stop.max()>28+1e-10:
        raise ValueError('Day-3 intervals must start on the landmark daily grid within 28 days')
    for boundary in (2,7,14):
        if ((r.t_start<boundary)&(r.t_stop>boundary)).any():
            raise ValueError('Risk interval crosses a prespecified time-band boundary')
    return {'intervals':len(r),'stays':int(r.stay_id.nunique()),'patients':int(r.subject_id.nunique()),
            'admissions':int(r.hadm_id.nunique()),'events':int(r.event.sum()),
            'risk_person_days':float(r.duration_days.sum()),'overlapping_or_touching_stay_windows':0,
            'intervals_crossing_time_band_boundary':0}


def assign_collections(r, c):
    """Assign each specimen once; internal boundaries go forward, final end is included."""
    if c.collection_key.isna().any() or c.collection_key.duplicated().any():
        raise ValueError('Collection keys must be complete and globally unique after deduplication')
    r=r.reset_index(drop=True)
    out=c.copy().reset_index(drop=True)
    assigned=np.full(len(out),-1,dtype=np.int64)
    grouped=out.groupby('hadm_id',sort=False).indices
    for _,g in r.groupby('stay_id',sort=False):
        g=g.sort_values('day_start')
        candidate=grouped.get(g.hadm_id.iloc[0],np.array([],dtype=int))
        if not len(candidate): continue
        e=out.iloc[candidate]
        starts=g.day_start.astype('int64').to_numpy(); ends=g.stop_datetime.astype('int64').to_numpy()
        times=e.event_time.astype('int64').to_numpy()
        pos=np.searchsorted(starts,times,side='right')-1
        valid=pos>=0
        valid[valid]=times[valid]<=ends[pos[valid]]
        valid &= e.event_time.notna().to_numpy()
        valid &= e.subject_id.eq(g.subject_id.iloc[0]).to_numpy()
        ci=candidate[valid]
        if (assigned[ci]>=0).any(): raise ValueError('A collection belongs to more than one stay/interval')
        assigned[ci]=g.index.to_numpy()[pos[valid]]
    out['risk_row']=assigned
    out=out[out.risk_row.ge(0)].copy()
    if out.collection_key.duplicated().any(): raise ValueError('Duplicate assigned collection')
    return out


def collapse_occasions(assigned):
    """A timestamp grouping sensitivity, not a clinically validated culture-set definition."""
    return assigned.groupby(['risk_row','event_time'],as_index=False).agg(positive=('positive','max'),specimens=('collection_key','size'))


def raw_collection_audit(r, prior, hashes):
    cache=SEC/'sampling_collections_source_checked.csv.gz'
    audit_path=OUT/'sampling_source_audit.json'
    if cache.exists() and audit_path.exists():
        audit=json.loads(audit_path.read_text())
        if audit.get('input_hashes')==hashes and audit.get('audit_version')==1:
            print('Using source-checked culture cache',flush=True)
            return pd.read_csv(cache,parse_dates=['event_time','charttime']),audit
    print('Auditing raw blood specimen keys and cancelled-only records',flush=True)
    pieces=[]; raw_rows=0; blood_rows=0; missing_time=0; missing_specimen=0
    cols=['subject_id','hadm_id','micro_specimen_id','chartdate','charttime','spec_type_desc','org_name']
    for chunk in pd.read_csv(RAW,usecols=cols,chunksize=500_000,dtype={'subject_id':'Int64','hadm_id':'Int64','micro_specimen_id':'Int64'}):
        raw_rows+=len(chunk)
        b=chunk[chunk.spec_type_desc.fillna('').str.contains('blood',case=False)].copy()
        blood_rows+=len(b)
        b['charttime']=pd.to_datetime(b.charttime,errors='raise')
        b['event_time']=b.charttime.fillna(pd.to_datetime(b.chartdate,errors='raise'))
        missing_time+=int(b.event_time.isna().sum())
        b=b[b.event_time.notna()].copy()
        missing_specimen+=int(b.micro_specimen_id.isna().sum())
        if b.micro_specimen_id.isna().any():
            raise RuntimeError('Unexpected missing specimen key: freeze fallback-key amendment before fitting')
        b['collection_key']='specimen:'+b.micro_specimen_id.astype(str)
        labels=b.org_name.fillna('').unique()
        positive_map={x:int(not source.invalid_organism_label(x) and source.norm(x)!='') for x in labels}
        cancelled_map={x:int(source.norm(x) in {'cancelled','canceled'}) for x in labels}
        b['positive']=b.org_name.fillna('').map(positive_map)
        b['has_non_cancelled']=1-b.org_name.fillna('').map(cancelled_map)
        b['cancelled_rows']=1-b.has_non_cancelled
        b['raw_rows']=1
        meta=['collection_key','subject_id','hadm_id','event_time','charttime']
        b=b.groupby(meta,dropna=False,as_index=False).agg(positive=('positive','max'),has_non_cancelled=('has_non_cancelled','max'),cancelled_rows=('cancelled_rows','sum'),raw_rows=('raw_rows','sum'))
        pieces.append(b)
    meta=pd.concat(pieces,ignore_index=True)
    grouped=meta.groupby('collection_key',sort=False)
    conflicts=grouped.agg(subjects=('subject_id','nunique'),admissions=('hadm_id','nunique'),exact_times=('charttime','nunique'),times=('event_time','nunique'))
    master=grouped.agg(subject_id=('subject_id','first'),hadm_id=('hadm_id','first'),event_time=('event_time','first'),charttime=('charttime','first'),positive=('positive','max'),has_non_cancelled=('has_non_cancelled','max'),cancelled_rows=('cancelled_rows','sum'),raw_rows=('raw_rows','sum')).reset_index()
    master['event_time']=master.charttime.fillna(master.event_time)
    target=master.hadm_id.isin(r.hadm_id.unique())
    target_conf=conflicts.loc[master.loc[target,'collection_key']]
    bad=(target_conf.subjects>1)|(target_conf.admissions>1)|(target_conf.exact_times>1)|((target_conf.exact_times==0)&(target_conf.times>1))
    if bad.any():
        pd.Series({'ambiguous_relevant_specimen_keys':int(bad.sum())}).to_json(OUT/'sampling_source_block.json')
        raise RuntimeError('Conflicting patient/admission/collection times in relevant raw specimen keys; requires amendment')
    master=master[target].copy()
    old=prior[prior.hadm_id.isin(r.hadm_id.unique())]
    check=master.merge(old[['collection_key','subject_id','hadm_id','event_time','positive']],on='collection_key',how='outer',suffixes=('_raw','_old'),indicator=True,validate='one_to_one')
    both=check['_merge'].eq('both')
    mismatch={col:int(check.loc[both,col+'_raw'].ne(check.loc[both,col+'_old']).sum()) for col in ['subject_id','hadm_id','event_time','positive']}
    audit={'audit_version':1,'input_hashes':hashes,'raw_microbiology_rows':raw_rows,'raw_blood_rows':blood_rows,'blood_rows_without_time':missing_time,'blood_rows_without_specimen_key':missing_specimen,'all_blood_specimen_keys':int(len(conflicts)),
           'all_keys_with_multiple_subjects':int(conflicts.subjects.gt(1).sum()),'all_keys_with_multiple_nonmissing_admissions':int(conflicts.admissions.gt(1).sum()),'all_keys_with_multiple_exact_times':int(conflicts.exact_times.gt(1).sum()),
           'relevant_admission_specimens':len(master),'relevant_ambiguous_specimen_keys':0,
           'relevant_keys_raw_only':int(check['_merge'].eq('left_only').sum()),'relevant_keys_old_only':int(check['_merge'].eq('right_only').sum()),'raw_vs_previous_mismatches':mismatch,
           'relevant_cancelled_only_specimens':int(master.has_non_cancelled.eq(0).sum()),'relevant_mixed_cancelled_and_noncancelled':int((master.has_non_cancelled.eq(1)&master.cancelled_rows.gt(0)).sum()),
           'global_reconciliation_policy':'A specimen must map to one patient and at most one nonmissing admission and exact collection timestamp. A single known charttime takes precedence over date-only rows. Conflicting relevant keys stop analysis.'}
    master.to_csv(cache,index=False)
    audit_path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return master,audit


def add_outcomes(r, assigned):
    x=r.copy().reset_index(drop=True)
    for col in ['collection_count','positive_collection_count','occasion_count','positive_occasion_count']:
        x[col]=0
    g=assigned.groupby('risk_row').agg(collection_count=('collection_key','size'),positive_collection_count=('positive','sum'))
    x.loc[g.index,g.columns]=g
    occasion=collapse_occasions(assigned)
    g=occasion.groupby('risk_row').agg(occasion_count=('event_time','size'),positive_occasion_count=('positive','sum'))
    x.loc[g.index,g.columns]=g
    x['has_collection']=x.collection_count.gt(0).astype(int)
    x['has_positive_collection']=x.positive_collection_count.gt(0).astype(int)
    x['time_band']=pd.Categorical(np.select([x.t_start<2,x.t_start<7,x.t_start<14],BANDS[:3],default=BANDS[3]),categories=BANDS)
    x['documented_chronic_count']=x[CHRONIC_CONDITIONS].fillna(0).sum(axis=1)
    if not x.documented_chronic_count.eq(x.corrected_chronic_score).all(): raise ValueError('Chronic score disagreement')
    if x.groupby('stay_id').documented_chronic_count.nunique().gt(1).any(): raise ValueError('Chronic descriptor changed within stay')
    x['prior_ventilation']=x.lag_mechanical_ventilation.astype(int)
    x['prior_creatinine_rise']=x.lag_aki.eq(1).astype(int)
    x['prior_high_lactate']=x.lag_lactate_high.eq(1).astype(int)
    if not np.allclose(x.documented_chronic_count+x.prior_ventilation+x.prior_creatinine_rise+x.prior_high_lactate,x.lag_corrected_score):
        raise ValueError('Component reconstruction does not reproduce corrected score')
    return x,occasion


def aggregate_sampling(x, groups):
    rows=[]
    for key,g in x.groupby(groups,observed=True):
        if not isinstance(key,tuple): key=(key,)
        c=int(g.collection_count.sum()); p=int(g.positive_collection_count.sum()); days=float(g.duration_days.sum())
        rows.append(dict(zip(groups,key))|{'intervals':len(g),'stays':int(g.stay_id.nunique()),'patients':int(g.subject_id.nunique()),'risk_person_days':days,
                     'collections':c,'sampled_intervals':int(g.has_collection.sum()),'positive_collections':p,'positive_sampled_intervals':int(g.has_positive_collection.sum()),
                     'sampling_occasions':int(g.occasion_count.sum()),'positive_occasions':int(g.positive_occasion_count.sum()),'detected_BSI_events':int(g.event.sum()),
                     'collections_per_1000_risk_days':c/days*1000,'occasions_per_1000_risk_days':g.occasion_count.sum()/days*1000,'positive_collection_fraction':p/c if c else np.nan})
    return pd.DataFrame(rows)


def fit_poisson(x,name,outcome,rhs):
    print('Fitting',name,flush=True)
    design=patsy.dmatrix(rhs,x,return_type='dataframe',NA_action='raise')
    y=x[outcome].to_numpy(dtype=float); offset=np.log(x.duration_days.to_numpy())
    model=sm.GLM(y,design,family=sm.families.Poisson(),offset=offset)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit=model.fit(maxiter=200,tol=1e-10,cov_type='cluster',cov_kwds={'groups':x.subject_id.to_numpy(),'use_correction':True})
    cov=fit.cov_params().to_numpy(); information=-model.hessian(fit.params); expected=fit.fittedvalues.to_numpy()
    rank=int(np.linalg.matrix_rank(cov)); n_parameters=len(design.columns)
    diag={'model':name,'outcome':outcome,'intervals':len(x),'stays':int(x.stay_id.nunique()),'patients':int(x.subject_id.nunique()),'outcome_count':int(y.sum()),
          'risk_person_days':float(x.duration_days.sum()),'parameters':n_parameters,'converged':bool(fit.converged),'design_rank':int(np.linalg.matrix_rank(design)),
          'covariance_rank':rank,'information_rank':int(np.linalg.matrix_rank(information)),'information_condition':float(np.linalg.cond(information)),
          'max_abs_score':float(np.abs(model.score(fit.params)).max()),'max_abs_coefficient':float(np.abs(fit.params).max()),'finite_covariance':bool(np.isfinite(cov).all()),
          'pearson_dispersion':float(fit.pearson_chi2/fit.df_resid),'min_fitted_count':float(expected.min()),'max_fitted_count':float(expected.max()),
          'min_fitted_rate_per_day':float((expected/x.duration_days).min()),'max_fitted_rate_per_day':float((expected/x.duration_days).max()),
          'small_sample_cluster_correction':True,'wald_reference':'standard normal','warnings':' | '.join(str(w.message) for w in caught)}
    if not diag['converged'] or not diag['finite_covariance'] or min(diag['design_rank'],rank,diag['information_rank'])!=n_parameters or diag['max_abs_score']>0.01 or diag['max_abs_coefficient']>15:
        raise RuntimeError(f'Poisson model diagnostic failure: {diag}')
    beta=fit.params.to_numpy(); se=np.sqrt(np.diag(cov)); z=beta/se
    table=pd.DataFrame({'model':name,'term':design.columns,'log_rate_ratio':beta,'robust_se':se,'rate_ratio':np.exp(beta),'ci_lower':np.exp(beta-1.96*se),'ci_upper':np.exp(beta+1.96*se),'wald_z':z,'p_value':2*norm.sf(np.abs(z)),
                        'intervals':len(x),'patients':x.subject_id.nunique(),'collections_or_occasions':int(y.sum()),'risk_person_days':x.duration_days.sum()})
    write_csv(table,name+'_coefficients.csv')
    np.savez_compressed(OUT/(name+'_covariance.npz'),terms=np.array(design.columns),coefficients=beta,covariance=cov)
    return table,diag


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--audit-only',action='store_true'); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); SEC.mkdir(parents=True,exist_ok=True)
    r=pd.read_csv(RISK,parse_dates=['day_start','stop_datetime','landmark_time','observed_end'])
    summary=validate_risk(r)
    assert (summary['intervals'],summary['stays'],summary['patients'],summary['events'])==(60272,13863,12757,268)
    assert np.isclose(summary['risk_person_days'],56370.34172453703)
    prior=pd.read_csv(OLD_COLLECTIONS,usecols=['subject_id','hadm_id','collection_key','event_time','charttime','positive'],parse_dates=['event_time','charttime'])
    if prior.collection_key.duplicated().any(): raise RuntimeError('Previously extracted collection keys are not unique')
    paths=[RAW,RISK,OLD_COLLECTIONS,BASE/'code/prepare_source_audit.py',ROOT/'scripts/rebuild_locked_mimic_package_v4.py',BASE/'code/run_reanalysis.py',ROUTE/'ANALYSIS_PROTOCOL_ROUTE_A.md',Path(__file__)]
    manifest=pd.DataFrame([{'source_file':str(p),'sha256':sha256(p),'bytes':p.stat().st_size} for p in paths])
    write_csv(manifest,'sampling_input_manifest.csv')
    hashes={str(p):sha256(p) for p in [RAW,RISK,OLD_COLLECTIONS]}
    master,audit=raw_collection_audit(r,prior,hashes)
    old_assigned=assign_collections(r,prior)
    all_assigned=assign_collections(r,master)
    corrected=master[master.has_non_cancelled.eq(1)].copy()
    assigned=assign_collections(r,corrected)
    x,occasions=add_outcomes(r,assigned)
    old_count=len(old_assigned); before_count=len(all_assigned)
    amendment={'old_extracted_collections_in_risk':old_count,'source_reconstructed_collections_before_cancellation_exclusion':before_count,'cancelled_only_collections_excluded_from_risk':before_count-len(assigned),
               'corrected_collections_in_risk':len(assigned),'corrected_positive_collections':int(assigned.positive.sum()),'corrected_occasions':len(occasions),
               'retained_mixed_cancelled_and_noncancelled_collections_in_risk':int(assigned.cancelled_rows.gt(0).sum()),'positive_collections_old':int(old_assigned.positive.sum()),
               'definition':'Count one raw micro_specimen_id once after aggregation of all blood-result rows. Exclude only specimens whose every raw row is explicitly cancelled/canceled. Retain negative/blank/no-growth records and any specimen with a noncancelled row. Preserve specimen positivity if any valid organism is recorded.'}
    (OUT/'sampling_technical_amendment.json').write_text(json.dumps(amendment,indent=2),encoding='utf-8')
    (OUT/'SAMPLING_TECHNICAL_AMENDMENT.md').write_text('# Sampling technical amendment\n\nFrozen before the new sampling models were fitted on 20 September 2026.\n\n'+amendment['definition']+'\n\nThe previous extraction retained cancelled-only specimens as sampling events. This correction distinguishes cancelled records from documented noncancelled blood specimens; cancellation is not classified as a negative result. A mixed specimen is retained. Source key, timing and hospital linkage checks precede this amendment; source data and the BSI risk set remain unchanged.\n\n```json\n'+json.dumps(amendment,indent=2)+'\n```\n\nSpecimen identifiers do not establish independent venipunctures or verified culture sets. The same-stay, same-timestamp occasion sensitivity addresses this remaining uncertainty. Source file hashes are in sampling_input_manifest.csv.\n',encoding='utf-8')
    terminal=assigned.event_time.to_numpy()==x.loc[assigned.risk_row,'stop_datetime'].to_numpy()
    bsi_terminal=terminal & x.loc[assigned.risk_row,'event'].eq(1).to_numpy()
    summary.update(amendment)
    summary.update({'sampled_intervals':int(x.has_collection.sum()),'positive_sampled_intervals':int(x.has_positive_collection.sum()),'stays_with_collection':int(x.loc[x.has_collection.eq(1),'stay_id'].nunique()),
                    'patients_with_collection':int(x.loc[x.has_collection.eq(1),'subject_id'].nunique()),'terminal_time_collections':int(terminal.sum()),'BSI_terminal_time_collections':int(bsi_terminal.sum()),
                    'BSI_stays_with_terminal_collection':int(x.loc[assigned.loc[bsi_terminal,'risk_row'],'stay_id'].nunique()),'date_only_collections_in_risk':int(assigned.charttime.isna().sum()),
                    'multiple_specimens_collapsed_to_occasions':len(assigned)-len(occasions),'occasions_with_multiple_specimens':int(occasions.specimens.gt(1).sum()),'positive_occasions':int(occasions.positive.sum()),
                    'no_unassigned_collection_exported':True,'python_version':platform.python_version(),'pandas_version':pd.__version__,'numpy_version':np.__version__,'statsmodels_version':statsmodels.__version__})
    assert summary['BSI_stays_with_terminal_collection']==268
    x.to_csv(SEC/'sampling_intervals_day3.csv.gz',index=False)
    for groups,name in [(['exposure_vulnerability_category'],'sampling_by_vulnerability.csv'),(['exposure_indication_present'],'sampling_by_therapy.csv'),(['exposure_vulnerability_category','exposure_indication_present'],'sampling_by_six_states.csv'),(['time_band'],'sampling_by_time_band.csv')]:
        write_csv(aggregate_sampling(x,groups),name)
    (OUT/'sampling_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    if args.audit_only: return
    effects=[]; diagnostics=[]
    for name,outcome,rhs in [('sampling_primary_specimens','collection_count',PRIMARY),('sampling_components_specimens','collection_count',COMPONENT),('sampling_primary_occasions','occasion_count',PRIMARY),('sampling_components_occasions','occasion_count',COMPONENT)]:
        table,diag=fit_poisson(x,name,outcome,rhs); effects.append(table); diagnostics.append(diag)
    all_effects=pd.concat(effects,ignore_index=True)
    write_csv(all_effects,'sampling_all_coefficients.csv')
    selected=all_effects.term.str.startswith('C(exposure_vulnerability_category')|all_effects.term.isin(['exposure_indication_present','documented_chronic_count','prior_ventilation','prior_creatinine_rise','prior_high_lactate'])
    write_csv(all_effects[selected],'sampling_main_effects.csv')
    write_csv(pd.DataFrame(diagnostics),'sampling_model_diagnostics.csv')
    (OUT/'sampling_models_complete.json').write_text(json.dumps({'completed_models':len(diagnostics),'script_sha256':sha256(__file__),'all_converged':all(d['converged'] for d in diagnostics),'all_finite_full_rank_covariance':all(d['finite_covariance'] and d['covariance_rank']==d['parameters'] for d in diagnostics)},indent=2),encoding='utf-8')
    print('SAMPLING ANALYSIS COMPLETE',flush=True)


if __name__=='__main__':
    main()

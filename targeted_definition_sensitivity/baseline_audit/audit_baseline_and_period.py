"""Audit existing raw-built ARIC stage inputs and aggregate model exports.

Read-only inputs. No fitting, raw microbiology rescan, or patient-row export.
The period assessment maps admission years through patient anchor-year groups;
it never treats shifted MIMIC calendar years as true admission years.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import numpy as np
import pandas as pd
from scipy.stats import norm


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def public_count(n):
    n = int(n)
    return n if n == 0 or n >= 11 else '<11 suppressed'


def maxdiff(a, b):
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).max())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage-root', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    start = datetime.now().astimezone().isoformat()
    stage, raw, out = a.stage_root.resolve(), a.raw_root.resolve(), a.output_root.resolve()
    if out == stage or out == raw or stage in out.parents or raw in out.parents:
        raise ValueError('Output must be independent of the authorized input directories')
    out.mkdir(parents=True, exist_ok=True)
    aggregate = out / 'aggregate'
    aggregate.mkdir(exist_ok=True)
    inputs = {}
    for n in ('risk_final_day3.csv.gz', 'sampling_intervals_final_day3.csv.gz',
              'landmark_cohort_day3.csv.gz', 'clabsi_events_day3.csv.gz',
              'culture_collections_raw_day3.csv.gz'):
        inputs[n] = stage / 'secure_derived_inputs' / n
    risk = pd.read_csv(inputs['risk_final_day3.csv.gz'])
    sampling = pd.read_csv(inputs['sampling_intervals_final_day3.csv.gz'])
    cohort = pd.read_csv(inputs['landmark_cohort_day3.csv.gz'])
    events = pd.read_csv(inputs['clabsi_events_day3.csv.gz'])
    cultures = pd.read_csv(inputs['culture_collections_raw_day3.csv.gz'])
    for frame in (risk, sampling, cohort, events, cultures):
        for col in ('landmark_time','day_start','day_end','stop_datetime','followup_end',
                    'index_interval_start','index_interval_end','event_time','charttime',
                    'intime','outtime','dischtime','deathtime'):
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], errors='raise')
    expected = {'ICU_stays':13863,'patients':12757,'risk_intervals':60272,
                'detected_first_BSI_events':268,'risk_person_days':56370.34172453703,
                'noncancelled_specimens_in_risk':19160,'exact_time_sampling_occasions':16086}
    counts = {'ICU_stays':int(risk.stay_id.nunique()),'patients':int(risk.subject_id.nunique()),
              'risk_intervals':len(risk),'detected_first_BSI_events':int(risk.event.sum()),
              'risk_person_days':float(risk.duration_days.sum()),
              'noncancelled_specimens_in_risk':int(sampling.collection_count.sum()),
              'exact_time_sampling_occasions':int(sampling.occasion_count.sum())}
    comparison = [{'metric':k,'expected_previous':v,'independently_observed':counts[k],
                   'difference':counts[k]-v,'matches':bool(np.isclose(counts[k],v,rtol=0,atol=1e-9))}
                  for k,v in expected.items()]
    pd.DataFrame(comparison).to_csv(aggregate/'baseline_totals_comparison.csv',index=False)
    assert all(x['matches'] for x in comparison), 'Baseline mismatch; stop and explain rather than force matching'
    checks = {}
    required = ['subject_id','hadm_id','stay_id','day_start','stop_datetime','t_start','t_stop','duration_days','event']
    checks['required_risk_fields_missing'] = int(risk[required].isna().sum().sum())
    checks['duplicate_stay_start_keys'] = int(risk.duplicated(['stay_id','t_start']).sum())
    checks['nonpositive_intervals'] = int(risk.duration_days.le(0).sum())
    checks['duration_datetime_max_abs_difference_days'] = maxdiff((risk.stop_datetime-risk.day_start).dt.total_seconds()/86400,risk.duration_days)
    checks['start_landmark_max_abs_difference_days'] = maxdiff((risk.day_start-risk.landmark_time).dt.total_seconds()/86400,risk.t_start)
    checks['stop_landmark_max_abs_difference_days'] = maxdiff((risk.stop_datetime-risk.landmark_time).dt.total_seconds()/86400,risk.t_stop)
    checks['sampling_risk_row_field_disagreements'] = int((risk[required].astype('string').fillna('<missing>').to_numpy()!=sampling[required].astype('string').fillna('<missing>').to_numpy()).sum())
    checks['cohort_or_risk_stays_missing_from_other'] = int(len(set(cohort.stay_id)^set(risk.stay_id)))
    r = risk.sort_values(['stay_id','t_start'])
    prior_stop = r.groupby('stay_id').t_stop.shift()
    checks['within_stay_gaps_or_overlaps'] = int((prior_stop.notna() & ~np.isclose(prior_stop.fillna(0),r.t_start,rtol=0,atol=1e-10)).sum())
    checks['stays_not_entering_at_zero'] = int(r.groupby('stay_id').t_start.min().ne(0).sum())
    checks['stays_multiple_events'] = int(r.groupby('stay_id').event.sum().gt(1).sum())
    final = r.groupby('stay_id').tail(1)
    checks['events_not_terminal'] = int(r.event.sum()-final.event.sum())
    checks['stays_with_multiple_patient_ids'] = int(r.groupby('stay_id').subject_id.nunique().gt(1).sum())
    checks['stays_with_multiple_admission_ids'] = int(r.groupby('stay_id').hadm_id.nunique().gt(1).sum())
    checks['patient_clusters_with_multiple_stays'] = int(r[['subject_id','stay_id']].drop_duplicates().groupby('subject_id').size().gt(1).sum())
    checks['maximum_followup_days'] = float(r.t_stop.max())
    # Independently reconstruct the candidate-event eligibility and true terminal time.
    candidate = cohort.merge(events[['stay_id','event_time']],on='stay_id',how='left',validate='one_to_one')
    candidate['line_window_end'] = (candidate.index_interval_end-pd.Timedelta(nanoseconds=1)).dt.floor('D')+pd.Timedelta(days=2)
    candidate['risk_end'] = candidate[['line_window_end','followup_end']].min(axis=1)
    eligible = candidate.event_time.notna() & candidate.event_time.ge(candidate.landmark_time) & candidate.event_time.le(candidate.risk_end) & candidate.event_time.lt(candidate.line_window_end)
    observed_end = candidate.risk_end.where(~eligible,candidate.event_time)
    zero = eligible & observed_end.eq(candidate.landmark_time)
    observed_end.loc[zero] += pd.Timedelta(seconds=1)
    checks['reconstructed_event_eligible_count'] = int(eligible.sum())
    checks['zero_time_terminal_event_adjustments'] = public_count(zero.sum())
    endpoint_compare = final[['stay_id','stop_datetime','event']].merge(pd.DataFrame({'stay_id':candidate.stay_id,'reconstructed_end':observed_end,'reconstructed_event':eligible.astype(int)}),on='stay_id',validate='one_to_one')
    checks['reconstructed_endpoint_disagreements'] = int(endpoint_compare.stop_datetime.ne(endpoint_compare.reconstructed_end).sum())
    checks['reconstructed_event_indicator_disagreements'] = int(endpoint_compare.event.ne(endpoint_compare.reconstructed_event).sum())
    event_rows = r.loc[r.event.eq(1),['stay_id','stop_datetime']].merge(events[['stay_id','event_time']],on='stay_id',validate='one_to_one')
    checks['terminal_event_timestamp_disagreements'] = int(event_rows.stop_datetime.ne(event_rows.event_time).sum())
    candidate_detail = {
        'raw_stage_candidate_first_detected_BSI_rows':len(events),
        'final_risk_eligible_events':int(eligible.sum()),
        'candidate_events_outside_first_episode_attribution_window_or_followup':public_count((candidate.event_time.notna() & ~eligible).sum()),
        'candidates_at_or_after_first_episode_attribution_end':public_count((candidate.event_time.notna() & candidate.event_time.ge(candidate.line_window_end)).sum()),
        'candidates_after_risk_exit':public_count((candidate.event_time.notna() & candidate.event_time.gt(candidate.risk_end)).sum()),
        'explanation':'The source event table is a candidate table, not the final first-episode risk-set event count. Eligibility and endpoint reconstruction are checked independently.'}
    # Assign noncancelled specimen keys independently, reproducing start-inclusive,
    # internal-boundary-forward, final-end-inclusive descriptive sampling rules.
    cultures = cultures.loc[cultures.has_non_cancelled.eq(1)].copy().reset_index(drop=True)
    assert not cultures.collection_key.duplicated().any()
    assigned = np.full(len(cultures),-1,dtype=np.int64)
    by_admission = cultures.groupby('hadm_id',sort=False).indices
    r0 = risk.reset_index(drop=True)
    for _, g in r0.groupby('stay_id',sort=False):
        g = g.sort_values('day_start')
        ci = by_admission.get(g.hadm_id.iloc[0],np.array([],dtype=int))
        if not len(ci): continue
        c = cultures.iloc[ci]
        ts = c.event_time.astype('int64').to_numpy()
        starts = g.day_start.astype('int64').to_numpy()
        stops = g.stop_datetime.astype('int64').to_numpy()
        pos = np.searchsorted(starts,ts,side='right')-1
        ok = pos>=0
        ok[ok] &= ts[ok] <= stops[pos[ok]]
        ok &= c.event_time.notna().to_numpy() & c.subject_id.eq(g.subject_id.iloc[0]).to_numpy()
        assert not np.any(assigned[ci[ok]]>=0), 'Duplicate specimen assignment'
        assigned[ci[ok]] = g.index.to_numpy()[pos[ok]]
    c = cultures.loc[assigned>=0,['event_time']].copy()
    c['risk_row'] = assigned[assigned>=0]
    recount = c.groupby('risk_row').size().reindex(range(len(risk)),fill_value=0)
    occasions = c.drop_duplicates(['risk_row','event_time']).groupby('risk_row').size().reindex(range(len(risk)),fill_value=0)
    checks['independent_specimen_total'] = int(recount.sum())
    checks['independent_occasion_total'] = int(occasions.sum())
    checks['specimen_interval_count_disagreements'] = int(np.count_nonzero(recount.to_numpy()!=sampling.collection_count.to_numpy()))
    checks['occasion_interval_count_disagreements'] = int(np.count_nonzero(occasions.to_numpy()!=sampling.occasion_count.to_numpy()))
    must_zero = ['required_risk_fields_missing','duplicate_stay_start_keys','nonpositive_intervals','sampling_risk_row_field_disagreements','cohort_or_risk_stays_missing_from_other','within_stay_gaps_or_overlaps','stays_not_entering_at_zero','stays_multiple_events','events_not_terminal','stays_with_multiple_patient_ids','stays_with_multiple_admission_ids','reconstructed_endpoint_disagreements','reconstructed_event_indicator_disagreements','terminal_event_timestamp_disagreements','specimen_interval_count_disagreements','occasion_interval_count_disagreements']
    assert all(checks[k]==0 for k in must_zero)
    assert all(checks[k]<1e-10 for k in ['duration_datetime_max_abs_difference_days','start_landmark_max_abs_difference_days','stop_landmark_max_abs_difference_days'])
    # Independently join the immediately preceding day window and its raw-built
    # acute features. This checks the actual lagged model inputs rather than
    # assuming a field beginning with "lag_" is temporally valid.
    feature_path=stage/'secure_derived_inputs/corrected_features_day3.csv.gz'
    treatment_path=stage/'secure_derived_inputs/corrected_treatments_day3.csv.gz'
    antibiotic_path=stage/'secure_derived_inputs/antibiotic_exposure_day3.csv.gz'
    day_path=stage/'secure_derived_inputs/day_windows_day3.csv.gz'
    inputs.update({'corrected_features_day3.csv.gz':feature_path,'corrected_treatments_day3.csv.gz':treatment_path,'antibiotic_exposure_day3.csv.gz':antibiotic_path,'day_windows_day3.csv.gz':day_path})
    prior=pd.read_csv(day_path,usecols=['stay_id','local_day','day_end'],parse_dates=['day_end'])
    for path in [feature_path,treatment_path,antibiotic_path]:
        src=pd.read_csv(path)
        src=src.drop(columns=[col for col in src.columns if col not in ['stay_id','local_day'] and col in prior.columns])
        prior=prior.merge(src,on=['stay_id','local_day'],validate='one_to_one')
    prior['local_day']+=1
    prior=prior.rename(columns={col:'prior_source_'+col for col in prior if col not in ['stay_id','local_day']})
    link=risk.merge(prior,on=['stay_id','local_day'],how='left',validate='many_to_one')
    lag_checks={}
    lag_checks['unmatched_preceding_day_windows']=int(link.prior_source_day_end.isna().sum())
    lag_checks['preceding_window_end_after_risk_start']=int(link.prior_source_day_end.gt(link.day_start).sum())
    for name in ['aki','lactate_high','lactate_available','aki_assessable','vasopressor','mechanical_ventilation','renal_replacement_therapy','antibiotic_exposure']:
        lag_checks['lag_'+name+'_source_disagreements']=int((~np.isclose(link['lag_'+name],link['prior_source_'+name],rtol=0,atol=1e-10,equal_nan=True)).sum())
    for name in ['mechanical_ventilation','vasopressor','renal_replacement_therapy','antibiotic_exposure','lactate_available','aki_assessable']:
        lag_checks['exposure_'+name+'_lag_alias_disagreements']=int(link['exposure_'+name].ne(link['lag_'+name]).sum())
    lag_checks['vulnerability_score_lag_alias_disagreements']=int(link.exposure_vulnerability_burden_score.ne(link.lag_corrected_score).sum())
    lag_checks['indication_prior_components_disagreements']=int(link.exposure_indication_present.ne(link[['prior_source_vasopressor','prior_source_renal_replacement_therapy','prior_source_tpn']].max(axis=1)).sum())
    reconstructed_score=link.corrected_chronic_score+link.lag_mechanical_ventilation+link.lag_aki.eq(1)+link.lag_lactate_high.eq(1)
    lag_checks['score_component_disagreements']=int(link.exposure_vulnerability_burden_score.ne(reconstructed_score).sum())
    reconstructed_category=np.where(reconstructed_score.le(1),'low',np.where(reconstructed_score.le(3),'moderate','high'))
    lag_checks['fixed_category_threshold_disagreements']=int(link.exposure_vulnerability_category.ne(reconstructed_category).sum())
    lag_checks['within_stay_discharge_code_count_changes']=int(link.groupby('stay_id').corrected_chronic_score.nunique().gt(1).sum())
    assert all(v==0 for v in lag_checks.values()), 'Preceding-window covariate mismatch'
    checks['preceding_window_covariate_checks']=lag_checks
    checks['temporal_scope_note']='Acute feature/treatment lag inputs match the preceding source window. Static discharge-code counts remain retrospective and are not proven to have been recorded before observation.'
    (aggregate/'baseline_risk_and_count_integrity.json').write_text(json.dumps({'checks':checks,'candidate_event_reconciliation':candidate_detail},indent=2),encoding='utf-8')
    # Full-precision coefficient/covariance arithmetic verification, no fitting.
    model_checks = {}
    ag = stage/'aggregate'
    for label, n, beta_col, effect_col in [('Cox','raw_reproduced_main_cox_full_precision.csv','coefficient','hazard_ratio'),('Poisson','raw_reproduced_sampling_primary_specimens_coefficients.csv','log_rate_ratio','rate_ratio')]:
        coef = pd.read_csv(ag/n)
        inputs[n] = ag/n
        if label=='Cox':
            vpath=ag/'raw_reproduced_main_cox_covariance.csv'
            cov = pd.read_csv(vpath,index_col=0).loc[coef.term,coef.term].to_numpy()
            dpath=ag/'raw_reproduced_main_cox_diagnostics.csv'
            diag = pd.read_csv(dpath).iloc[0].to_dict()
            code_path=stage/'fit_main_cox_survival.R'
        else:
            vpath=ag/'raw_reproduced_sampling_primary_specimens_covariance.npz'
            saved=np.load(vpath,allow_pickle=True)
            assert list(saved['terms'])==coef.term.tolist()
            cov=saved['covariance']
            assert np.allclose(saved['coefficients'],coef[beta_col],rtol=0,atol=1e-12)
            dpath=ag/'raw_reproduced_sampling_primary_diagnostics.json'
            diag=json.loads(dpath.read_text())['model']
            code_path=stage/'portable_code/ARIC_revision_20260920/route_A/code/run_sampling_analysis.py'
        inputs[vpath.name]=vpath; inputs[dpath.name]=dpath; inputs['audited_'+code_path.name]=code_path
        beta=coef[beta_col].to_numpy(); se=np.sqrt(np.diag(cov))
        # Verify against the actual export rule, not an assumed critical value.
        # The Poisson source uses the conventional 1.96 approximation; R Cox
        # uses qnorm(.975). The former is not an analysis failure.
        z=norm.ppf(.975) if label=='Cox' else 1.96
        diffs={'robust_se_max_abs_difference':maxdiff(se,coef.robust_se),
               'effect_max_abs_difference':maxdiff(np.exp(beta),coef[effect_col]),
               'lower_CI_max_abs_difference':maxdiff(np.exp(beta-z*se),coef.ci_lower),
               'upper_CI_max_abs_difference':maxdiff(np.exp(beta+z*se),coef.ci_upper),
               'P_max_abs_difference':maxdiff(2*norm.sf(abs(beta/se)),coef.p_value)}
        assert max(diffs.values())<1e-12
        assert np.isfinite(cov).all() and maxdiff(cov,cov.T)<1e-12
        model_checks[label]={'parameters':len(beta),'finite_coefficients':bool(np.isfinite(beta).all()),'finite_covariance':bool(np.isfinite(cov).all()),'covariance_rank':int(np.linalg.matrix_rank(cov)),'covariance_min_eigenvalue':float(np.linalg.eigvalsh(cov).min()),'CI_critical_value_used_by_existing_export':float(z),'output_arithmetic_checks':diffs,'source_configuration':diag,'verification_scope':'Existing source code and full-precision exports checked; model not refitted in this audit.'}
        coef.to_csv(aggregate/('verified_'+n),index=False)
        pd.DataFrame(cov,index=coef.term,columns=coef.term).to_csv(aggregate/(label.lower()+'_verified_robust_covariance.csv'))
    (aggregate/'primary_model_output_audit.json').write_text(json.dumps(model_checks,indent=2),encoding='utf-8')
    # Anchor-based broad-period feasibility. Read only small source tables.
    pat_path=raw/'hosp/patients.csv.gz'; adm_path=raw/'hosp/admissions.csv.gz'
    patients=pd.read_csv(pat_path,usecols=['subject_id','anchor_year','anchor_year_group'])
    admissions=pd.read_csv(adm_path,usecols=['subject_id','hadm_id','admittime'],parse_dates=['admittime'])
    inputs['patients.csv.gz']=pat_path; inputs['admissions.csv.gz']=adm_path
    period=cohort[['subject_id','hadm_id','stay_id']].merge(admissions,on=['subject_id','hadm_id'],how='left',validate='one_to_one').merge(patients,on='subject_id',how='left',validate='many_to_one')
    extracted=period.anchor_year_group.astype('string').str.extract(r'^(\d{4})\s*-\s*(\d{4})$').astype('Int64')
    period['real_year_lower']=extracted[0]+period.admittime.dt.year-period.anchor_year
    period['real_year_upper']=extracted[1]+period.admittime.dt.year-period.anchor_year
    complete=period[['admittime','anchor_year','real_year_lower','real_year_upper']].notna().all(axis=1)
    intervals=[('2008–2013',2008,2013),('2014–2019',2014,2019),('2020–2022',2020,2022)]
    period['broad_period']='boundary-crossing or outside proposed coverage'
    for name,lower,upper in intervals:
        assigned_period=complete & period.real_year_lower.ge(lower) & period.real_year_upper.le(upper)
        period.loc[assigned_period,'broad_period']=name
    period.loc[~complete,'broad_period']='unavailable anchor/admission fields'
    period_rows=[]
    for name,g in period.groupby('broad_period',sort=True):
        local_risk=risk.loc[risk.stay_id.isin(g.stay_id)]
        local_sampling=sampling.loc[sampling.stay_id.isin(g.stay_id)]
        period_rows.append({'period_status':name,'stays':public_count(len(g)),'patients':public_count(g.subject_id.nunique()),'risk_intervals':public_count(len(local_risk)),'first_detected_BSI_events':public_count(local_risk.event.sum()),'risk_person_days':float(local_risk.duration_days.sum()),'specimens':public_count(local_sampling.collection_count.sum())})
    pd.DataFrame(period_rows).to_csv(aggregate/'anchor_period_feasibility.csv',index=False)
    cross=complete & period.broad_period.eq('boundary-crossing or outside proposed coverage')
    cross_2013_2014 = complete & period.real_year_lower.le(2013) & period.real_year_upper.ge(2014)
    cross_2019_2020 = complete & period.real_year_lower.le(2019) & period.real_year_upper.ge(2020)
    outside=complete & (period.real_year_lower.lt(2008) | period.real_year_upper.gt(2022))
    field_note={'status':'executed source-field and broad-period feasibility audit; no period model fitted',
        'source':'MIMIC-IV v3.1 hosp/patients.csv.gz (anchor_year, anchor_year_group) joined to hosp/admissions.csv.gz (admittime) by authorized local keys',
        'official_source':'https://physionet.org/content/mimiciv/3.1/',
        'formula':'possible real admission year lower/upper = anchor_year_group lower/upper + year(admittime) - anchor_year',
        'exact_real_admission_year_available':False,'shifted_calendar_year_is_real_year':False,
        'candidate_broad_periods':[n for n,_,_ in intervals],
        'cohort_stays':len(period),'complete_anchor_admission_fields':public_count(complete.sum()),
        'unavailable_anchor_admission_fields':public_count((~complete).sum()),
        'fully_assignable_to_one_candidate_period':public_count((complete & ~cross).sum()),
        'boundary_crossing_or_outside_candidate_coverage':public_count(cross.sum()),
        'crosses_2013_2014_boundary':public_count(cross_2013_2014.sum()),
        'crosses_2019_2020_boundary':public_count(cross_2019_2020.sum()),
        'possible_year_interval_extends_outside_2008_2022':public_count(outside.sum()),
        'assessment':'Broad eras can be assigned only when the full possible admission-year interval is contained in that era. Boundary-crossing cases cannot be classified as an exact era without an explicit uncertainty-handling rule. This audit is not evidence of exact true dates or temporal transportability.',
        'cache_retains_anchor_fields':False,
        'cache_assessment':'The current cohort/risk cache lacks anchor_year and anchor_year_group; feasibility used the authorized small raw patients/admissions tables. Patient-level joins remain in memory and are not exported.'}
    (aggregate/'anchor_period_source_assessment.json').write_text(json.dumps(field_note,indent=2),encoding='utf-8')
    manifest={'started_at':start,'completed_at':datetime.now().astimezone().isoformat(),'software':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__},'input_sha256':{name:sha(path) for name,path in inputs.items()},'code_sha256':sha(Path(__file__)),'privacy':'Only nonpatient aggregate checks, coefficients and covariance are exported; no patient IDs, dates or rows.'}
    (aggregate/'baseline_audit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    status={'executed':['Independent stage-based cohort totals, timing and event-exit reconstruction','Independent specimen and exact-time-occasion assignment','Full-precision model export/covariance/SE/CI/P-value arithmetic checks and fitting-code configuration review','Anchor-based broad-era feasibility assessment using authorized small raw source tables'],
            'not_triggered':['S1 cohort/model refitting is conditional on commensal-rule audit; not performed by this independent baseline audit','Period-adjusted fitting requires the targeted instruction and a fixed uncertainty-handling rule'],
            'data_insufficient':['Exact true admission years are not recoverable from shifted dates and 3-year anchor intervals','Discharge-diagnosis recording time is not supplied by these source fields'],
            'execution_failed':[],
            'resolved_execution_attempt_failures':['Initial audit assertion assumed qnorm(.975) for both exported models; source inspection showed Poisson export uses 1.96. Audit corrected to the documented source rule and rerun. No source analysis was changed.']}
    (aggregate/'execution_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps({'baseline_counts':counts,'all_required_baseline_checks_passed':True,'candidate_event_reconciliation':candidate_detail,'period_feasibility':field_note,'status':status},ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':
    main()

"""Audit broad eras from authorized small raw sources and fit two S0 Poisson models.

Patient rows are written exclusively to secure_derived_inputs. No original file is
modified. Statistical specification is copied from the frozen S0 primary model;
the only modeled delta is the fixed broad_period factor in the same restricted
cohort. Called once by period_run_all.py, not a subgroup/model search.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import patsy
import scipy
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels

PRIMARY = "C(exposure_vulnerability_category, Treatment(reference='low')) + exposure_indication_present + age_per_10y + female + C(icu_group) + C(admission_group) + multiple_line + C(cvc_type_group) + exposure_antibiotic_exposure + exposure_lactate_available + exposure_aki_assessable + C(time_band, Treatment(reference='0 to <2'))"
PERIOD = "C(broad_period, Treatment(reference='2008-2013'))"
ERAS = [('2008-2013', 2008, 2013), ('2014-2019', 2014, 2019), ('2020-2022', 2020, 2022)]

def stamp(): return datetime.now().astimezone().isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def writejson(p, x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def fit_poisson(x, name, rhs, out):
    started=stamp()
    design=patsy.dmatrix(rhs,x,return_type='dataframe',NA_action='raise')
    y=x.collection_count.to_numpy(dtype=float)
    offset=np.log(x.duration_days.to_numpy())
    model=sm.GLM(y,design,family=sm.families.Poisson(),offset=offset)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit=model.fit(maxiter=200,tol=1e-10,cov_type='cluster',cov_kwds={'groups':x.subject_id.to_numpy(),'use_correction':True})
    cov=fit.cov_params().to_numpy(); beta=fit.params.to_numpy(); se=np.sqrt(np.diag(cov))
    info=-model.hessian(fit.params); n,p=design.shape; g=x.subject_id.nunique()
    # Independent cluster sandwich check: subject sums of X*(y-mu), full bread.
    score=np.asarray(design)*(y-np.asarray(fit.fittedvalues))[:,None]
    grouped=pd.DataFrame(score).groupby(x.subject_id.to_numpy()).sum().to_numpy()
    bread=np.linalg.inv(info)
    correction=g/(g-1)*(n-1)/(n-p)
    independent=bread @ (grouped.T @ grouped) @ bread * correction
    diff=float(np.max(np.abs(independent-cov)))
    assert np.allclose(independent,cov,rtol=1e-9,atol=1e-12), diff
    diag={'model':name,'status':'completed','started_at':started,'completed_at':stamp(),
          'formula':rhs,'outcome':'collection_count','intervals':n,'stays':int(x.stay_id.nunique()),
          'patients':int(g),'outcome_count':int(y.sum()),'BSI_events':int(x.event.sum()),
          'risk_person_days':float(x.duration_days.sum()),'parameters':p,
          'converged':bool(fit.converged),'design_rank':int(np.linalg.matrix_rank(design)),
          'covariance_rank':int(np.linalg.matrix_rank(cov)),'information_rank':int(np.linalg.matrix_rank(info)),
          'information_condition':float(np.linalg.cond(info)),'finite_coefficients':bool(np.isfinite(beta).all()),
          'max_abs_score':float(np.abs(model.score(fit.params)).max()),'max_abs_coefficient':float(np.abs(beta).max()),
          'finite_covariance':bool(np.isfinite(cov).all()),'pearson_dispersion':float(fit.pearson_chi2/fit.df_resid),
          'offset':'log(actual duration_days)','cluster':'subject_id','small_sample_cluster_correction':True,
          'correction_factor':float(correction),'wald_reference':'standard normal','CI_critical_value':1.96,
          'independent_sandwich_max_abs_difference':diff,'warnings':[str(w.message) for w in caught]}
    writejson(out/(name+'_diagnostics.json'),diag)
    assert diag['converged'] and diag['finite_coefficients'] and diag['finite_covariance']
    assert min(diag['design_rank'],diag['covariance_rank'],diag['information_rank'])==p
    assert diag['max_abs_score']<.01 and diag['max_abs_coefficient']<15
    table=pd.DataFrame({'model':name,'term':design.columns,'coefficient':beta,'robust_se':se,
        'rate_ratio':np.exp(beta),'ci_lower':np.exp(beta-1.96*se),'ci_upper':np.exp(beta+1.96*se),
        'p_value':2*norm.sf(np.abs(beta/se)),'intervals':n,'stays':x.stay_id.nunique(),'patients':g,
        'specimens':int(y.sum()),'risk_person_days':x.duration_days.sum()})
    table.to_csv(out/(name+'_coefficients.csv'),index=False)
    pd.DataFrame(cov,index=design.columns,columns=design.columns).to_csv(out/(name+'_covariance.csv'))
    print(json.dumps(diag,ensure_ascii=True),flush=True)
    return design,diag

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--s0-stage',type=Path,required=True)
    ap.add_argument('--raw-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True)
    a=ap.parse_args(); out=a.output_root/'aggregate'; sec=a.output_root/'secure_derived_inputs'
    out.mkdir(exist_ok=True,parents=True);sec.mkdir(exist_ok=True,parents=True)
    sources={'risk':a.s0_stage/'secure_derived_inputs/risk_final_day3.csv.gz',
        'sampling':a.s0_stage/'secure_derived_inputs/sampling_intervals_final_day3.csv.gz',
        'cohort':a.s0_stage/'secure_derived_inputs/landmark_cohort_day3.csv.gz',
        'patients':a.raw_root/'hosp/patients.csv.gz','admissions':a.raw_root/'hosp/admissions.csv.gz'}
    manifest={k:{'path':str(v.resolve()),'bytes':v.stat().st_size,'sha256':sha(v)} for k,v in sources.items()}
    writejson(sec/'period_input_provenance_local.json',manifest)
    writejson(out/'period_input_hashes.json',{k:{'basename':Path(v['path']).name,'bytes':v['bytes'],'sha256':v['sha256']} for k,v in manifest.items()})
    r=pd.read_csv(sources['risk']);x=pd.read_csv(sources['sampling'])
    cohort=pd.read_csv(sources['cohort'],usecols=['subject_id','hadm_id','stay_id'])
    keys=['stay_id','day_start']; assert not r.duplicated(keys).any() and not x.duplicated(keys).any()
    r=r.sort_values(keys).reset_index(drop=True);x=x.sort_values(keys).reset_index(drop=True)
    assert r[keys].equals(x[keys])
    check_cols=['subject_id','hadm_id','t_start','t_stop','duration_days','event','exposure_vulnerability_category',
        'exposure_indication_present','age_per_10y','female','icu_group','admission_group','multiple_line',
        'cvc_type_group','exposure_antibiotic_exposure','exposure_lactate_available','exposure_aki_assessable']
    for c in check_cols: pd.testing.assert_series_equal(r[c],x[c],check_names=False)
    totals={'stays':r.stay_id.nunique(),'patients':r.subject_id.nunique(),'intervals':len(r),
        'events':int(r.event.sum()),'risk_person_days':float(r.duration_days.sum()),
        'specimens':int(x.collection_count.sum()),'occasions':int(x.occasion_count.sum())}
    assert (totals['stays'],totals['patients'],totals['intervals'],totals['events'],totals['specimens'],totals['occasions'])==(13863,12757,60272,268,19160,16086)
    assert abs(totals['risk_person_days']-56370.34172453703)<1e-6
    assert len(cohort)==13863 and cohort.stay_id.is_unique and cohort.hadm_id.is_unique
    assert set(cohort.stay_id)==set(r.stay_id)
    assert r.t_stop.gt(r.t_start).all() and r.duration_days.gt(0).all()
    assert np.allclose(r.t_stop-r.t_start,r.duration_days,rtol=0,atol=1e-12)
    prev=r.groupby('stay_id').t_stop.shift()
    assert np.allclose(r.loc[prev.notna(),'t_start'],prev.dropna(),rtol=0,atol=1e-12)
    assert r.groupby('stay_id').t_start.min().eq(0).all()
    assert r.groupby('stay_id').event.sum().le(1).all()
    assert int(r.groupby('stay_id').tail(1).event.sum())==268
    patients=pd.read_csv(sources['patients'],usecols=['subject_id','anchor_year','anchor_year_group'])
    admissions=pd.read_csv(sources['admissions'],usecols=['subject_id','hadm_id','admittime'],parse_dates=['admittime'])
    period=cohort.merge(admissions,on=['subject_id','hadm_id'],how='left',validate='one_to_one').merge(patients,on='subject_id',how='left',validate='many_to_one')
    bounds=period.anchor_year_group.astype('string').str.extract(r'^(\d{4})\s*-\s*(\d{4})$').astype('Int64')
    period['possible_lower']=bounds[0]+period.admittime.dt.year-period.anchor_year
    period['possible_upper']=bounds[1]+period.admittime.dt.year-period.anchor_year
    complete=period[['admittime','anchor_year','possible_lower','possible_upper']].notna().all(axis=1)
    period['broad_period']='uncertain'
    for label,lo,hi in ERAS:
        period.loc[complete & period.possible_lower.ge(lo) & period.possible_upper.le(hi),'broad_period']=label
    period.loc[~complete,'broad_period']='unavailable'
    period=period.merge(r.groupby('stay_id').agg(event=('event','sum'),risk_intervals=('event','size'),risk_duration_days=('duration_days','sum')).reset_index(),on='stay_id',validate='one_to_one')
    period=period.merge(x.groupby('stay_id').agg(specimens=('collection_count','sum'),occasions=('occasion_count','sum')).reset_index(),on='stay_id',validate='one_to_one')
    period.to_csv(sec/'period_stay_mapping_S0.csv.gz',index=False)
    feasibility=period.groupby('broad_period',sort=True).agg(stays=('stay_id','size'),patients=('subject_id','nunique'),BSI_events=('event','sum'),intervals=('risk_intervals','sum'),risk_person_days=('risk_duration_days','sum'),specimens=('specimens','sum'),occasions=('occasions','sum')).reset_index()
    feasibility.to_csv(out/'period_feasibility.csv',index=False)
    assignable=period.broad_period.isin([v[0] for v in ERAS])
    mapping=period.loc[assignable,['stay_id','broad_period']]
    assert int(assignable.sum())==12064 and int(period.loc[assignable,'event'].sum())==233
    expected={'2008-2013':(5222,117),'2014-2019':(5975,88),'2020-2022':(867,28),'uncertain':(1799,35)}
    for _,row in feasibility.iterrows(): assert (int(row.stays),int(row.BSI_events))==expected[row.broad_period]
    matched=r.merge(mapping,on='stay_id',how='inner',validate='many_to_one')
    sample=x.merge(mapping,on='stay_id',how='inner',validate='many_to_one')
    assert matched[keys].equals(sample[keys])
    for c in check_cols: pd.testing.assert_series_equal(matched[c],sample[c],check_names=False)
    # Each retained stay keeps every original risk row; no event/time redefinition.
    expected_r=r[r.stay_id.isin(mapping.stay_id)].reset_index(drop=True)
    pd.testing.assert_frame_equal(matched.drop(columns='broad_period').reset_index(drop=True),expected_r)
    matched.to_csv(sec/'period_matched_risk_S0.csv.gz',index=False)
    sample.to_csv(sec/'period_matched_sampling_S0.csv.gz',index=False)
    assessment={'status':'eligibility_gate_passed','full_S0':totals,'complete_anchor_fields':int(complete.sum()),
        'classifiable_stays':int(assignable.sum()),'classifiable_events':int(period.loc[assignable,'event'].sum()),
        'uncertain_stays':int((~assignable).sum()),'uncertain_events':int(period.loc[~assignable,'event'].sum()),
        'cross_2013_2014':int((complete & period.possible_lower.le(2013) & period.possible_upper.ge(2014)).sum()),
        'cross_2019_2020':int((complete & period.possible_lower.le(2019) & period.possible_upper.ge(2020)).sum()),
        'outside_2008_2022':int((complete & (period.possible_lower.lt(2008)|period.possible_upper.gt(2022))).sum()),
        'matched_intervals':len(matched),'matched_patients':int(matched.subject_id.nunique()),
        'matched_risk_days':float(matched.duration_days.sum()),'matched_specimens':int(sample.collection_count.sum()),
        'all_selected_original_rows_and_covariates_preserved':True,'patient_level_exports':'secure_derived_inputs only',
        'era_formula':'possible_lower/upper = anchor_year_group lower/upper + year(admittime) - anchor_year',
        'strict_whole_interval_assignment':True,'midpoint_assignment_or_imputation':False,
        'software':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
                    'scipy':scipy.__version__,'patsy':patsy.__version__,'statsmodels':statsmodels.__version__}}
    writejson(out/'period_source_assessment.json',assessment)
    print(json.dumps(assessment,ensure_ascii=True),flush=True)
    base,diag0=fit_poisson(sample,'period_poisson_without_era',PRIMARY,out)
    extra,diag1=fit_poisson(sample,'period_poisson_with_era',PRIMARY+' + '+PERIOD,out)
    assert base.columns.isin(extra.columns).all()
    assert np.array_equal(np.asarray(base),np.asarray(extra[base.columns]))
    writejson(out/'period_poisson_matched_design_check.json',{'same_rows':len(sample),'shared_design_columns':len(base.columns),
        'all_shared_design_values_identical':True,'only_added_columns':[c for c in extra.columns if c not in base.columns]})

if __name__=='__main__':main()

"""Reconstruct frozen Cox output and rerun corrected coding in a new directory."""
from pathlib import Path
import hashlib,importlib.util,json,platform,sys,warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2,norm,spearmanr
import statsmodels
from statsmodels.duration.hazard_regression import PHReg,PHRegResults
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'ARIC_revision_20260920';HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analysis import host_vulnerability_advanced as adv

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def revision_for(output):
    spec=importlib.util.spec_from_file_location('read_only_previous_fit_code',BASE/'code/run_reanalysis.py')
    rev=importlib.util.module_from_spec(spec);spec.loader.exec_module(rev)
    rev.OUT=output;output.mkdir(parents=True,exist_ok=True)
    assert rev.OUT.resolve()!= (BASE/'results').resolve()
    return rev

def code_corrected_stays(stays):
    raw=Path((__import__("os").environ.get("MIMIC_ROOT", "MIMIC_ROOT") + '/hosp/diagnoses_icd.csv.gz'))
    cohort=set(stays.hadm_id);parts=[]
    for z in pd.read_csv(raw,usecols=['subject_id','hadm_id','icd_code','icd_version'],chunksize=500000):
        x=z[z.hadm_id.isin(cohort)]
        if len(x):parts.append(x)
    dx=pd.concat(parts,ignore_index=True)
    code=dx.icd_code.fillna('').astype(str).str.replace('.','',regex=False).str.upper()
    old=((dx.icd_version.eq(10)&code.str.startswith(('C77','C78','C79','C80')))|
         (dx.icd_version.eq(9)&code.str.startswith(('196','197','198','199'))))
    corrected=((dx.icd_version.eq(10)&(code.str.startswith(('C77','C78','C79'))|code.eq('C800')))|
         (dx.icd_version.eq(9)&(code.str.startswith(('196','197','198'))|code.eq('1990'))))
    agg=dx[['subject_id','hadm_id']].copy();agg['old']=old;agg['corrected']=corrected
    aggregate=agg.groupby(['subject_id','hadm_id'],as_index=False).max()
    z=stays[['subject_id','hadm_id','stay_id','metastatic_cancer']].merge(aggregate,on=['subject_id','hadm_id'],how='left')
    z['old']=z.old.fillna(False).astype(bool);z['corrected']=z.corrected.fillna(False).astype(bool)
    assert z.metastatic_cancer.astype(bool).eq(z.old).all()
    bad=set(z.loc[z.old&~z.corrected,'stay_id'])
    assert len(bad)==6
    return bad,sha(raw)

def corrected_risk(d,bad):
    r=d.copy();affected=r.stay_id.isin(bad)
    assert r.loc[affected,'metastatic_cancer'].eq(1).all()
    r.loc[affected,'metastatic_cancer']=0
    r.loc[affected,'exposure_vulnerability_burden_score']-=1
    r['exposure_vulnerability_category']=pd.Categorical(np.select([r.exposure_vulnerability_burden_score.le(1),r.exposure_vulnerability_burden_score.le(3)],['low','moderate'],default='high'),categories=['low','moderate','high'])
    r['vmod']=r.exposure_vulnerability_category.eq('moderate').astype(int)
    r['vhigh']=r.exposure_vulnerability_category.eq('high').astype(int)
    return r

def reconstruction_original(d,rev,output):
    src=BASE/'results';p=src/'primary_day3_covariance.npz';table=pd.read_csv(src/'primary_day3_coefficients.csv')
    with np.load(p,allow_pickle=False) as a:
        beta=a['coefficients'];cov=a['covariance'];terms=a['terms'].tolist()
    data=d.copy();data.t_start=np.nextafter(data.t_start.to_numpy(),np.inf)
    x,design=adv.design_matrix(data,rev.RHS)
    if list(x.columns)!=terms or terms!=table.term.tolist():
        raise AssertionError(f'Saved design mismatch: x={list(x.columns)} npz={terms} csv={table.term.tolist()}')
    model=PHReg(data.t_stop.astype(float),x,status=data.event.astype(int),entry=data.t_start.astype(float),ties='efron')
    assert np.max(np.abs(table.coefficient.to_numpy()-beta))<1e-12
    assert np.max(np.abs(table.robust_se.to_numpy()-np.sqrt(np.diag(cov))))<1e-12
    score_rows=np.asarray(model.score_residuals(beta),dtype=float);valid=np.isfinite(score_rows).all(axis=1)
    group=d.subject_id.to_numpy();unique,inv=np.unique(group[valid],return_inverse=True)
    sums=np.zeros((len(unique),len(beta)));np.add.at(sums,inv,score_rows[valid])
    h=model.hessian(beta);h_inv=np.linalg.inv(h);cov_rebuilt=h_inv@(sums.T@sums)@h_inv
    max_cov_diff=float(np.max(np.abs(cov_rebuilt-cov)));assert max_cov_diff<1e-9
    z=norm.ppf(.975);hr=np.exp(beta);low=np.exp(beta-z*np.sqrt(np.diag(cov)));high=np.exp(beta+z*np.sqrt(np.diag(cov)))
    assert np.max(np.abs(table.hazard_ratio.to_numpy()-hr))<1e-12
    assert np.max(np.abs(table.ci_lower.to_numpy()-low))<1e-12
    assert np.max(np.abs(table.ci_upper.to_numpy()-high))<1e-12
    saved=PHRegResults(model,beta,cov)
    scho=np.asarray(saved.schoenfeld_residuals);event=d.event.eq(1).to_numpy();rows=[];quartiles=[]
    times=d.loc[event,'t_stop'].to_numpy();q=pd.qcut(times,4,labels=False,duplicates='drop')
    for j,term in enumerate(terms):
        v=scho[event,j];good=np.isfinite(v)
        rho=float(spearmanr(times[good],v[good]).statistic) if good.sum()>3 and np.std(v[good])>0 else np.nan
        rows.append({'term':term,'event_residual_rows':int(good.sum()),'Spearman_event_time_residual_correlation_descriptive_only':rho,
                     'residual_mean':float(np.mean(v[good])) if good.sum() else np.nan,'note':'No residual-correlation p-value: risk sets and within-patient events are dependent.'})
        for k in sorted(set(q)):
            slicev=v[q==k]
            quartiles.append({'term':term,'event_time_quartile':int(k)+1,'events':len(slicev),'mean_event_time_days':float(np.mean(times[q==k])),
                              'mean_Schoenfeld_residual':float(np.nanmean(slicev))})
    pd.DataFrame(rows).to_csv(output/'schoenfeld_descriptive_trends_original.csv',index=False)
    pd.DataFrame(quartiles).to_csv(output/'schoenfeld_time_quartiles_original.csv',index=False)
    report={'source_cox_formula':rev.RHS,'ties':'efron','t_start_adjustment':'np.nextafter(t_start,+inf) for left-open interval',
            'patient_cluster_robust_covariance_reconstructed':True,'maximum_absolute_robust_covariance_difference':max_cov_diff,
            'maximum_absolute_score_gradient':float(np.max(np.abs(model.score(beta)))),
            'hessian_rank':int(np.linalg.matrix_rank(h)),'hessian_condition':float(np.linalg.cond(h)),
            'model_intervals':len(d),'model_events':int(d.event.sum()),'model_patients':int(d.subject_id.nunique()),
            'model_ICU_stays':int(d.stay_id.nunique()),'event_residual_rows':int(np.isfinite(scho[event]).all(axis=1).sum()),
            'source_coefficient_npz_sha256':sha(p),'source_coefficient_csv_sha256':sha(src/'primary_day3_coefficients.csv'),
            'reference_risk_sha256':sha(BASE/'secure_derived_inputs/revised_risk_day3.csv.gz')}
    (output/'original_main_cox_reconstruction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def fit_and_report(rev,r,name,formula):
    f,diag=rev.fit(r,name,formula)
    b=np.asarray(f.result.params);v=np.asarray(f.result.cov_params());se=np.sqrt(np.diag(v));z=norm.ppf(.975)
    result=pd.DataFrame({'model':name,'term':f.columns,'coefficient':b,'robust_se_log_hr':se,'hazard_ratio':np.exp(b),
        'ci_lower':np.exp(b-z*se),'ci_upper':np.exp(b+z*se),'p_value':2*norm.sf(np.abs(b/se)),
        'intervals':len(r),'events':int(r.event.sum()),'stays':int(r.stay_id.nunique()),'patients':int(r.subject_id.nunique())})
    result.to_csv(HERE/'corrected_models'/f'{name}_full_precision.csv',index=False)
    return f,diag

def time_interaction(f,name,term_names,timevar):
    idx=[f.columns.index(z) for z in term_names]
    b=np.asarray(f.result.params)[idx];cov=np.asarray(f.result.cov_params())[np.ix_(idx,idx)]
    assert np.linalg.matrix_rank(cov)==len(idx)
    w=float(b@np.linalg.solve(cov,b))
    return {'model':name,'time_transform':timevar,'terms':' | '.join(term_names),'chi_square_robust_Wald':w,'df':len(idx),
            'p_value_exploratory':float(chi2.sf(w,len(idx))),'interpretation':'A null result does not prove proportional hazards at all times.'}

def main():
    output=HERE/'corrected_models';output.mkdir(exist_ok=True)
    rev=revision_for(output)
    d=pd.read_csv(BASE/'secure_derived_inputs/revised_risk_day3.csv.gz')
    d['exposure_vulnerability_category']=pd.Categorical(d.exposure_vulnerability_category,categories=['low','moderate','high'])
    orig=reconstruction_original(d,rev,output)
    stays=d[['subject_id','hadm_id','stay_id','metastatic_cancer']].drop_duplicates('stay_id')
    bad,rawsha=code_corrected_stays(stays)
    r=corrected_risk(d,bad)
    assert len(r)==len(d) and r.event.eq(d.event).all() and r.subject_id.eq(d.subject_id).all()
    changed_category=int(r.exposure_vulnerability_category.astype(str).ne(d.exposure_vulnerability_category).sum());assert changed_category==2
    corrected,diag=fit_and_report(rev,r,'primary_day3_corrected_metastatic',rev.RHS)
    original=pd.read_csv(BASE/'results/primary_day3_coefficients.csv')
    revised=pd.read_csv(output/'primary_day3_corrected_metastatic_full_precision.csv')
    joined=revised.merge(original[['term','coefficient','hazard_ratio','ci_lower','ci_upper']],on='term',suffixes=('_corrected','_historical'),validate='one_to_one')
    joined['hazard_ratio_difference']=joined.hazard_ratio_corrected-joined.hazard_ratio_historical
    joined.to_csv(output/'historical_vs_corrected_primary_full_precision.csv',index=False)
    # Piecewise landmark-day-7 interaction matches the existing diagnostic timescale.
    f7,d7=fit_and_report(rev,r,'primary_day3_corrected_day7_interaction',rev.RHS+' + vmod:late7 + vhigh:late7 + exposure_indication_present:late7')
    log_r=r.copy();log_r['log_t']=np.log1p(log_r.t_start)
    flog,dlog=fit_and_report(rev,log_r,'primary_day3_corrected_logtime_interaction',rev.RHS+' + vmod:log_t + vhigh:log_t + exposure_indication_present:log_t')
    tests=[time_interaction(f7,'primary_day3_corrected_day7_interaction',['vmod:late7','vhigh:late7','exposure_indication_present:late7'],'step at interval start day 7'),
           time_interaction(flog,'primary_day3_corrected_logtime_interaction',['vmod:log_t','vhigh:log_t','exposure_indication_present:log_t'],'log(1 + interval start day)')]
    pd.DataFrame(tests).to_csv(output/'main_exposure_time_interaction_tests.csv',index=False)
    # Conditional log-HR profiles (not absolute risks).
    base={'moderate':"C(exposure_vulnerability_category, Treatment(reference='low'))[T.moderate]",'high':"C(exposure_vulnerability_category, Treatment(reference='low'))[T.high]",'indication':'exposure_indication_present'}
    additional={'moderate':'vmod:log_t','high':'vhigh:log_t','indication':'exposure_indication_present:log_t'}
    rows=[];coef=np.asarray(flog.result.params);cov=np.asarray(flog.result.cov_params());z=norm.ppf(.975)
    for label in base:
        for day in [0,7,14,21,28]:
            contrast=np.zeros(len(coef));contrast[flog.columns.index(base[label])]=1;contrast[flog.columns.index(additional[label])]=np.log1p(day)
            estimate=float(contrast@coef);se=float(np.sqrt(contrast@cov@contrast))
            rows.append({'contrast':label,'interval_start_day':day,'hazard_ratio':float(np.exp(estimate)),'ci_lower':float(np.exp(estimate-z*se)),
                'ci_upper':float(np.exp(estimate+z*se)),'time_interaction_model':'piecewise constant covariates on daily risk intervals; exploratory'})
    pd.DataFrame(rows).to_csv(output/'time_varying_contrast_profiles.csv',index=False)
    # Exact-48-hour primary sensitivity is refit only if its score category actually changes.
    exact=pd.read_csv(BASE/'secure_derived_inputs/revised_risk_exact48h.csv.gz')
    exact_corrected=corrected_risk(exact,bad)
    exact_changed=int(exact_corrected.exposure_vulnerability_category.astype(str).ne(exact.exposure_vulnerability_category).sum())
    if exact_changed:
        rhs=rev.RHS.replace(' + exposure_aki_assessable','')
        _,de=fit_and_report(rev,exact_corrected,'primary_exact48h_corrected_metastatic',rhs)
    else:de=None
    # Component Cox uses chronic-flag count, even if category remains unchanged.
    component_r=r.copy()
    chronic=['chf','copd','ckd','diabetes','cancer','metastatic_cancer','liver_disease','cerebrovascular_disease','peripheral_vascular_disease','immunosuppression']
    component_r['documented_chronic_count']=component_r[chronic].fillna(0).sum(axis=1)
    component_r['exposure_creatinine_rise_documented']=component_r.lag_aki.eq(1).astype(int)
    component_r['exposure_lactate_high_documented']=component_r.lag_lactate_high.eq(1).astype(int)
    formula='documented_chronic_count + exposure_mechanical_ventilation + exposure_creatinine_rise_documented + exposure_lactate_high_documented + exposure_indication_present + '+rev.ADJUST
    _,dc=fit_and_report(rev,component_r,'bsi_components_corrected_metastatic',formula)
    # Never modify the original risk or model-output directories.
    manifest={'error_definition':'C80.1/199.1 (primary/unknown) and C80.2/199.2 (transplant-related) are not metastatic; C80.0/199.0 disseminated remain positive',
      'bad_stays':len(bad),'affected_day3_intervals':int(r.stay_id.isin(bad).sum()),'changed_day3_category_intervals':changed_category,
      'changed_exact48h_category_intervals':exact_changed,'original_main_reconstruction':orig,
      'corrected_main_counts':{'intervals':len(r),'events':int(r.event.sum()),'patients':int(r.subject_id.nunique()),'stays':int(r.stay_id.nunique())},
      'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},
      'source_sha256':{'risk_day3':sha(BASE/'secure_derived_inputs/revised_risk_day3.csv.gz'),'risk_exact48h':sha(BASE/'secure_derived_inputs/revised_risk_exact48h.csv.gz'),
                       'raw_diagnosis':rawsha,'fit_source':sha(BASE/'code/run_reanalysis.py'),'cox_helper_source':sha(ROOT/'src/analysis/host_vulnerability_advanced.py'),
                       'scoring_source':sha(ROOT/'src/analysis/host_vulnerability_indication.py')},
      'model_formula':{'primary_day3':rev.RHS,'primary_exact48h':rev.RHS.replace(' + exposure_aki_assessable',''),'components':formula},
      'original_outputs_overwritten':False,'patient_rows_exported':False,
      'diagnostics':{'main_corrected':diag,'day7_interaction':d7,'logtime_interaction':dlog,'exact48h_corrected':de,'component_corrected':dc}}
    (output/'reanalysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'day3_changed_category_intervals':changed_category,'exact48h_changed_category_intervals':exact_changed,
                      'original_main_robust_covariance_max_difference':orig['maximum_absolute_robust_covariance_difference'],
                      'time_interactions':tests,'outputs':str(output)},indent=2),flush=True)

if __name__=='__main__':main()

"""Rerun all score-dependent association and sampling models; export aggregate only."""
from pathlib import Path
import importlib.util,json,platform,sys
import numpy as np
import pandas as pd
from scipy.stats import chi2
import scipy,statsmodels
from coding_rules_final import RAW_DIAGNOSES,stay_flag_deltas,correct_frame
from run_primary_cox_audit import revision_for,reconstruction_original,sha

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'ARIC_revision_20260920'
HERE=Path(__file__).resolve().parents[1]
OUT=HERE/'corrected_models_final'
ROUTE=BASE/'route_A'

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def fit(rev,r,name,rhs):
    f,diag=rev.fit(r,name,rhs)
    diag['formula']=rhs
    return f,diag

def main():
    OUT.mkdir(exist_ok=True)
    rev=revision_for(OUT)
    original=pd.read_csv(BASE/'secure_derived_inputs/revised_risk_day3.csv.gz')
    original['exposure_vulnerability_category']=pd.Categorical(original.exposure_vulnerability_category,categories=['low','moderate','high'])
    reconstruction_original(original,rev,OUT)
    stays=original[['subject_id','hadm_id','stay_id','cancer','metastatic_cancer']].drop_duplicates('stay_id')
    delta=stay_flag_deltas(stays)
    r=correct_frame(original,delta)
    assert len(r)==len(original) and r.event.eq(original.event).all()
    old_category=original.exposure_vulnerability_category.astype(str)
    shift=r.exposure_vulnerability_category.astype(str).ne(old_category)
    aggregate={'stays_new_cancer_flag':int(delta.delta_cancer.gt(0).sum()),
        'stays_removed_metastatic_flag':int(delta.delta_meta.lt(0).sum()),
        'stays_new_metastatic_flag':int(delta.delta_meta.gt(0).sum()),
        'stays_with_any_flag_change':int((delta.delta_cancer.ne(0)|delta.delta_meta.ne(0)).sum()),
        'stays_with_score_change':int(delta.delta_score.ne(0).sum()),
        'risk_intervals_with_any_score_change':int(r.stay_id.isin(delta.loc[delta.delta_score.ne(0),'stay_id']).sum()),
        'risk_intervals_changing_category':int(shift.sum()),
        'stays_changing_category':int(r.loc[shift,'stay_id'].nunique()),
        'events_in_category_changed_intervals':int(r.loc[shift,'event'].sum()),
        'cancer_and_metastatic_overlap_stays_final':int(((delta.new_cancer==1)&(delta.new_meta==1)).sum()),
        'reference_intervals':len(r),'reference_events':int(r.event.sum()),'reference_patients':int(r.subject_id.nunique()),'reference_stays':int(r.stay_id.nunique())}
    delta_summary=delta[['delta_cancer','delta_meta','delta_score']].value_counts().rename('stays').reset_index()
    delta_summary.to_csv(OUT/'score_flag_transition_aggregate.csv',index=False)
    cats=pd.DataFrame({'historical_category':old_category,'corrected_category':r.exposure_vulnerability_category.astype(str),'event':r.event})
    cats.groupby(['historical_category','corrected_category'],as_index=False).agg(intervals=('event','size'),events=('event','sum')).to_csv(OUT/'score_category_transition_aggregate.csv',index=False)
    (OUT/'scoring_correction_summary.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
    diags=[]
    f,d=fit(rev,r,'primary_day3_final',rev.RHS);diags.append(d)
    rev.write(rev.contrasts(f,r),'primary_day3_final_contrasts')
    joint,d=fit(rev,r,'joint_states_final',f'{rev.VULN} * exposure_indication_present + {rev.ADJUST}');diags.append(d)
    states,within=rev.adv.interaction_contrasts(joint,r)
    rev.write(states.rename(columns={'catheter_days':'risk_intervals'}),'six_state_adjusted_final')
    rev.write(within.rename(columns={'catheter_days':'risk_intervals'}),'indication_within_vulnerability_final')
    rev.write(rev.adv.global_interaction_test(joint),'joint_interaction_test_final')
    rev.write(rev.baseline_table(r),'baseline_characteristics_final')
    rev.write(rev.rates(r,['exposure_vulnerability_category']),'vulnerability_rates_final')
    rev.write(rev.rates(r,['exposure_indication_present']),'indication_rates_final')
    rev.write(rev.rates(r,['exposure_vulnerability_category','exposure_indication_present']),'six_state_rates_final')
    for name,data,rhs in [
        ('record_frequency_indication_final',r.assign(exposure_indication_present=r.lag_original_indication_with_frequency.astype(int)),rev.RHS),
        ('without_creatinine_marker_final',r.assign(exposure_vulnerability_category=rev.category(r.lag_corrected_score-r.lag_aki.eq(1))),rev.RHS),
        ('nonoverlapping_cancer_final',r.assign(exposure_vulnerability_category=rev.category(r.lag_corrected_score-(r.cancer.eq(1)&r.metastatic_cancer.eq(1)))),rev.RHS),
        ('both_labs_observed_final',r[r.exposure_lactate_available.eq(1)&r.exposure_aki_assessable.eq(1)].assign(icu_group=lambda x:x.icu_group.replace({'neuro':'surgical_trauma'})),rev.RHS.replace(' + exposure_lactate_available + exposure_aki_assessable','')),
        ('matched_48h_adjustment_day3_final',r,rev.RHS.replace(' + exposure_aki_assessable','')),
        ('component_adjusted_final',r,rev.RHS+' + exposure_mechanical_ventilation + exposure_vasopressor + exposure_renal_replacement_therapy'),
        ('recognized_pathogen_cause_specific_final',r.assign(event=r.event*r.recognized_pathogen.fillna(0).astype(int)),rev.RHS),
    ]:
        ff,dd=fit(rev,data,name,rhs);diags.append(dd);rev.write(rev.contrasts(ff,data),name+'_contrasts')
    # Rebuild the active-episode risk set; the event/risk windows remain exactly the frozen ones.
    daily,lm,events=rev.load('day3')
    active,_,_=rev.build_risk(daily,lm,events,True)
    active=correct_frame(active,delta)
    ff,dd=fit(rev,active,'active_episode_only_final',rev.RHS);diags.append(dd)
    rev.write(rev.contrasts(ff,active),'active_episode_only_final_contrasts')
    exact=pd.read_csv(BASE/'secure_derived_inputs/revised_risk_exact48h.csv.gz')
    exact=correct_frame(exact,delta)
    ff,dd=fit(rev,exact,'primary_exact48h_final',rev.RHS.replace(' + exposure_aki_assessable',''));diags.append(dd)
    rev.write(rev.contrasts(ff,exact),'primary_exact48h_final_contrasts')
    # All original score-related time diagnostics use the corrected risk set.
    day7,dd=fit(rev,r,'day7_time_interactions_final',rev.RHS+' + vmod:late7 + vhigh:late7 + exposure_indication_present:late7');diags.append(dd)
    late_terms=['vmod:late7','vhigh:late7','exposure_indication_present:late7']
    rows=[]
    def wald(model,terms,block,transform):
        idx=[model.columns.index(z) for z in terms];b=np.asarray(model.result.params)[idx];v=np.asarray(model.result.cov_params())[np.ix_(idx,idx)]
        w=float(b@np.linalg.solve(v,b))
        return {'block':block,'time_transform':transform,'df':len(idx),'chi_square_cluster_robust_Wald':w,'p_value_exploratory':float(chi2.sf(w,len(idx))),
                'terms':' | '.join(terms)}
    rows.append(wald(day7,late_terms,'main_exposures','day 7 step'))
    log_r=r.copy();log_r['log_t']=np.log1p(log_r.t_start)
    logfit,dd=fit(rev,log_r,'logtime_main_interactions_final',rev.RHS+' + vmod:log_t + vhigh:log_t + exposure_indication_present:log_t');diags.append(dd)
    rows.append(wald(logfit,['vmod:log_t','vhigh:log_t','exposure_indication_present:log_t'],'main_exposures','log(1 + interval start day)'))
    # Blockwise PH checks avoid a nonestimable 36-parameter full-time model.
    blocks={
        'ICU_group':[x for x in f.columns if x.startswith('C(icu_group)')],
        'admission_group':[x for x in f.columns if x.startswith('C(admission_group)')],
        'CVC_type':[x for x in f.columns if x.startswith('C(cvc_type_group)')],
        'continuous_and_other_binary':[x for x in f.columns if x not in ['exposure_indication_present'] and not x.startswith(('C(exposure_vulnerability_category','C(icu_group)','C(admission_group)','C(cvc_type_group)'))],
    }
    for label,terms in blocks.items():
        cols=[]
        for k,term in enumerate(terms):
            col='ph_'+label+'_'+str(k)
            # Construct the same design column independently of score changing over time.
            design,_=rev.adv.design_matrix(log_r,rev.RHS)
            log_r[col]=design[term].to_numpy()*log_r.log_t.to_numpy()
            cols.append(col)
        try:
            mf,md=fit(rev,log_r,'ph_'+label+'_final',rev.RHS+' + '+' + '.join(cols))
            diags.append(md);rows.append(wald(mf,cols,label,'log(1 + interval start day)'))
        except Exception as exc:
            rows.append({'block':label,'time_transform':'log(1 + interval start day)','df':len(cols),'status':'not estimable','diagnostic_error':str(exc)[:500]})
    pd.DataFrame(rows).to_csv(OUT/'proportional_hazards_time_interaction_tests_final.csv',index=False)
    # Route A BSI component and lag models, using locked alignment and fixed outcome.
    lag=module(ROUTE/'code/run_lag_and_component_bsi.py','lag_readonly')
    lag.OUT=OUT;lag.revision_module=lambda: rev
    component_formula=('documented_chronic_count + exposure_mechanical_ventilation + exposure_creatinine_rise_documented + exposure_lactate_high_documented + exposure_indication_present + '+rev.ADJUST)
    ref=pd.read_csv(ROUTE/'secure_derived_inputs/bsi_reference_with_components.csv.gz')
    ref=correct_frame(ref,delta)
    ff,dd=fit(rev,ref,'bsi_components_final',component_formula);diags.append(dd)
    lag_rhs=rev.RHS.replace(' + exposure_aki_assessable','')
    for hours in (24,48):
        additional=pd.read_csv(ROUTE/f'secure_derived_inputs/bsi_additional_{hours}h_risk.csv.gz')
        additional=correct_frame(additional,delta)
        matched=ref[ref.analysis_interval_id.isin(additional.analysis_interval_id)].copy()
        assert len(matched)==len(additional) and matched.event.sum()==additional.event.sum()
        for label,data in [('matched_reference',matched),('additional_lag',additional)]:
            ff,dd=fit(rev,data,f'bsi_lag{hours}_{label}_final',lag_rhs);diags.append(dd)
            rev.write(rev.contrasts(ff,data),f'bsi_lag{hours}_{label}_final_contrasts')
    # Sampling outcomes and intervals are retained; only score-dependent predictors change.
    sampling=module(ROUTE/'code/run_sampling_analysis.py','sampling_readonly');sampling.OUT=OUT
    sample=pd.read_csv(ROUTE/'secure_derived_inputs/sampling_intervals_day3.csv.gz')
    sample=correct_frame(sample,delta)
    assert sample.documented_chronic_count.add(sample.prior_ventilation).add(sample.prior_creatinine_rise).add(sample.prior_high_lactate).eq(sample.lag_corrected_score).all()
    for groups,name in [(['exposure_vulnerability_category'],'sampling_by_vulnerability_final.csv'),(['exposure_indication_present'],'sampling_by_therapy_final.csv'),(['exposure_vulnerability_category','exposure_indication_present'],'sampling_by_six_states_final.csv')]:
        sampling.write_csv(sampling.aggregate_sampling(sample,groups),name)
    sampling_diags=[]
    for name,outcome,rhs in [('sampling_primary_specimens_final','collection_count',sampling.PRIMARY),('sampling_components_specimens_final','collection_count',sampling.COMPONENT),('sampling_primary_occasions_final','occasion_count',sampling.PRIMARY),('sampling_components_occasions_final','occasion_count',sampling.COMPONENT)]:
        _,sd=sampling.fit_poisson(sample,name,outcome,rhs);sd['formula']=rhs;sampling_diags.append(sd)
    pd.DataFrame(diags).to_csv(OUT/'all_corrected_cox_model_diagnostics.csv',index=False)
    pd.DataFrame(sampling_diags).to_csv(OUT/'all_corrected_sampling_model_diagnostics.csv',index=False)
    manifest={'coding_rule':'CODING_AMENDMENT_20261004.md','main_formula':rev.RHS,'main_model_ties':'efron',
        'patient_cluster_robust_variance':True,'reference_cohort_unchanged':True,'score_transition':aggregate,
        'source_hashes':{'raw_diagnoses':sha(RAW_DIAGNOSES),'original_risk':sha(BASE/'secure_derived_inputs/revised_risk_day3.csv.gz'),
                         'original_analysis_code':sha(BASE/'code/run_reanalysis.py'),'final_coding_code':sha(Path(__file__).parent/'coding_rules_final.py'),
                         'final_rerun_code':sha(Path(__file__))},
        'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},
        'patient_rows_exported_publicly':False,'models_cox_fitted':len(diags),'sampling_models_fitted':len(sampling_diags),
        'historical_files_modified':False,'full_36_parameter_time_interaction_model':'not estimable; blockwise checks reported'}
    (OUT/'final_coding_reanalysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'aggregate':aggregate,'cox_fits':len(diags),'sampling_fits':len(sampling_diags),'ph_tests':rows},indent=2),flush=True)

if __name__=='__main__':main()

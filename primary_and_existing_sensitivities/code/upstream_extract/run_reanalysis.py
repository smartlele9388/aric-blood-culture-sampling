"""Locked post-review MIMIC analysis; sources remain read-only, exports aggregate."""
from __future__ import annotations
from pathlib import Path
import hashlib, json, sys, platform, warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parent
sys.path.insert(0,str(ROOT))
from src.analysis import host_vulnerability_advanced as adv
from src.analysis.host_vulnerability_indication import CHRONIC_CONDITIONS, assert_aggregate_export
SEC=BASE/'secure_derived_inputs'; OUT=BASE/'results'; OUT.mkdir(exist_ok=True)
ORIGINAL=ROOT/'secure_derived_inputs/host_vulnerability_indication_day3_landmark'
VULN="C(exposure_vulnerability_category, Treatment(reference='low'))"
ADJUST='age_per_10y + female + C(icu_group) + C(admission_group) + multiple_line + C(cvc_type_group) + exposure_antibiotic_exposure + exposure_lactate_available + exposure_aki_assessable'
RHS=f'{VULN} + exposure_indication_present + {ADJUST}'

def write(d,name):
    assert_aggregate_export(d)
    d.to_csv(OUT/f'{name}.csv',index=False)

def load(scenario):
    d=pd.read_csv(ORIGINAL/f'patient_day_states_{scenario}.csv.gz',parse_dates=['day_start','day_end','landmark_time','followup_end'])
    f=pd.read_csv(SEC/f'corrected_features_{scenario}.csv.gz')
    for c in f.columns:
        if c not in ['stay_id','local_day'] and c in d: d.drop(columns=c,inplace=True)
    d=d.merge(f,on=['stay_id','local_day'],validate='one_to_one')
    treatment=pd.read_csv(SEC/f'corrected_treatments_{scenario}.csv.gz')
    d=d.drop(columns=[c for c in treatment if c not in ['stay_id','local_day']]).merge(treatment,on=['stay_id','local_day'],validate='one_to_one')
    # Persist NA lab features, and separately count only documented positive markers.
    d['corrected_chronic_score']=d[CHRONIC_CONDITIONS].fillna(0).sum(axis=1)
    d['corrected_score']=d.corrected_chronic_score+d.mechanical_ventilation+d.aki.eq(1)+d.lactate_high.eq(1)
    d['corrected_indication']=d[['vasopressor','renal_replacement_therapy','tpn']].max(axis=1)
    d['original_indication_with_frequency']=d[['vasopressor','renal_replacement_therapy','tpn','high_frequency_iv_meds']].max(axis=1)
    d=d.sort_values(['stay_id','local_day']).reset_index(drop=True)
    lag=['corrected_score','corrected_indication','original_indication_with_frequency','mechanical_ventilation','vasopressor','renal_replacement_therapy','antibiotic_exposure','lactate_available','aki_assessable','aki','lactate_high','number_of_concurrent_lines']
    for c in lag: d['lag_'+c]=d.groupby('stay_id')[c].shift()
    lm=pd.read_csv(ORIGINAL/f'landmark_cohort_{scenario}.csv.gz',parse_dates=['landmark_time','index_interval_start','index_interval_end','followup_end','deathtime','outtime','dischtime'])
    ev=pd.read_csv(ORIGINAL/f'clabsi_events_{scenario}.csv.gz',parse_dates=['event_time'])
    return d,lm,ev

def category(score):
    if pd.isna(score).any():
        raise ValueError('Missing vulnerability score cannot be assigned to a category')
    return pd.Categorical(np.where(score<=1,'low',np.where(score<=3,'moderate','high')),categories=['low','moderate','high'])

def build_risk(d,lm,ev,active_only=False):
    lm=lm.copy()
    lm['line_window_end']=lm.index_interval_end if active_only else (lm.index_interval_end-pd.Timedelta(nanoseconds=1)).dt.floor('D')+pd.Timedelta(days=2)
    lm['risk_end']=pd.concat([lm.line_window_end,lm.followup_end],axis=1).min(axis=1)
    e=ev[['stay_id','event_time','recognized_pathogen','amr_bsi_r','amr_bsi_ri']]
    lm=lm.merge(e,on='stay_id',how='left',validate='one_to_one')
    lm['event_eligible']=lm.event_time.notna()&(lm.event_time>=lm.landmark_time)&(lm.event_time<=lm.risk_end)&(lm.event_time<lm.line_window_end)
    lm['observed_end']=lm.risk_end.where(~lm.event_eligible,lm.event_time)
    # Exact boundary events at time zero cannot contribute a positive Cox interval.
    zero=lm.event_eligible&lm.observed_end.eq(lm.landmark_time)
    lm.loc[zero,'observed_end']=lm.loc[zero,'landmark_time']+pd.Timedelta(seconds=1)
    r=d[d.local_day.gt(0)].merge(lm[['stay_id','observed_end','risk_end','index_interval_end','event_time','event_eligible','recognized_pathogen','amr_bsi_r','amr_bsi_ri']],on='stay_id',validate='many_to_one')
    r=r[r.day_start<r.observed_end].copy()
    r['stop_datetime']=pd.concat([r.day_end,r.observed_end],axis=1).min(axis=1)
    r['t_start']=(r.day_start-r.landmark_time).dt.total_seconds()/86400
    r['t_stop']=(r.stop_datetime-r.landmark_time).dt.total_seconds()/86400
    r['duration_days']=r.t_stop-r.t_start
    r=r[r.duration_days>0].copy()
    r['event']=(r.event_eligible&r.stop_datetime.eq(r.observed_end)).astype(int)
    r['active_duration_days']=(pd.concat([r.stop_datetime,r.index_interval_end],axis=1).min(axis=1)-r.day_start).dt.total_seconds().clip(lower=0)/86400
    r['age_per_10y']=(r.age-60)/10
    r['multiple_line']=r.lag_number_of_concurrent_lines.gt(1).astype(int)
    r['icu_group']=r.icu_type.map(adv.collapse_icu_type)
    r['admission_group']=r.admission_type.map(adv.collapse_admission_type)
    # Collapse baseline device type without consulting future intervals.
    baseline=d[d.local_day.eq(0)].set_index('stay_id').cvc_type_group
    r['cvc_type_group']=r.stay_id.map(baseline).fillna('unknown')
    r['exposure_vulnerability_burden_score']=r.lag_corrected_score
    r['exposure_vulnerability_category']=category(r.lag_corrected_score)
    r['exposure_indication_present']=r.lag_corrected_indication.astype(int)
    for c in ['mechanical_ventilation','vasopressor','renal_replacement_therapy','antibiotic_exposure','lactate_available','aki_assessable']:
        r['exposure_'+c]=r['lag_'+c].astype(int)
    r['log_catheter_day']=np.log1p(r.local_day+2) # Only needed by legacy contrast utility; never a predictor.
    r['late7']=(r.t_start>=7).astype(int)
    r['vmod']=r.exposure_vulnerability_category.eq('moderate').astype(int)
    r['vhigh']=r.exposure_vulnerability_category.eq('high').astype(int)
    assert r.event.sum()==lm.event_eligible.sum()
    assert r.groupby('stay_id').event.sum().max()<=1
    assert (r.t_stop>r.t_start).all()
    # Continuous episodes have no omitted gaps or overlapping intervals.
    gap=r.groupby('stay_id').t_start.shift(-1)-r.t_stop
    assert np.allclose(gap.dropna(),0)
    return r.reset_index(drop=True),lm,int(zero.sum())

def fit(r,name,rhs=RHS):
    print('Fitting',name,'intervals',len(r),'events',int(r.event.sum()),flush=True)
    data=r.copy()
    data['t_start']=np.nextafter(data.t_start.to_numpy(),np.inf)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        f=adv.fit_counting_process_cox(data,rhs,name,'preceding complete observation window; actual event time')
    h=-f.result.model.hessian(f.result.params)
    cov=f.result.cov_params()
    diag={'model':name,'intervals':len(r),'events':int(r.event.sum()),'stays':int(r.stay_id.nunique()),'patients':int(r.subject_id.nunique()),'parameters':len(f.columns),'hessian_rank':int(np.linalg.matrix_rank(h)),'hessian_condition':float(np.linalg.cond(h)),'max_abs_gradient':float(np.max(np.abs(f.result.model.score(f.result.params)))),'finite_covariance':bool(np.isfinite(cov).all()),'max_abs_coefficient':float(np.max(np.abs(f.result.params))),'max_robust_se':float(np.sqrt(np.diag(cov)).max()),'risk_person_days':float(r.duration_days.sum()),'warnings':' | '.join(str(w.message) for w in caught)}
    if (not np.isfinite(cov).all() or diag['hessian_rank']!=len(f.columns)
        or diag['max_abs_gradient']>0.01 or diag['hessian_condition']>1e8
        or diag['max_abs_coefficient']>10 or 'failed to converge' in diag['warnings']):
        raise RuntimeError(f'Model diagnostic failure: {diag}')
    write(adv.coefficient_table(f).rename(columns={'catheter_days':'risk_intervals'}),name+'_coefficients')
    np.savez_compressed(OUT/f'{name}_covariance.npz',covariance=cov,coefficients=f.result.params,terms=np.array(f.columns))
    return f,diag

def contrasts(f,r):
    return adv.adjusted_main_contrasts(f,r).rename(columns={'catheter_days':'risk_intervals'})

def rates(r,group):
    rows=[]
    for key,g in r.groupby(group,observed=True):
        if not isinstance(key,tuple):key=(key,)
        n=int(g.event.sum()); t=float(g.duration_days.sum())
        low=0 if n==0 else chi2.ppf(.025,2*n)/2/t*1000
        high=chi2.ppf(.975,2*(n+1))/2/t*1000
        rows.append(dict(zip(group,key))|{'risk_intervals':len(g),'events':n,'risk_person_days':t,'active_cvc_person_days':float(g.active_duration_days.sum()),'rate_per_1000_person_days':n/t*1000,'ci_lower':low,'ci_upper':high})
    return pd.DataFrame(rows)

def baseline_table(r):
    first=r.sort_values(['stay_id','local_day']).drop_duplicates('stay_id')
    rows=[]
    for label,g in [('Overall',first)]+[(str(k),g) for k,g in first.groupby('exposure_vulnerability_category',observed=True)]:
        rows.append({'group':label,'ICU_stays':len(g),'patients':g.subject_id.nunique(),'age_median':g.age.median(),'age_q1':g.age.quantile(.25),'age_q3':g.age.quantile(.75),'female_n':int(g.female.sum()),'female_percent':g.female.mean()*100,'baseline_indication_n':int(g.exposure_indication_present.sum()),'baseline_indication_percent':g.exposure_indication_present.mean()*100,'baseline_ventilation_n':int(g.exposure_mechanical_ventilation.sum()),'baseline_lactate_measured_n':int(g.exposure_lactate_available.sum()),'baseline_creatinine_assessable_n':int(g.exposure_aki_assessable.sum())})
    return pd.DataFrame(rows)

def competing(r,lm):
    first=r.sort_values(['stay_id','local_day']).drop_duplicates('stay_id').set_index('stay_id')
    t=lm[['subject_id','stay_id','landmark_time','observed_end','risk_end','line_window_end','deathtime','outtime','dischtime','event_eligible']].copy()
    t['time_days']=(t.observed_end-t.landmark_time).dt.total_seconds()/86400
    t['cause']=0
    # Event priority at ties is BSI > death > discharge > attribution-window end.
    t.loc[t.risk_end.eq(t.line_window_end),'cause']=4
    t.loc[t.risk_end.eq(t.outtime)|t.risk_end.eq(t.dischtime),'cause']=3
    t.loc[t.risk_end.eq(t.deathtime),'cause']=2
    t.loc[t.event_eligible,'cause']=1
    t['baseline_group']=t.stay_id.map(first.exposure_vulnerability_category).astype(str)
    t.to_csv(SEC/'competing_events_day3.csv.gz',index=False)
    grid=np.arange(0,29)
    rows=[]; rng=np.random.default_rng(20260920)
    for group,g in [('Overall',t)]+[(str(k),g) for k,g in t.groupby('baseline_group')]:
        patients,idx=np.unique(g.subject_id,return_inverse=True)
        # All nonadministrative stops are explicit competing events: AJ equals observed first-event proportions.
        den=np.bincount(idx,minlength=len(patients))
        weights=rng.multinomial(len(patients),np.repeat(1/len(patients),len(patients)),size=300)
        bootden=weights@den
        for cause,label in [(1,'Proxy BSI'),(2,'Death'),(3,'ICU or hospital discharge'),(4,'End of line-attribution window')]:
            values=np.zeros((len(patients),len(grid)))
            for j,day in enumerate(grid):
                values[:,j]=np.bincount(idx,weights=((g.cause==cause)&(g.time_days<=day)).astype(int),minlength=len(patients))
            point=values.sum(axis=0)/len(g)
            boot=(weights@values)/bootden[:,None]
            low,high=np.quantile(boot,[.025,.975],axis=0)
            rows += [{'group':group,'cause':label,'day':int(day),'cumulative_incidence':float(v),'ci_lower':float(lo),'ci_upper':float(hi),'stays':len(g),'patient_bootstrap_replicates':300} for day,v,lo,hi in zip(grid,point,low,high)]
    write(pd.DataFrame(rows),'competing_risk_curves')
    summary=t.groupby('cause').size().rename('stays').reset_index()
    summary['percent']=summary.stays/len(t)*100
    write(summary,'terminal_event_counts')

def culture_process(r):
    c=pd.read_csv(SEC/'culture_collections.csv.gz',usecols=['hadm_id','event_time','positive'],parse_dates=['event_time'])
    counts=np.zeros(len(r),int); positives=np.zeros(len(r),int)
    byhadm={k:g for k,g in c.groupby('hadm_id',sort=False)}
    for _,g in r.groupby('stay_id',sort=False):
        e=byhadm.get(g.hadm_id.iloc[0])
        if e is None:continue
        a=g.day_start.astype('int64').to_numpy(); b=g.stop_datetime.astype('int64').to_numpy()
        times=e.event_time.astype('int64').to_numpy()
        pos=np.searchsorted(a,times,side='right')-1
        valid=pos>=0
        valid[valid]=times[valid]<=b[pos[valid]]
        counts[g.index]=np.bincount(pos[valid],minlength=len(g))
        positives[g.index]=np.bincount(pos[valid],weights=e.positive.to_numpy()[valid],minlength=len(g)).astype(int)
    v=r.copy();v['culture_count']=counts;v['positive_count']=positives
    out=v.groupby(['exposure_vulnerability_category','exposure_indication_present'],observed=True).agg(risk_person_days=('duration_days','sum'),collections=('culture_count','sum'),positive_collections=('positive_count','sum'),events=('event','sum')).reset_index()
    out['collections_per_1000_person_days']=out.collections/out.risk_person_days*1000
    write(out,'culture_process')

def event_audit(r,lm,scenario,original_states,ev):
    allb=pd.read_csv(SEC/f'first_all_bsi_{scenario}.csv.gz',parse_dates=['event_time'])
    legacy=adv.prepare_counting_process_data(original_states)
    legacyall=adv.prepare_counting_process_data(original_states,outcome_col='outcome_all_hospital_onset_bsi')
    rows=[{'set':'Full ICU follow-up first qualifying BSI','events':len(allb)},
          {'set':'Full ICU follow-up first line-timed proxy BSI','events':len(ev)},
          {'set':'Legacy active-row all-BSI model','events':int(legacyall.event.sum())},
          {'set':'Legacy proxy model with remapped events','events':int(legacy.event.sum())},
          {'set':'Revised first-episode attribution-window proxy BSI','events':int(r.event.sum())},
          {'set':'Proxy events excluded after first episode attribution window','events':int(len(ev)-r.event.sum())}]
    write(pd.DataFrame(rows),f'event_set_reconciliation_{scenario}')
    audit=ev[['stay_id','event_time']].merge(lm[['stay_id','risk_end','event_eligible']],on='stay_id',how='left')
    audit['reason']=np.where(audit.event_eligible,'included_actual_recorded_time','after_first_episode_followup')
    audit.to_csv(SEC/f'event_audit_{scenario}.csv.gz',index=False)

def main():
    summary={}; diagnostics=[]; main_effects=[]; all_r={}
    for scenario in ['day3','exact48h']:
        d,lm,ev=load(scenario)
        r,end,zero=build_risk(d,lm,ev)
        r.to_csv(SEC/f'revised_risk_{scenario}.csv.gz',index=False)
        scenario_rhs=RHS if scenario=='day3' else RHS.replace(' + exposure_aki_assessable','')
        f,diag=fit(r,'primary_'+scenario,scenario_rhs);diagnostics.append(diag)
        main_effects.append(contrasts(f,r))
        all_r[scenario]=(d,lm,ev,r,end,f)
        summary[scenario]={'stays':int(r.stay_id.nunique()),'patients':int(r.subject_id.nunique()),'intervals':len(r),'events':int(r.event.sum()),'risk_person_days':float(r.duration_days.sum()),'active_cvc_person_days':float(r.active_duration_days.sum()),'zero_time_event_one_second_adjustment':zero,'lactate_missing_intervals':int(r.exposure_lactate_available.eq(0).sum()),'creatinine_unassessable_intervals':int(r.exposure_aki_assessable.eq(0).sum()),'age_min':float(r.age.min()),'age_max':float(r.age.max())}
        original=pd.read_csv(ORIGINAL/f'patient_day_states_{scenario}.csv.gz')
        event_audit(r,end,scenario,original,ev)
        if scenario=='day3':
            write(baseline_table(r),'baseline_characteristics')
            write(rates(r,['exposure_vulnerability_category']),'vulnerability_rates')
            write(rates(r,['exposure_indication_present']),'indication_rates')
            write(rates(r,['exposure_vulnerability_category','exposure_indication_present']),'six_state_rates')
            fi,di=fit(r,'joint_states',f'{VULN} * exposure_indication_present + {ADJUST}');diagnostics.append(di)
            states,within=adv.interaction_contrasts(fi,r)
            write(states.rename(columns={'catheter_days':'risk_intervals'}),'six_state_adjusted')
            write(within.rename(columns={'catheter_days':'risk_intervals'}),'indication_within_vulnerability')
            write(adv.global_interaction_test(fi).rename(columns={'catheter_days':'risk_intervals'}),'interaction_test')
            for name,data,rhs in [
                ('active_episode_only',build_risk(d,lm,ev,True)[0],RHS),
                ('record_frequency_indication',r.assign(exposure_indication_present=r.lag_original_indication_with_frequency.astype(int)),RHS),
                ('without_creatinine_marker',r.assign(exposure_vulnerability_category=category(r.lag_corrected_score-r.lag_aki.eq(1))),RHS),
                ('nonoverlapping_cancer',r.assign(exposure_vulnerability_category=category(r.lag_corrected_score-(r.cancer.eq(1)&r.metastatic_cancer.eq(1)))),RHS),
                ('both_labs_observed',r[r.exposure_lactate_available.eq(1)&r.exposure_aki_assessable.eq(1)].assign(icu_group=lambda x:x.icu_group.replace({'neuro':'surgical_trauma'})),RHS.replace(' + exposure_lactate_available + exposure_aki_assessable','')),
                ('matched_48h_adjustment_day3',r,RHS.replace(' + exposure_aki_assessable','')),
                ('component_adjusted',r,RHS+' + exposure_mechanical_ventilation + exposure_vasopressor + exposure_renal_replacement_therapy'),
                ('recognized_pathogen_cause_specific',r.assign(event=r.event*r.recognized_pathogen.fillna(0).astype(int)),RHS),
            ]:
                fs,ds=fit(data,name,rhs);diagnostics.append(ds);main_effects.append(contrasts(fs,data))
            ph,dp=fit(r,'early_late_associations',RHS+' + vmod:late7 + vhigh:late7 + exposure_indication_present:late7');diagnostics.append(dp)
            inds=[i for i,n in enumerate(ph.columns) if ':late7' in n]
            beta=ph.result.params[inds]; v=ph.result.cov_params()[np.ix_(inds,inds)]
            stat=float(beta@np.linalg.solve(v,beta))
            write(pd.DataFrame([{'test':'Joint change after landmark day 7 for three exposure contrasts','chi_square':stat,'df':len(inds),'p_value':chi2.sf(stat,len(inds))}]),'time_interaction_test')
            culture_process(r)
            competing(r,end)
    write(pd.concat(main_effects,ignore_index=True),'all_main_effects')
    write(pd.DataFrame(diagnostics),'model_diagnostics')
    summary['software']={'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'statsmodels':sm.__version__}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    manifest=[]
    for file in sorted((BASE/'code').glob('*.py')):
        manifest.append({'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
    write(pd.DataFrame(manifest),'analysis_code_sha256')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

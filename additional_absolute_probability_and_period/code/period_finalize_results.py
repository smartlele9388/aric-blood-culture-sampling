"""Verify existing era fit exports and summarize matched contrasts. No fits."""
from __future__ import annotations
import argparse,json,hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm

def stamp():return datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--s0-stage',type=Path,required=True)
    a=p.parse_args();out=a.output_root/'aggregate';started=stamp();checks=[];allrows=[];selected=[]
    reference={
        'cox':pd.read_csv(a.s0_stage/'aggregate/raw_reproduced_main_cox_full_precision.csv'),
        'poisson':pd.read_csv(a.s0_stage/'aggregate/raw_reproduced_sampling_primary_specimens_coefficients.csv')}
    for family,effect,critical in [('cox','hazard_ratio',norm.ppf(.975)),('poisson','rate_ratio',1.96)]:
        for variant in ['without_era','with_era']:
            name='period_'+family+'_'+variant
            c=pd.read_csv(out/(name+'_coefficients.csv'));v=pd.read_csv(out/(name+'_covariance.csv'),index_col=0)
            assert list(c.term)==list(v.index)==list(v.columns)
            beta=c.coefficient.to_numpy();cov=v.to_numpy();se=np.sqrt(np.diag(cov))
            discrepancies={
                'robust_se':float(np.max(abs(se-c.robust_se))),
                'effect':float(np.max(abs(np.exp(beta)-c[effect]))),
                'lower':float(np.max(abs(np.exp(beta-critical*se)-c.ci_lower))),
                'upper':float(np.max(abs(np.exp(beta+critical*se)-c.ci_upper))),
                'P':float(np.max(abs(2*norm.sf(abs(beta/se))-c.p_value)))}
            assert max(discrepancies.values())<1e-10
            assert np.isfinite(cov).all() and np.isfinite(beta).all()
            assert np.max(abs(cov-cov.T))<1e-12
            assert np.linalg.matrix_rank(cov)==len(c) and np.linalg.eigvalsh(cov).min()>0
            assert c.intervals.eq(52620).all() and c.stays.eq(12064).all() and c.patients.eq(11338).all()
            checks.append({'model':name,'parameters':len(c),'finite_and_full_rank':True,
                'covariance_min_eigenvalue':float(np.linalg.eigvalsh(cov).min()),'export_arithmetic_max_differences':discrepancies})
            records=c.rename(columns={effect:'effect'}).copy();records['family']=family
            allrows.extend(records.to_dict('records'))
            terms=({'high_vs_low':'exposure_vulnerability_categoryhigh','moderate_vs_low':'exposure_vulnerability_categorymoderate',
                'treatment_marker':'exposure_indication_present'} if family=='cox' else
                {'high_vs_low':"C(exposure_vulnerability_category, Treatment(reference='low'))[T.high]",
                 'moderate_vs_low':"C(exposure_vulnerability_category, Treatment(reference='low'))[T.moderate]",
                 'treatment_marker':'exposure_indication_present'})
            for label,term in terms.items():
                row=c.loc[c.term.eq(term)].iloc[0]
                selected.append({'family':family,'measure':'HR' if family=='cox' else 'IRR','contrast':label,
                    'cohort':'same era-classifiable S0 cohort','model':variant,'stays':12064,'patients':11338,'BSI_events':233,
                    'intervals':52620,'risk_person_days':49267.6509375,'effect':row[effect],
                    'ci_lower':row.ci_lower,'ci_upper':row.ci_upper,'p_value':row.p_value})
    pd.DataFrame(allrows).to_csv(out/'period_all_model_coefficients.csv',index=False)
    compare=pd.DataFrame(selected);compare.to_csv(out/'period_adjustment_matched_results.csv',index=False)
    background=[];changes=[]
    for family in ['cox','poisson']:
        effect='hazard_ratio' if family=='cox' else 'rate_ratio'
        for contrast in compare.contrast.unique():
            part=compare[(compare.family==family)&(compare.contrast==contrast)].set_index('model')
            no=part.loc['without_era'];yes=part.loc['with_era']
            changes.append({'family':family,'contrast':contrast,'without_era':no.effect,'with_era':yes.effect,
                'absolute_effect_ratio_change':yes.effect-no.effect,
                'relative_effect_ratio_change_percent':(yes.effect/no.effect-1)*100,
                'interpretation':'descriptive change in effect-ratio point estimate; not a formal test of coefficient difference'})
        background.extend(reference[family].assign(family=family,scope='original full S0 cohort, background only').to_dict('records'))
    pd.DataFrame(changes).to_csv(out/'period_adjustment_effect_changes.csv',index=False)
    pd.DataFrame(background).to_csv(out/'period_original_full_cohort_background.csv',index=False)
    ccheck=pd.read_csv(out/'period_cox_matched_design_check.csv').iloc[0]
    assert bool(ccheck.shared_design_identical) and ccheck.maximum_numeric_difference==0
    pcheck=json.loads((out/'period_poisson_matched_design_check.json').read_text())
    assert pcheck['all_shared_design_values_identical']
    diagnostics=[];phnotes=[]
    for variant in ['without_era','with_era']:
        d=pd.read_csv(out/('period_cox_'+variant+'_diagnostics.csv')).iloc[0]
        assert d.fitted_rows==52620 and d.fitted_events==233 and d.iterations<50
        assert pd.isna(d.warnings) or d.warnings==''
        z=pd.read_csv(out/('period_cox_'+variant+'_zph_log_event_time.csv'))
        flags=z[(z.term!='GLOBAL') & z.p.lt(.05)]
        phnotes.append({'model':'period_cox_'+variant,'global_PH_p':float(z.loc[z.term.eq('GLOBAL'),'p'].iloc[0]),
            'individual_PH_p_below_0_05':flags[['term','p']].to_dict('records'),
            'interpretation':'diagnostic signals retained; global nonsignificance is not proof of proportional hazards'})
        diagnostics.append({'model':'period_cox_'+variant,'converged':True,'parameters':int(d.parameters),
            'design_rank':int(d.design_rank),'fitted_rows':int(d.fitted_rows),'events':int(d.fitted_events),'warnings':0})
        d=json.loads((out/('period_poisson_'+variant+'_diagnostics.json')).read_text())
        assert d['converged'] and d['intervals']==52620 and not d['warnings']
        diagnostics.append({'model':'period_poisson_'+variant,'converged':True,'parameters':d['parameters'],
            'design_rank':d['design_rank'],'fitted_rows':d['intervals'],'events':d['BSI_events'],'warnings':len(d['warnings'])})
    runtime_log=(a.output_root/'logs/period_two_cox.log').read_text(encoding='utf-8-sig')
    locale_warning_lines=[line for line in runtime_log.splitlines() if 'Setting LC_' in line]
    check={'status':'completed','fit_exports_verified':checks,'diagnostics':diagnostics,'PH_diagnostics':phnotes,
        'all_matched_design_numeric_values_identical':True,'all_four_models_no_fit_warnings':True,
        'R_startup_locale_warning_count':len(locale_warning_lines),
        'R_startup_environment_warnings':locale_warning_lines,
        'PH_diagnostic_scope':'cox.zph nominal model-based screens, not multiplicity-adjusted or patient-cluster-bootstrap tests',
        'no_refit_in_finalization':True,'finalization_started_at':started,'finalization_completed_at':stamp()}
    (out/'period_independent_output_check.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    status_path=out/'period_execution_status.json';status=json.loads(status_path.read_text())
    if status.get('status')=='execution_failed':
        (out/'period_initial_execution_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    repaired=any(step.get('step')=='two_cox' and step.get('exit_code')!=0 for step in status['steps'])
    status.update(status='completed',actual_model_fits=4,
        final_code_sha256={n:sha(a.output_root/'code'/n) for n in ['period_fit_cox.R','period_validate_cox_design_no_fit.R','period_finalize_results.py']},
        output_finalization={'started_at':started,'completed_at':stamp(),'new_model_fits':0})
    if repaired:
        assert (a.output_root/'logs/period_cox_design_validation_only.log').exists()
        status['postfit_validation_repair']={'initial_failure':'R identical(unname(x0),unname(x1 subset)) also compared matrix attributes; both fits had completed successfully',
            'repair':'Compared dimensions and vectorized numeric entries; independent model.matrix reconstruction verified every common value identical',
            'new_model_fits':0,'validation_log':'logs/period_cox_design_validation_only.log',
            'finalization_started_at':started,'finalization_completed_at':stamp(),
            'original_failed_log_retained':'logs/period_two_cox.log'}
    status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(compare.to_string(index=False),flush=True)
    print(json.dumps(check,indent=2),flush=True)

if __name__=='__main__':main()

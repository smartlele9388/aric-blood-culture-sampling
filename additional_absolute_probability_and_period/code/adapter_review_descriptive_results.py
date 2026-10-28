"""Small independent review of derived day-28 fractions and aggregate point outputs."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();agg=a.root/'aggregate'
    d=pd.read_csv(a.root/'secure_derived_inputs/cohort_standardization_S0.csv')
    rows=[]
    for g in('low','moderate','high'):
        z=d[d.baseline_group.eq(g)];events=int(z.bsi_status.eq(1).sum())
        assert z.bsi_time_days.le(28).all()
        rows.append({'group':g,'eligible_stays':len(z),'first_detected_BSI_by_day28':events,'probability':events/len(z),'percent':100*events/len(z),'death_exits':int(z.bsi_status.eq(2).sum()),'discharge_exits':int(z.bsi_status.eq(3).sum()),'attribution_exits':int(z.bsi_status.eq(4).sum()),'administrative_censors_day28':int(z.bsi_status.eq(0).sum()),'interpretation':'Observed cumulative incidence with actual competing exits; not risk under forced continued catheterization for28days'})
    assert sum(r['first_detected_BSI_by_day28'] for r in rows)==268
    pd.DataFrame(rows).to_csv(agg/'BSI_by_day28_group_context.csv',index=False,float_format='%.17g')
    crude=pd.read_csv(agg/'crude_cif.csv');standard=pd.read_csv(agg/'standardized_cif_point.csv');rd=pd.read_csv(agg/'standardized_risk_differences_point.csv')
    merged=crude[['endpoint','group','time_days','probability']].merge(standard,on=['endpoint','group','time_days'],validate='one_to_one')
    assert len(merged)==18;merged['crude_percent']=100*merged.probability;merged=merged.rename(columns={'estimate':'standardized_percent'})
    merged['standardized_minus_crude_percentage_points']=merged.standardized_percent-merged.crude_percent
    assert np.isfinite(merged.standardized_percent).all() and merged.standardized_percent.between(0,100).all()
    errors=[]
    for row in rd.itertuples():
        z=standard[standard.endpoint.eq(row.endpoint)&standard.time_days.eq(row.time_days)].set_index('group').estimate
        errors.append(abs(z[row.contrast.replace('_minus_low','')]-z['low']-row.estimate))
    assert max(errors)<1e-11 and len(errors)==12
    for _,z in standard.groupby(['endpoint','group']):assert np.diff(z.sort_values('time_days').estimate).min()>=-1e-12
    merged.drop(columns='probability').to_csv(agg/'crude_vs_standardized_point_comparison.csv',index=False,float_format='%.17g')
    diagnostics=pd.read_csv(agg/'model_diagnostics.csv');ph=pd.read_csv(agg/'standardized_model_PH_diagnostics.csv')
    assert len(diagnostics)==8 and diagnostics.n.eq(13863).all() and diagnostics.parameters.eq(diagnostics.matrix_rank).all()
    assert diagnostics.finite_coefficients.all() and diagnostics.finite_covariance.all()
    globals=ph[ph.term.eq('GLOBAL')][['endpoint','cause','p']].to_dict('records')
    report={'status':'point_outputs_checked','standardized_point_cells':18,'paired_difference_cells':12,'maximum_pair_difference_arithmetic_error':max(errors),'all_eight_cause_models_fullrank_finite_fullcohort':True,'eight_cause_model_global_PH_diagnostics':globals,'CIF_inference_limitation':'Significant departures in sampling and competing-exit hazards remain; finite convergence and patient-block bootstrap do not establish PH or correct misspecification. Retain as exploratory model-dependent standardization.','figure_reading':'Numerical point-output check only; visual review is recorded separately in FIGURE_QA.md. This script does not inspect images.','full_bootstrap_review':'Not performed by this point-check script; see independent_result_qa.json for separately executed final coverage and percentile checks.'}
    (agg/'independent_scientific_point_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'completed','day28_BSI_total':268,'point_cells':len(merged),'paired_difference_max_error':max(errors)}))

if __name__=='__main__':main()

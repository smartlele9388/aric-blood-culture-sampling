"""Independent aggregate-only QA of the completed R exports and percentile CIs."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
root=Path(sys.argv[1]);a=root/'aggregate'
s=pd.read_csv(a/'bootstrap_attempt_status.csv');r=pd.read_csv(a/'bootstrap_standardized_cif.csv');d=pd.read_csv(a/'bootstrap_standardized_differences.csv')
pc=pd.read_csv(a/'standardized_cif.csv');dc=pd.read_csv(a/'standardized_risk_differences.csv')
assert len(s)==1000 and not s.replicate.duplicated().any() and set(s.replicate)==set(range(1,1001))
success=set(s.loc[s.success,'replicate']);bad=set(s.loc[~s.success,'replicate'])
assert set(r.replicate)==set(d.replicate)==success
assert not set(r.replicate).intersection(bad)
assert r.groupby('replicate').size().eq(18).all() and d.groupby('replicate').size().eq(12).all()
assert len(pc)==18 and len(dc)==12
assert pc.estimate.between(0,100).all() and r.estimate.between(0,100).all()
for _,z in r.groupby(['replicate','endpoint','group']):assert np.diff(z.sort_values('time_days').estimate).min()>=-1e-10
wide=r.pivot(index=['replicate','endpoint','time_days'],columns='group',values='estimate')
errors=[]
for g in ['moderate','high']:
 z=d[d.contrast.eq(g+'_minus_low')].set_index(['replicate','endpoint','time_days'])
 errors.append(float(np.max(np.abs(z.estimate-(wide[g]-wide.low)))))
assert max(errors)<1e-10
ci_errors=[]
for pt,bt,key in [(pc,r,'group'),(dc,d,'contrast')]:
 for _,z in pt.iterrows():
  x=bt.loc[bt.endpoint.eq(z.endpoint)&bt.time_days.eq(z.time_days)&bt[key].eq(z[key]),'estimate']
  assert len(x)==len(success)
  q=np.quantile(x,[.025,.975],method='linear')
  ci_errors.append(float(np.max(np.abs(q-[z.ci_lower,z.ci_upper]))))
 assert pt.successful_replicates.eq(len(success)).all() and pt.attempted_replicates.eq(1000).all()
assert max(ci_errors)<1e-10
dg=pd.read_csv(a/'model_diagnostics.csv');cf=pd.read_csv(a/'standardized_model_coefficients.csv');cv=pd.read_csv(a/'standardized_model_covariance.csv');ph=pd.read_csv(a/'standardized_model_PH_diagnostics.csv')
assert len(dg)==8 and len(cf)==88 and len(cv)==968 and len(ph)==48
assert dg.n.eq(13863).all() and dg.parameters.eq(11).all() and dg.matrix_rank.eq(11).all()
assert dg.iterations.lt(50).all() and dg.minimum_covariance_eigenvalue.gt(0).all()
assert dg.probability_closure_max_error.lt(1e-8).all() and dg.profile_collapse_max_error.lt(1e-12).all()
report={'status':'passed','planned_attempts':1000,'successful_attempts':len(success),'failed_attempts':len(bad),
 'failed_ids':sorted(bad),'all_successful_pairs_complete':True,'failed_pairs_not_replaced_or_retained':True,
 'paired_difference_arithmetic_max_absolute_error_pp':max(errors),
 'independent_percentile_CI_max_absolute_error_percent_or_pp':max(ci_errors),
 'point_CIF_rows':len(pc),'point_contrast_rows':len(dc),'point_cause_model_coefficient_rows':len(cf),
 'all_diagnostics_expected_dimensions':True,
 'contrasts_with_CI_crossing_zero':dc.loc[(dc.ci_lower<=0)&(dc.ci_upper>=0),['endpoint','time_days','contrast']].to_dict('records'),
 'PH_global_rows_nominal_p_under_005':ph.loc[ph.term.eq('GLOBAL')&ph.p.lt(.05),['endpoint','cause','p']].to_dict('records'),
 'note':'QA confirms implementation/export arithmetic, not validity of the PH model or clinical interpretation.'}
(a/'standardization_output_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

"""Aggregate-only final percentile/paired-contrast audit. No fitting or image review."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();agg=a.root/'aggregate'
status=pd.read_csv(agg/'bootstrap_attempt_status.csv');bs=pd.read_csv(agg/'bootstrap_standardized_cif.csv');bd=pd.read_csv(agg/'bootstrap_standardized_differences.csv')
assert len(status)==1000 and status.replicate.nunique()==1000 and set(status.replicate)==set(range(1,1001))
success=set(status.loc[status.success,'replicate']);assert len(success)==995 and set(bs.replicate)==set(bd.replicate)==success
assert len(bs)==995*18 and len(bd)==995*12
max_ci_error=0.
for name,boot,key in [('standardized_cif.csv',bs,'group'),('standardized_risk_differences.csv',bd,'contrast')]:
    final=pd.read_csv(agg/name)
    for _,r in final.iterrows():
        values=boot.loc[boot.endpoint.eq(r.endpoint)&boot.time_days.eq(r.time_days)&boot[key].eq(r[key]),'estimate']
        assert len(values)==995
        err=np.max(np.abs(np.quantile(values,[.025,.975])-np.array([r.ci_lower,r.ci_upper])))
        assert err<1e-11;max_ci_error=max(max_ci_error,float(err))
pivot=bs.pivot(index=['replicate','endpoint','time_days'],columns='group',values='estimate')
for g in('moderate','high'):
    observed=bd[bd.contrast.eq(g+'_minus_low')].set_index(['replicate','endpoint','time_days']).reindex(pivot.index).estimate
    assert np.allclose(pivot[g]-pivot.low,observed,atol=1e-11,rtol=0)
report={'status':'passed','attempts':1000,'paired_successes':995,'failed_attempts':5,'all_30_interval_cells_checked':True,'max_percentile_arithmetic_error':max_ci_error,'successful_paired_contrasts_checked':11940,'models_refitted':0,'individual_patient_data_read':False,'scope':'Aggregate numerical consistency; not clinical validation or proof of model assumptions. Manual narrative/visual review is documented separately.'}
(agg/'independent_scientific_final_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))

"""Independent aggregate arithmetic/coverage and original-input integrity checks.

No additional model is fitted. Patient tables are not read or shared here.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import norm

root=Path(__file__).resolve().parents[1];agg=root/'aggregate'
checks={}
crude=pd.read_csv(agg/'crude_cif.csv')
boot=pd.read_csv(agg/'crude_bootstrap_aggregate_estimates.csv')
assert len(crude)==18 and not crude.duplicated(['endpoint','group','time_days']).any()
assert (np.abs(crude.probability-crude.target_events_by_time/crude.initial_n)<1e-12).all()
for r in crude.itertuples():
    b=boot[(boot.endpoint==r.endpoint)&(boot.group==r.group)&(boot.time_days==r.time_days)]
    assert len(b)==1000 and b.replicate.nunique()==1000
    assert np.allclose(np.quantile(b.probability,[.025,.975]),[r.lower_95,r.upper_95],rtol=0,atol=1e-14)
    assert r.target_events_by_time+r.death_exits_by_time+r.discharge_exits_by_time+r.attribution_exits_by_time+r.administrative_censors_by_time+r.n_at_risk_after==r.initial_n
checks['crude_18_cells_and_percentile_intervals']='passed; actual AJ equals independent event fraction; exit partition sums to denominators'

s=pd.read_csv(agg/'standardized_cif.csv');rd=pd.read_csv(agg/'standardized_risk_differences.csv')
bs=pd.read_csv(agg/'bootstrap_standardized_cif.csv');bd=pd.read_csv(agg/'bootstrap_standardized_differences.csv')
st=pd.read_csv(agg/'bootstrap_attempt_status.csv')
assert len(st)==1000 and set(st.replicate)==set(range(1,1001))
ok=set(st.loc[st.success,'replicate']);assert len(ok)>=950
assert set(bs.replicate)==set(bd.replicate)==ok
assert len(s)==18 and len(rd)==12 and len(bs)==18*len(ok) and len(bd)==12*len(ok)
for table,btable,key in [(s,bs,'group'),(rd,bd,'contrast')]:
    assert not table.duplicated(['endpoint','time_days',key]).any()
    for _,r in table.iterrows():
        b=btable[(btable.endpoint==r.endpoint)&(btable.time_days==r.time_days)&(btable[key]==r[key])]
        assert len(b)==len(ok)
        assert np.allclose(np.quantile(b.estimate,[.025,.975]),[r.ci_lower,r.ci_upper],atol=1e-11,rtol=0)
for table,diffs,keys in [(s,rd,['endpoint','time_days']),(bs,bd,['replicate','endpoint','time_days'])]:
    piv=table.pivot(index=keys,columns='group',values='estimate')
    for g in ['moderate','high']:
        z=diffs[diffs.contrast.eq(g+'_minus_low')].set_index(keys).reindex(piv.index)
        assert np.allclose(piv[g]-piv.low,z.estimate,atol=1e-11,rtol=0)
checks['standardized_all_cells_bootstrap_percentiles_and_paired_differences']='passed; all requested groups/times and 1000 attempted replicate IDs accounted'

models=[]
for family in ['cox','poisson']:
    for spec in ['without_era','with_era']:
        stem=f'period_{family}_{spec}'
        d=pd.read_csv(agg/f'{stem}_coefficients.csv')
        cov=pd.read_csv(agg/f'{stem}_covariance.csv',index_col=0).reindex(index=d.term,columns=d.term)
        assert np.isfinite(cov.to_numpy()).all()
        assert np.allclose(np.diag(cov),d.robust_se**2,atol=1e-13,rtol=1e-12)
        ratio=d.hazard_ratio if family=='cox' else d.rate_ratio
        q=norm.ppf(.975) if family=='cox' else 1.96
        assert np.allclose(np.exp(d.coefficient),ratio,atol=1e-12,rtol=1e-12)
        assert np.allclose(np.exp(d.coefficient-q*d.robust_se),d.ci_lower,atol=1e-12,rtol=1e-12)
        assert np.allclose(np.exp(d.coefficient+q*d.robust_se),d.ci_upper,atol=1e-12,rtol=1e-12)
        assert np.allclose(2*norm.sf(abs(d.coefficient/d.robust_se)),d.p_value,atol=1e-12,rtol=1e-10)
        assert set(d.stays)=={12064} and set(d.intervals)=={52620}
        models.append({'model':stem,'coefficient_count':len(d),'full_covariance_rank':int(np.linalg.matrix_rank(cov))})
checks['period_all_coefficients_covariance_and_CI_arithmetic']=models

provenance=json.loads((root/'input_provenance_local.json').read_text(encoding='utf-8'))
inputchecks=[]
for name,record in provenance['sources'].items():
    p=Path(record['path'])
    if p.is_file() and p.suffix=='.gz':
        h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
        assert h==record['sha256']
        inputchecks.append({'source_basename':p.name,'sha256_unchanged':True})
checks['original_S0_patient_cache_read_only_integrity']=inputchecks
checks['limitations']='Numerical checks do not prove endpoint clinical validity, PH, absence of confounding or a causal interpretation.'
(agg/'independent_result_qa.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2),encoding='utf-8')
print(json.dumps({'status':'passed','successful_standardized_replicates':len(ok),'crude_cells':18,'standardized_cells':18,'differences':12,'period_models':4}))

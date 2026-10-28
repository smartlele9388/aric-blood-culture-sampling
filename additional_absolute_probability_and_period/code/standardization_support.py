"""Aggregate overlap/time support; no patient rows or identifiers are exported."""
from pathlib import Path
import sys,json,itertools
import numpy as np
import pandas as pd
root=Path(sys.argv[1]);out=root/'aggregate'
d=pd.read_csv(root/'secure_derived_inputs/cohort_standardization_S0.csv')
groups=['low','moderate','high'];rows=[];overlap=[];age=[];causes=[]
small=lambda n: '<11' if 0<int(n)<11 else str(int(n))
for e in ['sampling','bsi']:
 t=d[e+'_time_days'];st=d[e+'_status']
 for g in ['overall']+groups:
  ix=np.ones(len(d),dtype=bool) if g=='overall' else d.baseline_group.eq(g)
  for h in [1,3,7]:
   row={'endpoint':e,'group':g,'time_days':h,'initial_stays':int(ix.sum()),
       'at_risk_just_before':int((ix&t.ge(h)).sum()),'still_observed_after':int((ix&t.gt(h)).sum()),
       'last_observed_time_days':float(t[ix].max()),'beyond_observed_range':bool(h>t[ix].max())}
   for c in range(5):
    v=t[ix&st.eq(c)&t.le(h)]
    row[f'cause_{c}_by_time']=int(len(v));row[f'cause_{c}_last_event_time_at_or_before_horizon']=None if not len(v) else float(v.max())
   rows.append(row)
 for var in ['baseline_group','icu_group','admission_group','female']:
  for cat,z in d.groupby(var):
   for c in range(5):
    n=int(z[e+'_status'].eq(c).sum())
    causes.append({'endpoint':e,'variable':var,'level':str(cat),'cause':c,'events_or_censors':small(n),
                   'zero_events':n==0,'sparse_1_to_10':0<n<11,'level_stays':len(z)})
for g,z in d.groupby('baseline_group'):
 a=z.age if 'age' in z else z.age_per_10y*10+60
 q=a.quantile([0,.05,.25,.5,.75,.95,1])
 age.append({'group':g,'n':len(z),'min':q.iloc[0],'p05':q.iloc[1],'q25':q.iloc[2],'median':q.iloc[3],
             'q75':q.iloc[4],'p95':q.iloc[5],'max':q.iloc[6],'mean':a.mean(),'sd':a.std()})
for var in ['icu_group','admission_group','female']:
 for cat in sorted(d[var].unique()):
  for g in groups:
   n=int((d[var].eq(cat)&d.baseline_group.eq(g)).sum())
   overlap.append({'variable':var,'level':str(cat),'group':g,'n':small(n),'empty':n==0,'sparse_1_to_10':0<n<11})
# Joint support used for a diagnostic only; no outcome-driven filtering or alternate cohort.
joint=[]
for key,z in d.groupby(['icu_group','admission_group','female']):
 counts=z.baseline_group.value_counts().reindex(groups,fill_value=0)
 joint.append({'icu_group':key[0],'admission_group':key[1],'female':key[2],
               **{g:small(counts[g]) for g in groups},'missing_group_count':int(counts.eq(0).sum())})
age_min=max(x['min'] for x in age);age_max=min(x['max'] for x in age)
missing_joint_keys=[k for k,z in d.groupby(['icu_group','admission_group','female']) if set(z.baseline_group)!=set(groups)]
mask=np.zeros(len(d),dtype=bool)
for k in missing_joint_keys:mask |= d.icu_group.eq(k[0])&d.admission_group.eq(k[1])&d.female.eq(k[2])
assessment={'no_early_non_event_censoring':bool(all(not((d[e+'_status']==0)&(d[e+'_time_days']<28-1e-10)).any() for e in ['sampling','bsi'])),
 'all_requested_horizons_within_every_group_observed_range':not any(r['beyond_observed_range'] for r in rows),
 'marginal_empty_cells':sum(r['empty'] for r in overlap),'marginal_sparse_1_to_10':sum(r['sparse_1_to_10'] for r in overlap),
 'observed_joint_icu_admission_sex_cells':len(joint),'joint_cells_missing_at_least_one_group':len(missing_joint_keys),
 'stays_in_joint_cells_missing_a_group':int(mask.sum()),'fraction_stays_in_joint_cells_missing_a_group':float(mask.mean()),
 'common_age_range':[float(age_min),float(age_max)],'stays_outside_common_age_range':int(((d.age_per_10y*10+60<age_min)|(d.age_per_10y*10+60>age_max)).sum()),
 'interpretation':'Support is assessed for measured covariates only. Additive Cox standardization does not remove unmeasured confounding or guarantee support for every exact age/categorical combination. No unsupported persons were deleted.',
 'secondary_support_restricted_analysis':'not_triggered; inspect quantitative support and retain full-S0 primary target.'}
for name,rr in [('overlap_and_time_support.csv',rows),('standardization_covariate_overlap.csv',overlap),('standardization_age_distribution.csv',age),('standardization_joint_covariate_support.csv',joint),('standardization_cause_category_support.csv',causes)]:
 pd.DataFrame(rr).to_csv(out/name,index=False)
(out/'standardization_support_assessment.json').write_text(json.dumps(assessment,indent=2),encoding='utf-8')
print(json.dumps(assessment,indent=2))

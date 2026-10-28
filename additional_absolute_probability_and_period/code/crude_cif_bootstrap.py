"""Competing-risk Aalen-Johansen, audited against fractions, patient-block bootstrap.

Whole admissions are the units of the estimand. Subjects are sampled with
replacement and all admissions retained with each draw's multiplicity.
"""
from __future__ import annotations
import argparse,json,time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

GROUPS=('low','moderate','high');HORIZONS=(1.,3.,7.)

def aalen_johansen(times,status,horizons):
    times=np.asarray(times,float);status=np.asarray(status,int);order=np.argsort(times,kind='stable');times=times[order];status=status[order]
    ts,starts=np.unique(times,return_index=True)
    counts=np.stack([np.add.reduceat((status==k).astype(int),starts) for k in range(5)],axis=1)
    n=len(times);surv=1.;cif=np.zeros(4);curve=[]
    for t,c in zip(ts,counts):
        if n<=0:raise RuntimeError('Risk set exhausted before last observed time')
        d=c[1:].sum();cif=cif+surv*c[1:]/n;surv=surv*(1-d/n)
        curve.append(cif.copy());n-=c.sum()
    assert n==0
    result=np.zeros((len(horizons),4));curve=np.asarray(curve)
    ix=np.searchsorted(ts,horizons,side='right')-1;ok=ix>=0;result[ok]=curve[ix[ok]]
    return result

def software_checks():
    checks=[]
    # Exact time zero occurs before positive-time exits; no epsilon.
    z=aalen_johansen([0,1,2,28],[1,2,3,0],[0,1,3,28]);assert np.allclose(z[:,0],[.25,.25,.25,.25]);checks.append('exact_time_zero_retained')
    # All at a tied time remain in the event denominator, including censors.
    x=aalen_johansen([1,1,1,2],[1,2,0,1],[1,2]);assert np.allclose(x[:,0],[.25,.75]);checks.append('tied_events_before_censoring')
    # Early censoring invalidates naive proportions; AJ still returns expected .5.
    x=aalen_johansen([.5,1,2],[0,1,2],[1,2]);assert np.allclose(x[:,0],[.5,.5]);checks.append('early_censoring_not_naive_fraction')
    for times,status in [([0,1,1,2,28],[1,2,1,3,0]),([1,1,2,3],[2,3,4,1])]:
        h=np.array([0.,1.,3.,28.]);x=aalen_johansen(times,status,h)
        for k in range(1,5):assert np.allclose(x[:,k-1],[np.mean((np.array(times)<=t)&(np.array(status)==k)) for t in h])
    checks.append('no_early_censoring_AJ_equals_fractions_for_all_four_causes')
    # A subject with two admissions contributes both every time selected.
    ids=np.array([10,10,20]);_,ix=np.unique(ids,return_inverse=True);draw=np.array([0,0]);weights=np.bincount(draw,minlength=2);assert (weights[ix]==[2,2,0]).all();assert weights[ix].sum()==4
    checks.append('repeat_patient_draw_preserves_all_admissions_and_multiplicity')
    return checks

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--replicates',type=int,default=1000);p.add_argument('--seed',type=int,default=20261015);p.add_argument('--prior-cif',type=Path);a=p.parse_args()
    started=datetime.now().astimezone().isoformat();clock=time.monotonic();checks=software_checks();d=pd.read_csv(a.cohort);out=a.output_root/'aggregate';out.mkdir(exist_ok=True)
    assert len(d)==13863 and d.bsi_status.eq(1).sum()==268
    for ep in('sampling','bsi'):
        t=d[ep+'_time_days'].to_numpy();s=d[ep+'_status'].to_numpy();assert not ((s==0)&(t<28)).any()
    patients,pidx=np.unique(d.subject_id,return_inverse=True);npat=len(patients)
    cells=[];rows=[];curverows=[];maxerr=0.
    for ep in('sampling','bsi'):
        for group in GROUPS:
            mask=d.baseline_group.eq(group).to_numpy();gd=d[mask];t=gd[ep+'_time_days'].to_numpy();s=gd[ep+'_status'].to_numpy();n=len(gd)
            aj=aalen_johansen(t,s,HORIZONS);grid=np.round(np.arange(0,7.00001,.1),10);curve=aalen_johansen(t,s,grid)
            for h,v in zip(grid,curve):curverows.append({'endpoint':ep,'group':group,'time_days':h,'probability':v[0],'initial_n':n})
            den=np.bincount(pidx,weights=mask.astype(int),minlength=npat)
            for j,h in enumerate(HORIZONS):
                event=(s==1)&(t<=h);frac=event.mean();maxerr=max(maxerr,abs(frac-aj[j,0]));assert abs(frac-aj[j,0])<1e-12
                row={'endpoint':ep,'group':group,'time_days':h,'initial_n':n,'target_events_by_time':int(event.sum()),'death_exits_by_time':int(((s==2)&(t<=h)).sum()),'discharge_exits_by_time':int(((s==3)&(t<=h)).sum()),'attribution_exits_by_time':int(((s==4)&(t<=h)).sum()),'administrative_censors_by_time':int(((s==0)&(t<=h)).sum()),'n_at_risk_just_before':int((t>=h).sum()),'n_at_risk_after':int((t>h).sum()),'probability':float(aj[j,0]),'fraction_check':frac}
                rows.append(row)
                hit=mask & d[ep+'_status'].eq(1).to_numpy() & d[ep+'_time_days'].le(h).to_numpy()
                num=np.bincount(pidx,weights=hit.astype(int),minlength=npat);cells.append((num,den))
    num=np.stack([c[0] for c in cells],axis=1).astype(np.int64);den=np.stack([c[1] for c in cells],axis=1).astype(np.int64)
    boot=np.full((a.replicates,len(cells)),np.nan);rng=np.random.default_rng(a.seed);logs=[]
    for i in range(a.replicates):
        draw=rng.integers(0,npat,size=npat);weight=np.bincount(draw,minlength=npat);bd=weight@den;bn=weight@num
        ok=bool((bd>0).all());boot[i]=bn/bd if ok else np.nan
        logs.append({'replicate':i+1,'success':ok,'reason':'' if ok else 'Empty baseline group','sampled_unique_subject_clusters':int((weight>0).sum()),'sampled_admissions':int(np.sum(np.bincount(pidx,minlength=npat)*weight))})
        if (i+1)%100==0:print(f'crude bootstrap {i+1}/{a.replicates}',flush=True)
    assert np.isfinite(boot).all()
    ci=np.quantile(boot,[.025,.975],axis=0)
    for j,row in enumerate(rows):row.update(lower_95=ci[0,j],upper_95=ci[1,j],bootstrap_successful=a.replicates)
    result=pd.DataFrame(rows);result.to_csv(out/'crude_cif.csv',index=False,float_format='%.17g')
    pd.DataFrame(curverows).to_csv(out/'crude_curve.csv',index=False,float_format='%.17g')
    pd.DataFrame(logs).to_csv(out/'crude_bootstrap_replicate_status.csv',index=False)
    long=[]
    for i in range(a.replicates):
        for j,row in enumerate(rows):long.append({'replicate':i+1,'endpoint':row['endpoint'],'group':row['group'],'time_days':row['time_days'],'probability':boot[i,j]})
    pd.DataFrame(long).to_csv(out/'crude_bootstrap_aggregate_estimates.csv',index=False,float_format='%.17g')
    comparison=None
    if a.prior_cif:
        prior=pd.read_csv(a.prior_cif)
        # Adapt actual prior header explicitly; never use rounded manuscript values.
        keep=prior[prior.baseline_group.isin(GROUPS)&prior.days_since_landmark.isin(HORIZONS)]
        merged=result[result.endpoint.eq('sampling')].merge(keep,left_on=['group','time_days'],right_on=['baseline_group','days_since_landmark'],validate='one_to_one')
        diff=np.abs(merged.probability-merged.first_sampling_cumulative_incidence)
        assert len(merged)==9 and diff.max()<1e-12
        comparison={'n_cells':len(merged),'max_abs_point_probability_difference':float(diff.max()),'CI_not_compared':'Seed changed from original20261011 to prescribed20261015; same point estimand.'}
    summary={'status':'completed','method':'Aalen-Johansen target CIF with three distinct competing causes; events before tied administrative censors','bootstrap':'Patient-block percentile 95% intervals. No early censoring before28d: each resample CIF at1/3/7d algebraically equals its event fraction, used after independent AJ check. Each patient draw retains all admissions and repeated draws.','replicates_planned':a.replicates,'replicates_attempted':a.replicates,'replicates_successful':int(np.isfinite(boot).all(axis=1).sum()),'replicates_failed':0,'seed':a.seed,'software_checks_passed':checks,'max_AJ_fraction_abs_difference':maxerr,'prior_first_sampling_point_comparison':comparison,'early_non_event_censoring':0,'started_at':started,'completed_at':datetime.now().astimezone().isoformat(),'elapsed_seconds':time.monotonic()-clock,'curve_grid':'0 to7days by0.1day; fixed elapsed-time grid, no individualeventtimes','privacy':'Broad fixedgroups only, no patient IDs/rows/exactdates. No blanket suppression of broadgroup aggregate counts; probabilities and known denominators also imply event counts.'}
    (out/'crude_bootstrap_execution_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'status':'completed','cells':len(result),'successful_bootstrap':a.replicates,'max_AJ_fraction_abs_difference':maxerr}),flush=True)

if __name__=='__main__':main()

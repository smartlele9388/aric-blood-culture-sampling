"""Refit sampling after removing event-time diagnostic specimens only."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'ARIC_revision_20260920';ROUTE=BASE/'route_A';OUT=HERE/'results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(BASE/'association_audit_20261003/code'))
from coding_rules_final import correct_frame

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def main():
    m=module(ROUTE/'code/run_sampling_analysis.py','sampling_original')
    m.OUT=OUT
    r=pd.read_csv(BASE/'secure_derived_inputs/revised_risk_day3.csv.gz',parse_dates=['day_start','stop_datetime','landmark_time','observed_end'])
    s=pd.read_csv(ROUTE/'secure_derived_inputs/sampling_intervals_day3.csv.gz')
    assert len(r)==len(s)==60272 and r.stay_id.eq(s.stay_id).all() and r.event.eq(s.event).all()
    corrected=pd.read_csv(HERE.parent/'secure_derived_inputs/final_score_deltas_stay.csv.gz')
    s=correct_frame(s,corrected)
    source=pd.read_csv(ROUTE/'secure_derived_inputs/sampling_collections_source_checked.csv.gz',parse_dates=['event_time','charttime'])
    retained=source[source.has_non_cancelled.eq(1)].copy()
    assigned=m.assign_collections(r,retained)
    group=assigned.groupby('risk_row').collection_key.size()
    assert np.array_equal(group.reindex(np.arange(len(s)),fill_value=0).to_numpy(),s.collection_count.to_numpy())
    terminal=r.iloc[assigned.risk_row.to_numpy()].reset_index(drop=True)
    at_event=terminal.event.eq(1).to_numpy() & assigned.event_time.reset_index(drop=True).eq(terminal.stop_datetime).to_numpy()
    ts=assigned.loc[at_event].copy()
    counts=ts.groupby('risk_row').collection_key.size()
    occasion=ts.groupby('risk_row').event_time.nunique()
    assert occasion.le(1).all(), 'Only one endpoint timestamp should occur per risk interval'
    reduced=s.copy()
    reduced['collection_count']=s.collection_count-counts.reindex(np.arange(len(s)),fill_value=0).to_numpy()
    reduced['occasion_count']=s.occasion_count-occasion.reindex(np.arange(len(s)),fill_value=0).to_numpy()
    assert reduced.collection_count.ge(0).all() and reduced.occasion_count.ge(0).all()
    assert int(s.collection_count.sum()-reduced.collection_count.sum())==len(ts)
    assert int(s.occasion_count.sum()-reduced.occasion_count.sum())==int(occasion.sum())
    summary={'reference_intervals':len(s),'reference_BSI_events':int(s.event.sum()),'risk_days':float(s.duration_days.sum()),
             'reference_specimens':int(s.collection_count.sum()),'terminal_event_time_specimens_removed':len(ts),
             'remaining_specimens':int(reduced.collection_count.sum()),'reference_occasions':int(s.occasion_count.sum()),
             'terminal_event_time_occasions_removed':int(occasion.sum()),'remaining_occasions':int(reduced.occasion_count.sum()),
             'BSI_event_intervals_with_terminal_specimen':int(len(counts)),'patient_rows_exported':False,
             'formula':m.PRIMARY,'count_policy':'Noncancelled specimens exactly at BSI event interval stop time removed; earlier and non-BSI collections retained'}
    (OUT/'terminal_specimen_sensitivity_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    diagnostics=[]
    for label,outcome in [('specimens','collection_count'),('occasions','occasion_count')]:
        _,diag=m.fit_poisson(reduced,'sampling_without_terminal_BSI_'+label,outcome,m.PRIMARY)
        diag['formula']=m.PRIMARY;diagnostics.append(diag)
    pd.DataFrame(diagnostics).to_csv(OUT/'terminal_specimen_sampling_diagnostics.csv',index=False)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

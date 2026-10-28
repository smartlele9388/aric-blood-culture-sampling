"""Reconstruct treatment markers with explicit item IDs and exact windows."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parent
sys.path.insert(0,str(ROOT))
RAW=Path((__import__("os").environ.get("MIMIC_ROOT", "MIMIC_ROOT") + ''))
SEC=BASE/'secure_derived_inputs'; SEC.mkdir(exist_ok=True)
MAPPING={'vasopressor':[221289,221653,221662,221749,221906,222315],
         'tpn':[225916,225917], 'mechanical_ventilation':[225792],
         'renal_replacement_therapy':[225441,225802,225803,225809,225955]}

def mark(days,events):
    result=pd.Series(0,index=days.index,dtype=int)
    for stay,idx in days.groupby('stay_id',sort=False).groups.items():
        e=events[events.stay_id.eq(stay)].sort_values('starttime')
        if e.empty: continue
        starts=e.starttime.astype('int64').to_numpy()
        ends=np.maximum.accumulate(e.endtime.astype('int64').to_numpy())
        low=days.loc[idx,'day_start'].astype('int64').to_numpy()
        high=days.loc[idx,'day_end'].astype('int64').to_numpy()
        pos=np.searchsorted(starts,high,side='left')-1
        good=pos>=0
        good[good]=ends[pos[good]]>low[good]
        result.loc[idx]=good.astype(int)
    return result

def main():
    states={s:pd.read_csv(ROOT/'secure_derived_inputs/host_vulnerability_indication_day3_landmark'/f'patient_day_states_{s}.csv.gz',parse_dates=['day_start','day_end']) for s in ['day3','exact48h']}
    stays=set(pd.concat(states.values()).stay_id)
    inputs=[]
    keep=set(MAPPING['vasopressor']+MAPPING['tpn'])
    for chunk in pd.read_csv(RAW/'icu/inputevents.csv.gz',usecols=['stay_id','itemid','starttime','endtime','statusdescription'],chunksize=500000):
        c=chunk[chunk.stay_id.isin(stays)&chunk.itemid.isin(keep)&~chunk.statusdescription.eq('Rewritten')]
        if len(c): inputs.append(c)
    inp=pd.concat(inputs,ignore_index=True)
    proc=pd.read_csv(RAW/'icu/procedureevents.csv.gz',usecols=['stay_id','itemid','starttime','endtime','statusdescription'])
    proc=proc[proc.stay_id.isin(stays)&~proc.statusdescription.eq('Rewritten')]
    for d in [inp,proc]:
        d['starttime']=pd.to_datetime(d.starttime); d['endtime']=pd.to_datetime(d.endtime)
        d.dropna(subset=['starttime','endtime'],inplace=True)
    qc=[]
    for scenario,d in states.items():
        output=d[['stay_id','local_day']].copy()
        for variable,ids in MAPPING.items():
            src=inp if variable in ['vasopressor','tpn'] else proc
            e=src[src.itemid.isin(ids)&(src.endtime>src.starttime)].drop_duplicates(['stay_id','itemid','starttime','endtime'])
            output[variable]=mark(d,e)
            qc.append({'scenario':scenario,'variable':variable,'old_positive_windows':int(d[variable].sum()),'new_positive_windows':int(output[variable].sum()),'changed_windows':int(d[variable].ne(output[variable]).sum())})
        output.to_csv(SEC/f'corrected_treatments_{scenario}.csv.gz',index=False)
        print(scenario,'treatment features complete',flush=True)
    pd.DataFrame(qc).to_csv(BASE/'audit_internal/treatment_changes.csv',index=False)
    items=pd.read_csv(RAW/'icu/d_items.csv.gz')
    mapping=pd.DataFrame([{'variable':v,'itemid':i} for v,ids in MAPPING.items() for i in ids]).merge(items[['itemid','label','linksto']],on='itemid')
    mapping.to_csv(BASE/'audit_internal/treatment_mapping.csv',index=False)

if __name__=='__main__': main()

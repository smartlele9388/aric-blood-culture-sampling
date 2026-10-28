"""Exact isolated functions from frozen original source; no entrypoint side effects."""
import numpy as np
import pandas as pd

def validate_risk(r):
    required = ['subject_id','hadm_id','stay_id','day_start','stop_datetime','t_start','t_stop','duration_days','event']
    if r[required].isna().any().any():
        raise ValueError('Required risk fields contain missing values')
    if r.duplicated(['stay_id','day_start']).any() or not r.duration_days.gt(0).all():
        raise ValueError('Duplicate interval start or nonpositive risk duration')
    if not np.allclose((r.stop_datetime-r.day_start).dt.total_seconds()/86400, r.duration_days):
        raise ValueError('Actual datetime duration differs from model duration')
    spans=[]
    for stay,g in r.groupby('stay_id', sort=False):
        g=g.sort_values('day_start')
        if g.subject_id.nunique()!=1 or g.hadm_id.nunique()!=1:
            raise ValueError('A stay maps to multiple patients or admissions')
        if len(g)>1 and not np.array_equal(g.day_start.to_numpy()[1:],g.stop_datetime.to_numpy()[:-1]):
            raise ValueError('Risk intervals overlap or have gaps within stay')
        if g.event.sum()>1 or (g.event.eq(1).any() and not g.event.iloc[-1]==1):
            raise ValueError('BSI must be a single terminal event')
        spans.append((stay,g.subject_id.iloc[0],g.hadm_id.iloc[0],g.day_start.iloc[0],g.stop_datetime.iloc[-1]))
    spans=pd.DataFrame(spans,columns=['stay_id','subject_id','hadm_id','start','stop'])
    if spans.groupby('hadm_id').subject_id.nunique().gt(1).any():
        raise ValueError('Hospital admission maps to multiple patients')
    # Endpoints are inclusive for terminal diagnostic collections: touching stays
    # would be ambiguous, so they cannot silently share a culture.
    for _,g in spans.groupby('hadm_id', sort=False):
        g=g.sort_values('start')
        if len(g)>1 and np.any(g.start.to_numpy()[1:]<=g.stop.to_numpy()[:-1]):
            raise ValueError('Hospital-to-stay risk windows overlap or touch')
    if not np.allclose(r.t_start,np.rint(r.t_start)) or r.t_start.min()<0 or r.t_stop.max()>28+1e-10:
        raise ValueError('Day-3 intervals must start on the landmark daily grid within 28 days')
    for boundary in (2,7,14):
        if ((r.t_start<boundary)&(r.t_stop>boundary)).any():
            raise ValueError('Risk interval crosses a prespecified time-band boundary')
    return {'intervals':len(r),'stays':int(r.stay_id.nunique()),'patients':int(r.subject_id.nunique()),
            'admissions':int(r.hadm_id.nunique()),'events':int(r.event.sum()),
            'risk_person_days':float(r.duration_days.sum()),'overlapping_or_touching_stay_windows':0,
            'intervals_crossing_time_band_boundary':0}

def assign_collections(r, c):
    """Assign each specimen once; internal boundaries go forward, final end is included."""
    if c.collection_key.isna().any() or c.collection_key.duplicated().any():
        raise ValueError('Collection keys must be complete and globally unique after deduplication')
    r=r.reset_index(drop=True)
    out=c.copy().reset_index(drop=True)
    assigned=np.full(len(out),-1,dtype=np.int64)
    grouped=out.groupby('hadm_id',sort=False).indices
    for _,g in r.groupby('stay_id',sort=False):
        g=g.sort_values('day_start')
        candidate=grouped.get(g.hadm_id.iloc[0],np.array([],dtype=int))
        if not len(candidate): continue
        e=out.iloc[candidate]
        starts=g.day_start.astype('int64').to_numpy(); ends=g.stop_datetime.astype('int64').to_numpy()
        times=e.event_time.astype('int64').to_numpy()
        pos=np.searchsorted(starts,times,side='right')-1
        valid=pos>=0
        valid[valid]=times[valid]<=ends[pos[valid]]
        valid &= e.event_time.notna().to_numpy()
        valid &= e.subject_id.eq(g.subject_id.iloc[0]).to_numpy()
        ci=candidate[valid]
        if (assigned[ci]>=0).any(): raise ValueError('A collection belongs to more than one stay/interval')
        assigned[ci]=g.index.to_numpy()[pos[valid]]
    out['risk_row']=assigned
    out=out[out.risk_row.ge(0)].copy()
    if out.collection_key.duplicated().any(): raise ValueError('Duplicate assigned collection')
    return out

def collapse_occasions(assigned):
    """A timestamp grouping sensitivity, not a clinically validated culture-set definition."""
    return assigned.groupby(['risk_row','event_time'],as_index=False).agg(positive=('positive','max'),specimens=('collection_key','size'))

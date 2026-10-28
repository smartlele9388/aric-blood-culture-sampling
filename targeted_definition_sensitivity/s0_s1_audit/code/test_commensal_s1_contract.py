"""Fixed-definition edge cases: timestamp confirmation and original day membership."""
import re
from types import SimpleNamespace
import pandas as pd
import audit_commensal_s0_s1 as audit

LOCKED=SimpleNamespace(COMMON_COMMENSAL_RE=re.compile('commensal'))
CONTEXT=pd.DataFrame([{'subject_id':1,'hadm_id':2,'stay_id':3,'admittime':pd.Timestamp('2020-01-01')}])


def source(rows):
    out=[]
    for key,spec,time,exact,organism in rows:
        out.append({'subject_id':1,'hadm_id':2,'collection_key':key,'isolate_key':key+'|'+organism,'event_time':pd.Timestamp(time),
            'charttime':pd.Timestamp(exact) if exact else pd.NaT,'micro_specimen_id':spec,'org_norm':organism,'org_name':organism})
    return pd.DataFrame(out)


def run(rows): return audit.build_s1(source(rows),CONTEXT,LOCKED)


def test_distinct_specimens_same_timestamp_do_not_confirm_s1():
    f0,_,f1,_,_,_,_=run([('specimen:10',10,'2020-01-02 12:00','2020-01-02 12:00','commensal'),('specimen:11',11,'2020-01-02 12:00','2020-01-02 12:00','commensal')])
    assert len(f0)==1 and f1.empty


def test_one_second_difference_confirms_no_minimum_interval_added():
    f0,_,f1,_,_,_,_=run([('specimen:10',10,'2020-01-02 12:00','2020-01-02 12:00','commensal'),('specimen:11',11,'2020-01-02 12:00:01','2020-01-02 12:00:01','commensal')])
    assert len(f0)==len(f1)==2


def test_missing_exact_timestamp_never_substitutes_chartdate_for_s1():
    f0,_,f1,_,_,_,_=run([('specimen:10',10,'2020-01-02',None,'commensal'),('specimen:11',11,'2020-01-03 12:00','2020-01-03 12:00','commensal')])
    assert len(f0)==2 and f1.empty


def test_missing_actual_specimen_id_fallback_does_not_confirm_s1():
    f0,_,f1,_,_,_,_=run([('fallback:A',None,'2020-01-02 12:00','2020-01-02 12:00','commensal'),('specimen:11',11,'2020-01-02 13:00','2020-01-02 13:00','commensal')])
    assert len(f0)==2 and f1.empty


def test_repeat_susceptibility_rows_from_one_specimen_do_not_confirm():
    row=('specimen:10',10,'2020-01-02 12:00','2020-01-02 12:00','commensal')
    f0,_,f1,_,_,_,_=run([row,row,row])
    assert f0.empty and f1.empty


def test_original_entire_day_membership_retained_not_pair_rows_only():
    f0,_,f1,i1,_,_,_=run([('specimen:10',10,'2020-01-02 12:00','2020-01-02 12:00','commensal'),('specimen:11',11,'2020-01-02 13:00','2020-01-02 13:00','commensal'),('fallback:C',None,'2020-01-03',None,'commensal')])
    assert len(f0)==len(f1)==len(i1)==3


def test_two_calendar_day_gap_not_an_original_s0_pair():
    f0,_,f1,_,_,_,_=run([('specimen:10',10,'2020-01-02 12:00','2020-01-02 12:00','commensal'),('specimen:11',11,'2020-01-04 13:00','2020-01-04 13:00','commensal')])
    assert f0.empty and f1.empty


def test_mixed_event_retains_recognized_pathogen_when_commensal_fails():
    f0,_,f1,_,_,_,_=run([('specimen:10',10,'2020-01-02 12:00','2020-01-02 12:00','commensal'),('specimen:11',11,'2020-01-02 12:00','2020-01-02 12:00','commensal'),('specimen:10',10,'2020-01-02 12:00','2020-01-02 12:00','pathogen')])
    assert len(f0)==len(f1)==1
    assert f0.iloc[0].confirmed_common_commensal==1 and f1.iloc[0].confirmed_common_commensal==0
    assert f1.iloc[0].recognized_pathogen==1

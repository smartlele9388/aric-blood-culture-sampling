"""Version-aware malignant-neoplasm flags for the retrospective score audit."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIAGNOSES = Path((__import__("os").environ.get("MIMIC_ROOT", "MIMIC_ROOT") + '/hosp/diagnoses_icd.csv.gz'))

def classify(code, version):
    c = str(code).upper().replace('.', '').strip()
    if str(version) in ('10', '10.0'):
        cancer = c.startswith('C')
        metastatic = c.startswith(('C77', 'C78', 'C79', 'C7B')) or c == 'C800'
    elif str(version) in ('9', '9.0'):
        prefix = int(c[:3]) if c[:3].isdigit() else -1
        malignant_209 = c.startswith(('2090', '2091', '2092', '2093', '2097'))
        cancer = 140 <= prefix <= 208 or malignant_209
        metastatic = c.startswith(('196', '197', '198', '2097')) or c == '1990'
    else:
        cancer = metastatic = False
    return int(cancer), int(metastatic)

def stay_flag_deltas(stays):
    ids = set(stays.hadm_id)
    pieces = []
    for chunk in pd.read_csv(RAW_DIAGNOSES, usecols=['subject_id','hadm_id','icd_code','icd_version'], chunksize=500_000):
        z = chunk[chunk.hadm_id.isin(ids)].copy()
        if len(z): pieces.append(z)
    dx = pd.concat(pieces, ignore_index=True)
    code = dx.icd_code.fillna('').astype(str).str.upper().str.replace('.', '', regex=False)
    ver9 = dx.icd_version.eq(9); ver10 = dx.icd_version.eq(10)
    three = pd.to_numeric(code.str[:3], errors='coerce')
    new_cancer = (ver10 & code.str.startswith('C')) | (ver9 & (three.between(140,208) | code.str.startswith(('2090','2091','2092','2093','2097'))))
    new_meta = (ver10 & (code.str.startswith(('C77','C78','C79','C7B')) | code.eq('C800'))) | (ver9 & (code.str.startswith(('196','197','198','2097')) | code.eq('1990')))
    old_cancer_9 = code.str[:3].isin([str(i) for i in list(range(140,166))+list(range(170,173))+list(range(174,177))+list(range(179,190))])
    old_cancer = (ver10 & code.str.startswith('C')) | (ver9 & old_cancer_9)
    old_meta = (ver10 & code.str.startswith(('C77','C78','C79','C80'))) | (ver9 & code.str.startswith(('196','197','198','199')))
    flags = dx[['subject_id','hadm_id']].copy()
    flags['old_cancer']=old_cancer; flags['old_meta']=old_meta
    flags['new_cancer']=new_cancer; flags['new_meta']=new_meta
    agg=flags.groupby(['subject_id','hadm_id'],as_index=False).max()
    result=stays.merge(agg,on=['subject_id','hadm_id'],how='left',validate='many_to_one')
    for col in ['old_cancer','old_meta','new_cancer','new_meta']:
        result[col]=result[col].fillna(False).astype(int)
    assert result.cancer.eq(result.old_cancer).all(), 'Frozen cancer flag failed raw-code reconciliation'
    assert result.metastatic_cancer.eq(result.old_meta).all(), 'Frozen metastatic flag failed raw-code reconciliation'
    result['delta_cancer']=result.new_cancer-result.old_cancer
    result['delta_meta']=result.new_meta-result.old_meta
    result['delta_score']=result.delta_cancer+result.delta_meta
    assert result.stay_id.is_unique
    return result

def correct_frame(frame, delta):
    """Update available score fields in a copy, keeping patient rows in memory only."""
    out=frame.copy()
    d=out.stay_id.map(delta.set_index('stay_id').delta_score).fillna(0).astype(int)
    dc=out.stay_id.map(delta.set_index('stay_id').delta_cancer).fillna(0).astype(int)
    dm=out.stay_id.map(delta.set_index('stay_id').delta_meta).fillna(0).astype(int)
    for name,add in [('cancer',dc),('metastatic_cancer',dm),('corrected_chronic_score',d),('lag_corrected_score',d),('corrected_score',d),('exposure_vulnerability_burden_score',d)]:
        if name in out: out[name]=out[name].astype(int)+add
    if 'documented_chronic_count' in out:out['documented_chronic_count']=out.documented_chronic_count.astype(int)+d
    if 'exposure_vulnerability_category' in out:
        s=out.exposure_vulnerability_burden_score if 'exposure_vulnerability_burden_score' in out else out.lag_corrected_score
        out['exposure_vulnerability_category']=pd.Categorical(np.select([s.le(1),s.le(3)],['low','moderate'],default='high'),categories=['low','moderate','high'])
        if 'vmod' in out:out['vmod']=out.exposure_vulnerability_category.eq('moderate').astype(int)
        if 'vhigh' in out:out['vhigh']=out.exposure_vulnerability_category.eq('high').astype(int)
    return out

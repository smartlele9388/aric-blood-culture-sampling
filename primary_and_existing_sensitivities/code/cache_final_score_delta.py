"""One-time local patient-level score-delta cache; never include it in delivery."""
from pathlib import Path
import json,sys
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'ARIC_revision_20260920'
sys.path.insert(0,str(BASE/'association_audit_20261003/code'))
from coding_rules_final import stay_flag_deltas

def main():
    risk=BASE/'secure_derived_inputs/revised_risk_day3.csv.gz'
    out=HERE.parent/'secure_derived_inputs';out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(risk,usecols=['subject_id','hadm_id','stay_id','cancer','metastatic_cancer'])
    stays=d.drop_duplicates('stay_id')
    delta=stay_flag_deltas(stays)
    public={'stays':len(delta),'any_flag_change':int((delta.delta_cancer.ne(0)|delta.delta_meta.ne(0)).sum()),
            'score_change':int(delta.delta_score.ne(0).sum()),'patient_rows_exported':False}
    delta[['stay_id','delta_cancer','delta_meta','delta_score']].to_csv(out/'final_score_deltas_stay.csv.gz',index=False)
    (HERE/'score_delta_cache_summary.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    print(json.dumps(public,indent=2))

if __name__=='__main__':main()

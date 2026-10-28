"""Verify unchanged raw-source bytes underpin reused full-window features."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--prior-stage',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);args=p.parse_args()
    extracted=json.loads((args.prior_stage/'aggregate/raw_extraction_checkpoint.json').read_text())['source_files_sha256']
    built=json.loads((args.prior_stage/'aggregate/main_risk_checkpoint.json').read_text())['input_hashes']
    references=extracted|{key:value for key,value in built.items() if key.endswith('.csv.gz')}
    paths={'procedureevents.csv.gz':'icu','microbiologyevents.csv.gz':'hosp','diagnoses_icd.csv.gz':'hosp','labevents.csv.gz':'hosp','inputevents.csv.gz':'icu','prescriptions.csv.gz':'hosp'}
    rows=[]
    for name,directory in paths.items():
        actual=digest(args.raw_root/directory/name)
        rows.append({'source':directory+'/'+name,'prior_sha256':references[name],'current_sha256':actual,'identical_raw_bytes':actual==references[name]})
        print(name+': '+('verified unchanged' if actual==references[name] else 'DIFFERENT; feature reuse must stop'),flush=True)
    out=args.output_root/'aggregate';out.mkdir(parents=True,exist_ok=True)
    report={'status':'executed','purpose':'Source equality required for reuse of raw-verified complete day-window features','all_source_hashes_match':all(row['identical_raw_bytes'] for row in rows),'sources':rows}
    (out/'S1_shared_raw_source_hash_reconciliation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['all_source_hashes_match']:raise RuntimeError('Raw source bytes differ: reuse is invalid')


if __name__=='__main__':main()

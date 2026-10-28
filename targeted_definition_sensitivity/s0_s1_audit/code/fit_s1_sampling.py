"""One S1 specimen-count Poisson sensitivity fit with original model specification."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);args=p.parse_args()
    root=args.output_root.resolve();base=root/'code/portable_baseline';sys.path.insert(0,str(base))
    source=base/'ARIC_revision_20260920/route_A/code/run_sampling_analysis.py'
    spec=importlib.util.spec_from_file_location('S1_sampling_fit',source);sampling=importlib.util.module_from_spec(spec);sys.modules[spec.name]=sampling;spec.loader.exec_module(sampling)
    if not(root/'aggregate/S1_primary_input_checkpoint.json').is_file():raise RuntimeError('S1 rebuilt input checkpoint required')
    sampling.OUT=root/'aggregate'
    x=pd.read_csv(root/'secure_derived_inputs/sampling_intervals_final_day3.csv.gz',parse_dates=['day_start','stop_datetime'])
    summary=sampling.validate_risk(x)
    effects,diagnostics=sampling.fit_poisson(x,'S1_sampling_primary_specimens','collection_count',sampling.PRIMARY)
    diagnostics['model_formula']=sampling.PRIMARY;diagnostics['offset']='log(actual interval duration_days)';diagnostics['cluster']='subject_id'
    diagnostics['definition']='S1 outcome-rule sensitivity, original S0 remains primary; no score/model optimization'
    (root/'aggregate/S1_sampling_primary_diagnostics.json').write_text(json.dumps({'risk_summary':summary,'model':diagnostics},indent=2),encoding='utf-8')
    print(f'S1 sampling primary fit complete; {len(x):,} intervals; {int(x.collection_count.sum()):,} specimens; {len(effects)} coefficients',flush=True)


if __name__=='__main__':main()

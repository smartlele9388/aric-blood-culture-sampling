"""Run only the two fixed S1 affected primary-model sensitivities."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


def run(command,log):
    with log.open('w',encoding='utf-8') as f:
        f.write('Executed command: '+repr(command)+'\n');f.flush()
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=False)
    if result.returncode:raise RuntimeError(f'Executed model failed, see {log.name}; return code {result.returncode}')
    return result.returncode


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--prior-stage',type=Path,required=True);p.add_argument('--rscript',type=Path,required=True);p.add_argument('--finalize-only',action='store_true',help='Export verified already-completed fits; never rerun a model');args=p.parse_args()
    root=args.output_root.resolve();agg=root/'aggregate';logs=root/'logs';logs.mkdir(exist_ok=True)
    if not(agg/'S1_primary_input_checkpoint.json').is_file():raise RuntimeError('Completed S1 rebuilt input required')
    sourcecheck=json.loads((agg/'S1_shared_raw_source_hash_reconciliation.json').read_text())
    if not sourcecheck['all_source_hashes_match']:raise RuntimeError('Shared-feature raw sources changed; fit prohibited')
    rbase=root/'code/portable_baseline/fit_main_cox_survival.R';rsource=root/'code/fit_S1_main_cox_survival.R'
    text=rbase.read_text().replace('raw_reproduced_main','S1_main')
    rsource.write_text('# S1 fixed-definition sensitivity: all original model options retained.\n'+text,encoding='utf-8')
    started=datetime.now().astimezone().isoformat()
    audit_root=Path(__file__).resolve().parents[1]
    commands=[([sys.executable,str(Path(__file__).with_name('fit_s1_sampling.py')),'--output-root',str(root)],audit_root/'fit_S1_sampling.log'),([str(args.rscript),str(rsource),str(root)],audit_root/'fit_S1_cox_R.log')]
    if args.finalize_only:
        for _,log in commands:
            if 'complete;' not in log.read_text():raise RuntimeError('Completed fit evidence required before export-only finalization')
        results=[0,0]
        print('Export-only finalization of two actually completed fits; no model rerun',flush=True)
    else:
        print('Starting exactly two independent fixed S1 fits: primary specimen Poisson and counting-process Cox',flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda pair:run(*pair),commands))
    poisson_npz=np.load(agg/'S1_sampling_primary_specimens_covariance.npz')
    covariance=poisson_npz['covariance']
    # The original exporter serializes NPZ terms as an object array. Obtain the
    # identical order from its coefficient CSV rather than enabling pickle.
    poisson_table=pd.read_csv(agg/'S1_sampling_primary_specimens_coefficients.csv')
    terms=poisson_table.term.astype(str).to_numpy()
    assert np.allclose(poisson_npz['coefficients'],poisson_table.log_rate_ratio,rtol=0,atol=1e-14)
    assert np.isfinite(covariance).all() and np.linalg.matrix_rank(covariance)==len(terms)
    pd.DataFrame(covariance,index=terms,columns=terms).to_csv(agg/'S1_sampling_primary_specimens_covariance.csv')
    for oldname,newname,fields,outname in [
        ('raw_reproduced_main_cox_full_precision.csv','S1_main_cox_full_precision.csv',['coefficient','robust_se','hazard_ratio','ci_lower','ci_upper','p_value'],'S0_S1_all_cox_coefficients.csv'),
        ('raw_reproduced_sampling_primary_specimens_coefficients.csv','S1_sampling_primary_specimens_coefficients.csv',['log_rate_ratio','robust_se','rate_ratio','ci_lower','ci_upper','p_value'],'S0_S1_all_sampling_coefficients.csv')]:
        old=pd.read_csv(args.prior_stage/'aggregate'/oldname);new=pd.read_csv(agg/newname)
        both=old[['term']+fields].merge(new[['term']+fields],on='term',how='outer',indicator=True,suffixes=('_S0','_S1'),validate='one_to_one')
        assert both._merge.eq('both').all(),'Model term structure unexpectedly changed'
        both.drop(columns='_merge').to_csv(agg/outname,index=False)
    diag=pd.read_csv(agg/'S1_main_cox_diagnostics.csv');assert len(diag)==1 and diag.robust_variance_min_eigenvalue.iloc[0]>0
    coeff=pd.read_csv(agg/'S1_main_cox_full_precision.csv');assert np.isfinite(coeff[['coefficient','robust_se','hazard_ratio','ci_lower','ci_upper','p_value']]).all().all()
    timing=({'finalization_started_at':started,'finalization_completed_at':datetime.now().astimezone().isoformat(),
        'fit_timing_note':'These times describe export-only finalization, not fitting. Original fits are evidenced by the two preserved fit logs and output file metadata; no fit duration is inferred.'} if args.finalize_only else {'model_run_started_at':started,'model_run_completed_at':datetime.now().astimezone().isoformat()})
    report={'status':'executed',**timing,'completed_models':2,
        'S0_remains_primary':True,'S1_definition_sensitivity':True,'additional_models':'not_triggered_by_scope',
        'Cox':'Original Surv(start,stop,event), Efron ties, patient-cluster robust covariance, no timefix; full-precision coefficients, robust/naive covariance and cox.zph exported',
        'Poisson':'Original PRIMARY RHS, actual log(duration_days) offset, patient-cluster robust covariance and original small-sample correction; full coefficient/covariance exports',
        'return_codes':results,'raw_input_checkpoint_sha256':hashlib.sha256((agg/'S1_primary_input_checkpoint.json').read_bytes()).hexdigest(),
        'export_finalization_only':args.finalize_only,
        'resolved_post_fit_export_issue':'Original NPZ term object array cannot load with allow_pickle=False; same exported CSV term order now used, no model refit' if args.finalize_only else None,
        'risk_input_sha256':hashlib.sha256((root/'secure_derived_inputs/risk_final_day3.csv.gz').read_bytes()).hexdigest(),
        'sampling_input_sha256':hashlib.sha256((root/'secure_derived_inputs/sampling_intervals_final_day3.csv.gz').read_bytes()).hexdigest(),
        'S1_R_script_sha256':hashlib.sha256(rsource.read_bytes()).hexdigest(),'S1_sampling_wrapper_sha256':hashlib.sha256(Path(__file__).with_name('fit_s1_sampling.py').read_bytes()).hexdigest()}
    (agg/'S1_two_models_execution.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'executed','completed_models':2,'S0_remains_primary':True,'aggregate_directory':str(agg)}),flush=True)


if __name__=='__main__':main()

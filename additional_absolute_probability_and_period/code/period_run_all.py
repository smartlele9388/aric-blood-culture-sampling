"""Run the frozen broad-era sensitivity once, retaining real stdout and failures."""
from __future__ import annotations
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
from datetime import datetime

def stamp():return datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--s0-stage',type=Path,required=True)
    ap.add_argument('--raw-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--rscript',type=Path,required=True);a=ap.parse_args()
    code=Path(__file__).resolve().parent;logs=a.output_root/'logs';logs.mkdir(exist_ok=True,parents=True)
    status={'status':'running','started_at':stamp(),'steps':[],
        'pre_fit_protocol':'code/period_protocol_locked.md',
        'protocol_sha256':sha(code/'period_protocol_locked.md'),
        'code_sha256':{n:sha(code/n) for n in ['period_prepare_and_poisson.py','period_fit_cox.R','period_run_all.py']}}
    target=a.output_root/'aggregate/period_execution_status.json'
    target.parent.mkdir(exist_ok=True,parents=True)
    cmds=[('prepare_and_two_poisson',[sys.executable,str(code/'period_prepare_and_poisson.py'),'--s0-stage',str(a.s0_stage),'--raw-root',str(a.raw_root),'--output-root',str(a.output_root)]),
          ('two_cox',[str(a.rscript),str(code/'period_fit_cox.R'),str(a.output_root)])]
    for name,cmd in cmds:
        entry={'step':name,'started_at':stamp(),'log':'logs/period_'+name+'.log'}
        target.write_text(json.dumps(status,indent=2),encoding='utf-8')
        with (logs/('period_'+name+'.log')).open('w',encoding='utf-8') as f:
            f.write('Started '+entry['started_at']+'\n');f.flush()
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,text=True)
            entry.update(completed_at=stamp(),exit_code=result.returncode)
            f.write('\nCompleted '+entry['completed_at']+'; exit '+str(result.returncode)+'\n')
        status['steps'].append(entry)
        if result.returncode:
            status.update(status='execution_failed',completed_at=stamp())
            target.write_text(json.dumps(status,indent=2),encoding='utf-8')
            raise SystemExit(result.returncode)
        print(name+' executed successfully',flush=True)
    status.update(status='completed',completed_at=stamp(),actual_model_fits=4)
    target.write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':main()

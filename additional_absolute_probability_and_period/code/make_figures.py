"""Render only executed aggregate results; all requested comparisons retained."""
from pathlib import Path
import argparse, json, shutil, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
QA = Path(__file__).resolve().parent / 'figure_qa_tools'
sys.path.insert(0, str(QA))
from audit_panel_alignment import require_matplotlib_panel_alignment

GROUPS = ['low', 'moderate', 'high']
COLORS = {'low':'#545454', 'moderate':'#0072B2', 'high':'#D55E00'}
MARKERS = {'low':'o','moderate':'s','high':'^'}
LABELS = {'sampling':'First blood-culture sampling', 'bsi':'First detected BSI'}
ENDPOINTS = ['sampling','bsi']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})

def normalize_crude(df):
    df = df.copy()
    aliases = {'baseline_group':'group','horizon_days':'time_days','cif':'cif_probability','probability':'cif_probability',
        'lower_95':'ci_lower','upper_95':'ci_upper',
        'lower':'ci_lower','upper':'ci_upper','ci_low':'ci_lower','ci_high':'ci_upper',
        'bootstrap_ci_lower':'ci_lower','bootstrap_ci_upper':'ci_upper'}
    df.rename(columns={k:v for k,v in aliases.items() if k in df and v not in df},inplace=True)
    if 'estimate' not in df: df['estimate'] = df['cif_probability'] * 100
    # Crude CI fields are probabilities, per the adapter contract.
    df['ci_lower'] *= 100
    df['ci_upper'] *= 100
    return df

def canvas():
    # 183 x 112 mm, expressed in inches for auditable final-size export.
    fig,axs=plt.subplots(1,2,figsize=(7.20472441,4.40944882))
    fig.subplots_adjust(left=.09,right=.985,bottom=.33,top=.83,wspace=.34)
    for i,(ax,e) in enumerate(zip(axs,ENDPOINTS)):
        ax.set_title(LABELS[e],pad=10)
        ax.annotate(chr(97+i),xy=(0,1),xycoords='axes fraction',xytext=(-25,10),
                    textcoords='offset points',fontweight='bold',fontsize=10)
        ax.tick_params(width=.7,length=3)
        ax.grid(axis='y',color='#e7e7e7',lw=.5,zorder=0)
    return fig,axs

def support(fig, crude, scale_axis='y'):
    for idx,e in enumerate(ENDPOINTS):
        z=crude[crude.endpoint.eq(e)]
        col=next((c for c in ['n_at_risk_just_before','at_risk_before','n_at_risk'] if c in z),None)
        if col:
            lines=['At risk before day 1 / 3 / 7:']
            for g in GROUPS:
                a=z[z.group.eq(g)].sort_values('time_days')
                lines.append(f"{g.capitalize()}: "+' / '.join(f'{int(v):,}' for v in a[col]))
            fig.text(.09+idx*.51,.235,'\n'.join(lines),ha='left',va='top',fontsize=7,linespacing=1.45)
    fig.text(.09,.055,'13,863 stays; 12,757 patients. Fixed groups: Low 3,046; Moderate 6,090; High 4,727.',fontsize=7)
    fig.text(.09,.018,f'Pointwise 95% patient-cluster bootstrap intervals. Day 7 is supplementary; outcome {scale_axis} scales differ.',fontsize=7)

def save(fig,stem,out):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(out/f'{stem}.alignment.json'),
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(out/f'{stem}.pdf',facecolor='white')
    fig.savefig(out/f'{stem}.svg',facecolor='white')
    fig.savefig(out/f'{stem}.png',dpi=300,facecolor='white')
    plt.close(fig)
    checks=[]
    for script,args in [
        ('audit_pdf_text.py',[str(out/f'{stem}.pdf'),'--min-pt','5','--json']),
        ('audit_figure_collisions.py',[str(out/f'{stem}.pdf'),'--json-out',str(out/f'{stem}.collision-audit.json')])]:
        run=subprocess.run([sys.executable,'-X','utf8',str(QA/script),*args],capture_output=True,text=True,encoding='utf-8')
        (out/f'{stem}.{script}.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
        checks.append({'tool':script,'exit_code':run.returncode})
        if run.returncode: raise RuntimeError(f'{stem}: {script} failed; inspect retained report')
    return checks

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--crude-only',action='store_true'); a=ap.parse_args()
    agg=a.root/'aggregate'; out=a.root/'figures'; out.mkdir(exist_ok=True)
    crude=normalize_crude(pd.read_csv(agg/'crude_cif.csv'))
    standardized=pd.read_csv(agg/'standardized_cif.csv') if not a.crude_only else None
    differences=pd.read_csv(agg/'standardized_risk_differences.csv') if not a.crude_only else None
    for df in ([crude] if a.crude_only else [crude,standardized,differences]):
        assert set(df.endpoint)==set(ENDPOINTS) and set(df.time_days)=={1,3,7}
        assert np.isfinite(df[['estimate','ci_lower','ci_upper']]).all().all()
        assert (df.ci_lower<=df.ci_upper).all()
    assert len(crude)==18
    if not a.crude_only: assert len(standardized)==18 and len(differences)==12
    audit={}
    displays=[('crude',crude,'Figure_sampling_and_BSI_crude_CIF')]
    if not a.crude_only: displays.append(('standardized',standardized,'Figure_standardized_probabilities'))
    for kind,df,stem in displays:
        fig,axs=canvas()
        curves_path=agg/'crude_curve.csv'
        curves=pd.read_csv(curves_path) if kind=='crude' and curves_path.exists() else None
        for ax,e in zip(axs,ENDPOINTS):
            for gi,g in enumerate(GROUPS):
                z=df[df.endpoint.eq(e)&df.group.eq(g)].sort_values('time_days')
                if curves is not None:
                    zz=curves[curves.endpoint.eq(e)&curves.group.eq(g)].sort_values('time_days')
                    pc=next(c for c in ['cif_probability','cif','probability'] if c in zz)
                    ax.step(zz.time_days,zz[pc]*100,where='post',color=COLORS[g],lw=1,
                            linestyle=['-','--',':'][gi],alpha=.85)
                    xs=z.time_days.to_numpy()
                else: xs=z.time_days.to_numpy()+[-.14,0,.14][gi]
                ax.errorbar(xs,z.estimate,yerr=np.array([z.estimate-z.ci_lower,z.ci_upper-z.estimate]),
                    color=COLORS[g],fmt=MARKERS[g],ms=4,elinewidth=.8,capsize=2,label=g.capitalize(),zorder=4)
            ax.set_xticks([1,3,7],['1','3','7 (supp.)']); ax.set_xlim(-.1,7.5)
            ax.set_ylim(0,df.loc[df.endpoint.eq(e),'ci_upper'].max()*1.14)
            ax.set_xlabel('Days from calendar-day-3 landmark')
            ax.set_ylabel(('Crude cumulative incidence' if kind=='crude' else 'Standardized probability')+' (%)')
        fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.995),ncol=3)
        support(fig,crude)
        df.to_csv(out/f'{stem}_source.csv',index=False)
        audit[stem]=save(fig,stem,out)
    if a.crude_only:
        print(json.dumps({'figures':1,'crude_only':True,'automated_checks_passed':True})); return
    fig,axs=canvas()
    # Horizontal differences need space for the complete supplementary-day labels.
    fig.subplots_adjust(left=.14,wspace=.40)
    for ax,e in zip(axs,ENDPOINTS):
        ax.axvline(0,color='#999999',lw=.8,linestyle='--',zorder=1)
        for gi,g in enumerate(['moderate','high']):
            z=differences[differences.endpoint.eq(e)&differences.contrast.eq(g+'_minus_low')].sort_values('time_days')
            assert len(z)==3
            yy=np.arange(3)+[-.1,.1][gi]
            ax.errorbar(z.estimate,yy,xerr=np.array([z.estimate-z.ci_lower,z.ci_upper-z.estimate]),
                color=COLORS[g],fmt=MARKERS[g],ms=4,capsize=2,elinewidth=1,label=g.capitalize()+' − Low',zorder=4)
        ax.set_yticks([0,1,2],['Day 1','Day 3','Day 7 (supp.)']); ax.invert_yaxis()
        ax.set_ylim(2.4,-.4); ax.grid(axis='y',visible=False)
        ax.set_xlabel('Standardized difference (percentage points)')
        ax.margins(x=.15)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.995),ncol=2)
    support(fig,crude,scale_axis='x')
    stem='Figure_standardized_differences'; differences.to_csv(out/f'{stem}_source.csv',index=False)
    audit[stem]=save(fig,stem,out)
    (out/'AUTOMATED_FIGURE_QA.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'figures':len(audit),'automated_checks_passed':True}))

if __name__=='__main__': main()

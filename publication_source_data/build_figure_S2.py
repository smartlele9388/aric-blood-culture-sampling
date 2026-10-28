"""Render a source-backed S0/S1 sensitivity comparison (no model fitting)."""
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = BASE / 'latest/S0_S1_MAIN_RESULTS_COMPARISON.csv'
d = pd.read_csv(SOURCE)
labels = ['Moderate vs low burden', 'High vs low burden', 'Treatment marker present']
def row_label(term):
    if 'moderate' in term: return labels[0]
    if 'high' in term: return labels[1]
    return labels[2]
rows=[]
for _, r in d[d.outcome != 'analysis totals'].iterrows():
    for definition in ['S0','S1']:
        rows.append(dict(outcome=r.outcome, label=row_label(r.term_or_metric), definition=definition,
                         term=r.term_or_metric, estimate=r[definition+'_estimate'],
                         ci_lower=r[definition+'_ci_lower'], ci_upper=r[definition+'_ci_upper'],
                         p_value=r[definition+'_p_value'],
                         source='S0_S1_MAIN_RESULTS_COMPARISON.csv'))
src=pd.DataFrame(rows)
src.to_csv(OUT/'Figure_S2_source.csv', index=False)
d[d.outcome == 'analysis totals'].to_csv(OUT/'Figure_S2_cohort_totals.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8.5,
                     'axes.labelsize':8.5,'axes.titlesize':9,
                     'xtick.labelsize':8, 'ytick.labelsize':8.5,
                     'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.linewidth':.65,'text.color':'#222222',
                     'axes.labelcolor':'#222222','xtick.color':'#444444'})
fig, axes=plt.subplots(1,2,figsize=(7.2,3.35),sharey=True)
fig.subplots_adjust(left=.275,right=.985,bottom=.22,top=.76,wspace=.18)
styles={'S0':dict(color='#255B85',marker='o',mfc='#255B85',offset=.12),
        'S1':dict(color='#B06B1F',marker='s',mfc='white',offset=-.12)}
panels=[('Blood-culture specimen IRR','a  Blood-culture sampling','Adjusted IRR (95% CI)',[.95,1.45],[1,1.1,1.2,1.3,1.4]),
        ('Detected line-timed BSI HR','b  Detected line-timed BSI','Adjusted HR (95% CI)',[.8,4.5],[1,1.5,2,3,4])]
for ax,(outcome,title,xlabel,xlim,ticks) in zip(axes,panels):
    ax.set_xscale('log'); ax.set_xlim(xlim); ax.set_ylim(-.55,2.55)
    ax.axvline(1,color='#858585',lw=.7,ls=(0,(3,3)),zorder=0)
    for yi,label in zip([2,1,0],labels):
        for definition,style in styles.items():
            r=src[(src.outcome==outcome)&(src.label==label)&(src.definition==definition)].iloc[0]
            est=float(r.estimate);lo=float(r.ci_lower);hi=float(r.ci_upper)
            assert lo <= est <= hi
            ax.errorbar(est,yi+style['offset'],xerr=[[est-lo],[hi-est]],
                        fmt=style['marker'],ms=4.5,mew=1.05,mfc=style['mfc'],mec=style['color'],
                        color=style['color'],lw=1.1,capsize=2.2,zorder=3)
    ax.set_title(title,loc='left',pad=10,fontweight='bold')
    ax.set_xlabel(xlabel,labelpad=8)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([2,1,0],labels)
    ax.tick_params(axis='y',length=0,pad=10)
    ax.tick_params(axis='x',length=3,width=.65)
    for s in ['top','right','left']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#888888')
handles=[Line2D([0],[0],color=s['color'],marker=s['marker'],mfc=s['mfc'],mec=s['color'],
                lw=1.1,markersize=4.5) for s in styles.values()]
fig.legend(handles,['S0: primary definition','S1: distinct timestamps required'],
           loc='upper center',bbox_to_anchor=(.61,.99),frameon=False,ncol=1,
           handlelength=2.2,labelspacing=.6,borderaxespad=0)
fig.text(.275,.035,'S0: 13,863 stays; 268 BSI events    S1: 13,870 stays; 263 BSI events',fontsize=7.5)
fig.savefig(OUT/'Figure_S2.pdf',facecolor='white')
fig.savefig(OUT/'Figure_S2.png',dpi=600,facecolor='white')
fig.savefig(OUT/'qa/Figure_S2_preview.png',dpi=180,facecolor='white')
plt.close(fig)

caption = '''Figure S2. Associations with blood-culture sampling and detected bloodstream infection under the primary and stricter common-commensal confirmation definitions. Panel a shows adjusted incidence rate ratios (IRRs) for blood-culture specimens; panel b shows cause-specific hazard ratios (HRs) for first detected line-timed bloodstream infection (BSI). Filled blue circles denote the primary definition (S0); open orange squares denote the sensitivity definition (S1). S1 retains the original organism, calendar-day and catheter-attribution rules and additionally requires a supporting commensal pair to have different nonmissing actual specimen identifiers and different nonmissing exact collection timestamps. No minimum elapsed-time separation was required. Cohort membership, first events, exits, risk intervals and specimen assignment were rebuilt for S1. Low, moderate and high burden correspond to scores of 0-1, 2-3 and 4 or more. The treatment marker denotes vasoactive/inotropic therapy, renal replacement treatment or total parenteral nutrition; the comparison is present versus absent. Horizontal lines show patient-clustered 95% confidence intervals, and dashed lines mark 1. Panels use different horizontal scales. Both models retain the primary covariate specifications. Estimates are descriptive comparisons without formal tests between definitions or between IRRs and HRs; neither endpoint definition is a clinical reference standard.'''
(OUT/'Figure_S2_caption.txt').write_text(caption+'\n')
print('Created Figure_S2.pdf, 600 dpi PNG, source CSV, totals CSV, caption, and preview.')

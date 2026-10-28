"""Regenerate the three score-dependent manuscript figures from final tables."""
from pathlib import Path
import json,os,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'ARIC_revision_20260920/association_audit_20261003/corrected_models_final'
R_COX=HERE.parent/'statistics_audit'/'r_batch_results'
OUT=HERE/'output';DATA=HERE/'source_data';QA=HERE/'qa'
for folder in [OUT,DATA,QA]:folder.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(QA/'matplotlib'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':8,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
                     'ytick.major.size':0,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
                     'figure.facecolor':'white','savefig.facecolor':'white','legend.frameon':False})
BLUE='#23608A';ORANGE='#B56622';GREY='#68747D';INK='#222B33'
CATMOD="C(exposure_vulnerability_category, Treatment(reference='low'))[T.moderate]"
CATHIGH="C(exposure_vulnerability_category, Treatment(reference='low'))[T.high]"
R_CATMOD='exposure_vulnerability_categorymoderate'
R_CATHIGH='exposure_vulnerability_categoryhigh'

def row(path,term):
    table=pd.read_csv(path);hit=table[table.term.eq(term)]
    if len(hit)!=1:raise ValueError(f'Missing or repeated term in {path.name}: {term}')
    return hit.iloc[0]

def entry(path,term,estimate):
    x=row(path,term)
    v=[float(x[estimate]),float(x.ci_lower),float(x.ci_upper)]
    if not (np.isfinite(v).all() and v[1]>0 and v[1]<=v[0]<=v[2]):raise ValueError(f'Invalid CI {path} {term}')
    return v

def export(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=QA/f'{name}.alignment.json',
        overlay_svg=QA/f'{name}.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png',dpi=600)
    fig.savefig(OUT/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def axis(ax,right):
    if right<=0:raise ValueError('Log-scale right bound must be positive')
    ax.set_xscale('log');ax.set_xlim(.65,right)
    ticks=[.75,1,1.5,2,3,4,6,8]
    ticks=[x for x in ticks if .65<=x<=right]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{x:g}' for x in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(1,color=GREY,lw=.8,ls='--',zorder=1)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x',color='#E7EAED',lw=.4,zorder=0)

def forest(ax,point,lo,hi,y,color,marker='o'):
    point=np.asarray(point,dtype=float);lo=np.asarray(lo,dtype=float);hi=np.asarray(hi,dtype=float)
    if not (np.isfinite(point).all() and np.isfinite(lo).all() and np.isfinite(hi).all() and
            np.all(lo>0) and np.all(lo<=point) and np.all(point<=hi)):
        raise ValueError('Invalid forest intervals')
    ax.errorbar(point,y,xerr=[point-lo,hi-point],fmt=marker,color=color,
                ecolor=color,markersize=4.7,capsize=2.7,linewidth=1.05,zorder=3)

def paired_figure(data,labels,name,size=(6.69,3.11),left=.31):
    fig,axes=plt.subplots(1,2,figsize=size,sharey=True)
    fig.subplots_adjust(left=left,right=.98,top=.81,bottom=.25,wspace=.25)
    max_hi=float(data[['IRR_high','HR_high']].max().max());right=max(4.6,max_hi*1.12)
    ys=np.arange(len(data))
    for ax,metric,color,title in zip(axes,['IRR','HR'],[BLUE,ORANGE],['a  Blood-culture sampling','b  Detected line-timed BSI']):
        forest(ax,data[metric],data[metric+'_low'],data[metric+'_high'],ys,color)
        axis(ax,right);ax.set_yticks(ys);ax.set_yticklabels(labels)
        ax.set_ylim(len(ys)-.45,-.55)
        ax.set_xlabel('Adjusted '+metric+' (95% CI)')
        ax.set_title(title,loc='left',fontsize=8.5,fontweight='bold',pad=12)
    export(fig,name)

def figure2():
    sampling=SOURCE/'sampling_primary_specimens_final_coefficients.csv'
    bsi=R_COX/'primary_day3_final_r_coefficients.csv'
    terms=[CATMOD,CATHIGH,'exposure_indication_present']
    r_terms=[R_CATMOD,R_CATHIGH,'exposure_indication_present']
    labels=['Moderate vs low burden','High vs low burden','Treatment marker present']
    records=[]
    for label,term,r_term in zip(labels,terms,r_terms):
        irr=entry(sampling,term,'rate_ratio');hr=entry(bsi,r_term,'hazard_ratio')
        records.append(dict(label=label,term=term,IRR=irr[0],IRR_low=irr[1],IRR_high=irr[2],
                            HR=hr[0],HR_low=hr[1],HR_high=hr[2],sampling_intervals=60272,BSI_events=268))
    data=pd.DataFrame(records);data.to_csv(DATA/'Figure_2_source.csv',index=False)
    paired_figure(data,labels,'Figure_2_sampling_and_BSI')

def figure3():
    sampling=SOURCE/'sampling_components_specimens_final_coefficients.csv'
    bsi=R_COX/'bsi_components_final_r_coefficients.csv'
    terms=['documented_chronic_count','exposure_mechanical_ventilation',
           'exposure_creatinine_rise_documented','exposure_lactate_high_documented','exposure_indication_present']
    sterms=['documented_chronic_count','prior_ventilation','prior_creatinine_rise','prior_high_lactate','exposure_indication_present']
    labels=['Diagnosis count, per flag','Invasive ventilation','Documented creatinine rise','Documented lactate ≥2 mmol/L','Treatment marker present']
    records=[]
    for label,sterm,bterm in zip(labels,sterms,terms):
        irr=entry(sampling,sterm,'rate_ratio');hr=entry(bsi,bterm,'hazard_ratio')
        records.append(dict(label=label,sampling_term=sterm,BSI_term=bterm,
                            IRR=irr[0],IRR_low=irr[1],IRR_high=irr[2],
                            HR=hr[0],HR_low=hr[1],HR_high=hr[2],sampling_intervals=60272,BSI_events=268))
    data=pd.DataFrame(records);data.to_csv(DATA/'Figure_3_source.csv',index=False)
    paired_figure(data,labels,'Figure_3_components',size=(6.69,4.13),left=.35)

def figure4():
    files=[('Usual window, full cohort',R_COX/'primary_day3_final_r_coefficients.csv',268,'full'),
           ('Usual window, 24 h eligible',R_COX/'bsi_lag24_matched_reference_final_r_coefficients.csv',216,'reference'),
           ('Additional 24 h lag, same risk set',R_COX/'bsi_lag24_additional_lag_final_r_coefficients.csv',216,'lag'),
           ('Usual window, 48 h eligible',R_COX/'bsi_lag48_matched_reference_final_r_coefficients.csv',175,'reference'),
           ('Additional 48 h lag, same risk set',R_COX/'bsi_lag48_additional_lag_final_r_coefficients.csv',175,'lag')]
    records=[]
    for label,file,events,kind in files:
        hr=entry(file,R_CATHIGH,'hazard_ratio')
        records.append(dict(label=label,events=events,kind=kind,HR=hr[0],HR_low=hr[1],HR_high=hr[2],model=file.stem))
    data=pd.DataFrame(records);data.to_csv(DATA/'Figure_4_source.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.69,3.98));fig.subplots_adjust(left=.48,right=.965,top=.93,bottom=.18)
    for i,r in data.iterrows():
        forest(ax,[r.HR],[r.HR_low],[r.HR_high],[i],BLUE if r.kind=='lag' else GREY,'o' if r.kind=='lag' else 's')
    axis(ax,max(5.5,float(data.HR_high.max())*1.12))
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([f'{r.label}\n{r.events} BSI events' for r in data.itertuples()])
    ax.set_ylim(len(data)-.45,-.55)
    ax.set_xlabel('High vs low burden: adjusted HR (95% CI)')
    for y in [.5,2.5]:ax.axhline(y,color='#D7DDE2',lw=.7,zorder=0)
    export(fig,'Figure_4_additional_lags')

def figure_s1():
    source=pd.read_csv(SOURCE/'competing_risk_curves_final.csv')
    source=source[source.day.eq(28)&source.group.isin(['low','moderate','high'])].copy()
    causes=['Proxy BSI','Death','ICU or hospital discharge','End of line-attribution window']
    display={'Proxy BSI':'Detected line-timed BSI','Death':'Death',
             'ICU or hospital discharge':'ICU or hospital discharge',
             'End of line-attribution window':'Attribution-window end'}
    colors={causes[0]:BLUE,causes[1]:ORANGE,causes[2]:GREY,causes[3]:'#6A5594'}
    groupnames={'low':'Low','moderate':'Moderate','high':'High'}
    rows=[]
    for cause in causes:
        for group in ['low','moderate','high']:
            selected=source[source.cause.eq(cause)&source.group.eq(group)]
            if len(selected)!=1:raise ValueError('Missing day-28 group/cause')
            r=selected.iloc[0]
            p=100*float(r.cumulative_incidence);lo=100*float(r.bootstrap_ci_lower);hi=100*float(r.bootstrap_ci_upper)
            if not (0<lo<=p<=hi<=100):raise ValueError('Invalid cumulative incidence CI')
            rows.append({'cause':cause,'group':group,'label':f'{display[cause]} — {groupnames[group]}',
                         'percent':p,'ci_lower_percent':lo,'ci_upper_percent':hi,
                         'stays':int(r.stays),'patients':int(r.patients),'bootstrap_replicates':int(r.patient_bootstrap_replicates)})
    data=pd.DataFrame(rows);data.to_csv(DATA/'Figure_S1_source.csv',index=False)
    ys=np.array([0,1,2,4,5,6,8,9,10,12,13,14])
    fig,ax=plt.subplots(figsize=(7.09,6.65));fig.subplots_adjust(left=.38,right=.77,top=.93,bottom=.10)
    for i,r in data.iterrows():
        forest(ax,[r.percent],[r.ci_lower_percent],[r.ci_upper_percent],[ys[i]],colors[r.cause],'o')
    ax.set_xscale('log');ax.set_xlim(.3,100)
    ticks=[.5,1,2,5,10,20,50,100]
    ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in ticks]))
    ax.xaxis.set_minor_locator(NullLocator());ax.grid(axis='x',color='#E7EAED',lw=.4)
    ax.set_ylim(14.6,-.7);ax.set_yticks(ys);ax.set_yticklabels(data.label)
    for y in [3,7,11]:ax.axhline(y,color='#D7DDE2',lw=.7,zorder=0)
    ax.set_xlabel('28-day first-event cumulative incidence (%)')
    ax.spines['left'].set_visible(False)
    ax.text(1.02,1.035,'Percent (95% CI)',transform=ax.transAxes,fontsize=7.5,ha='left',va='bottom')
    ymin,ymax=ax.get_ylim()
    for y,r in zip(ys,data.itertuples()):
        ax.text(1.02,(y-ymin)/(ymax-ymin),f'{r.percent:.2f} ({r.ci_lower_percent:.2f}–{r.ci_upper_percent:.2f})',
                transform=ax.transAxes,fontsize=7,ha='left',va='center')
    export(fig,'Figure_S1_competing_exits_by_baseline_group')

def main():
    figure2();figure3();figure4();figure_s1()
    audit={'source':'full-precision corrected-score clustered Poisson and independently verified R survival clustered Cox tables',
           'figures':['Figure_2_sampling_and_BSI','Figure_3_components','Figure_4_additional_lags',
                      'Figure_S1_competing_exits_by_baseline_group'],
           'plotting_backend':'Python Matplotlib','patient_rows_plotted':False,
           'interpretation_boundary':'IRR and HR are different outcome estimands, not a formal between-outcome test'}
    (QA/'figure_provenance.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()

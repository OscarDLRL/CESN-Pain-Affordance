"""Read measured baseline CSV; save IEEE-ready plots and summary."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from CLEAVER.config import OUT


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',default='grid2mm_seed42'); a=p.parse_args()
    src=OUT/'baselines'/a.run/'per_trajectory.csv'
    with src.open() as f: rows=list(csv.DictReader(f))
    dst=OUT/'figures'/'baselines'/a.run; dst.mkdir(parents=True,exist_ok=True)
    methods=['Nearest demonstration','Dense grid search','CESN L-BFGS']; ns=sorted({int(r['frames']) for r in rows}); summary=[]
    metrics=['correct','xy_error_mm','future_rmse','inference_ms','candidate_evaluations','margin']
    for m in methods:
        for n in ns:
            rr=[r for r in rows if r['method']==m and int(r['frames'])==n]; assert rr
            row=dict(method=m,frames=n,observed_percent=float(rr[0]['observed_percent']),n=len(rr))
            for k in metrics:
                v=np.array([float(r[k]) for r in rr]); v=v[np.isfinite(v)]; row[k]=float(np.mean(v)) if len(v) else np.nan; row[k+'_std']=float(np.std(v,ddof=1)) if len(v)>1 else np.nan
            summary.append(row)
    with (dst/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.labelsize':9,'legend.fontsize':7})
    for metric,ylabel,filename in [('correct','Intent accuracy (%)','01_intent_accuracy'),('xy_error_mm','Mean XY error (mm)','02_xy_error'),('future_rmse','Future latent RMSE','03_future_rmse'),('inference_ms','Mean inference time (ms)','04_inference_time'),('candidate_evaluations','Candidate evaluations','05_candidate_evaluations'),('margin','Mean signed loss margin','06_classification_margin')]:
        for early in ([False,True] if metric in ('correct','xy_error_mm','margin') else [False]):
            fig,ax=plt.subplots(figsize=(3.5,2.65))
            for m,marker in zip(methods,['s','^','o']):
                if metric=='margin' and m=='Nearest demonstration': continue # different objective scale
                rr=[r for r in summary if r['method']==m and (not early or r['frames']<=12)]
                x=np.array([r['frames' if early else 'observed_percent'] for r in rr]); y=np.array([r[metric] for r in rr]); y=y*100 if metric=='correct' else y
                ax.plot(x,y,marker=marker,markersize=3,label=m,linewidth=1.2)
            if metric=='correct': ax.axhline(50,ls=':',color='gray',label='Balanced baseline'); ax.set_ylim(0,105)
            if metric=='margin': ax.axhline(0,ls=':',color='gray')
            ax.set_xlabel('Observed frames' if early else 'Observed trajectory (%)'); ax.set_ylabel(ylabel); ax.grid(alpha=.2); ax.legend(loc='best'); fig.tight_layout()
            for ext in ('png',): fig.savefig(dst/(filename+('_early' if early else '')+'.'+ext),dpi=300,bbox_inches='tight')
            plt.close(fig)
    print('Saved:',dst)
if __name__=='__main__': main()

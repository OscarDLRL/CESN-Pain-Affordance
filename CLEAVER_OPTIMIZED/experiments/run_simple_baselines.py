#!/usr/bin/env python3
"""FINAL162 comparison. Run from repository root; no AE/CESN retraining."""
import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path.cwd(); BASE=ROOT/'CLEAVER_OPTIMIZED/outputs/final162'; PAPER=ROOT/'paper_results/final162'; OUT=PAPER/'baselines'; FIG=ROOT/'paper_figures/final162/baselines'
LAT=BASE/'autoencoder_lightweight128_matched/cleaver_lightweight_latent128.h5'
FRAMES=[1,2,3,5,10,12,23,45,67,112,157,223]; ALPHAS=[.01,.1,1.,10.,100.]

def load():
    split=json.loads((PAPER/'split_used.json').read_text()); manifest=json.loads((PAPER/'manifest.json').read_text()); norm=json.loads((PAPER/'normalization.json').read_text())
    assert hashlib.sha256(LAT.read_bytes()).hexdigest()==manifest['latent_sha256'],'Latent mismatch'
    sets={k:set(split[k]) for k in ('train','validation','test')}; assert [len(v) for v in sets.values()]==[54,9,18] and len(set.union(*sets.values()))==81
    result={k:[] for k in sets}; contacts={}
    with h5py.File(LAT,'r') as f:
        for key in sorted(f):
            if not key.startswith('trajectory_'):continue
            g=f[key]; p=int(g.attrs['pair_id']); l=int(g.attrs['pain_label']); s=np.array(g['z_static'],float); z=np.array(g['z_dynamic'],float); xy=np.array(g['target_world'],float)[:2]
            assert s.shape==(128,) and z.shape==(223,128) and np.isfinite(s).all() and np.isfinite(z).all()
            hit=np.flatnonzero(np.array(g['contact_impulses'])[:,:2].sum(1)>1e-4);assert bool(len(hit))==bool(l)
            if l:contacts[p]=int(hit[0])
            sn=next(k for k,v in sets.items() if p in v); result[sn].append((key,p,l,s,z,xy))
    data={}
    for sn,rr in result.items():
        key,p,l,s,z,xy=zip(*rr);assert len(rr)==2*len(sets[sn]);assert len(set(zip(p,l)))==len(rr)
        data[sn]=dict(key=np.array(key),pair=np.array(p),label=np.array(l),s=(np.stack(s)-norm['static_mean'])/norm['static_scale'],z=(np.stack(z)-norm['dynamic_mean'])/norm['dynamic_scale'],xy=np.stack(xy),contact=np.array([contacts[x] for x in p]))
    return data,manifest,norm

def ridge_fit(x,y,alpha):
    xm=x.mean(0);ym=y.mean(0);u=x-xm; v=y-ym
    # Dual ridge: at most 108 x 108 solve, even for flattened prefixes.
    w=u.T@np.linalg.solve(u@u.T+alpha*np.eye(len(u)),v)
    return lambda q:(q-xm)@w+ym

def features(d,k,static=False):
    s=d['s']/np.sqrt(128)
    if static:return s
    z=d['z'][:,:k].reshape(len(s),-1)/np.sqrt(k*128)
    return np.concatenate([s,z],axis=1)

def nearest(x,q):
    dist=np.maximum((q*q).sum(1)[:,None]+(x*x).sum(1)[None,:]-2*q@x.T,0)
    return dist.argmin(1)

def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    data,manifest,norm=load(); OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    tr,va=data['train'],data['validation']; rows=[]; tuning=[]; train_times=[]
    def emit(method,sn,k,d,lab,xy,future,ms):
        for i in range(len(lab)):
            rows.append(dict(method=method,split=sn,trajectory=d['key'][i],pair_id=int(d['pair'][i]),frames=k,true_label=int(d['label'][i]),pred_label=int(lab[i]),correct=int(lab[i]==d['label'][i]),xy_error_mm=float(np.linalg.norm(xy[i]-d['xy'][i])*1000),future_nrmse=None if future is None else float(np.sqrt(np.mean((future[i]-d['z'][i,k:])**2))),precontact=int(k<=d['contact'][i]),first_edge_contact_frame=int(d['contact'][i]),inference_ms=ms))
    for k in FRAMES:
        # Static-only and direct ridge: alpha selected separately on validation, no test tuning.
        for method,static in [('Static ridge',True),('Prefix ridge',False)]:
            x=features(tr,k,static);v=features(va,k,static)
            y=tr['xy']; candidates=[]
            for alpha in ALPHAS:
                reg=ridge_fit(x,y,alpha); clf=ridge_fit(x,tr['label'],alpha)
                err=np.linalg.norm(reg(v)-va['xy'],axis=1).mean();acc=np.mean((clf(v)>0.5)==va['label']);candidates.append((alpha,err,acc));tuning.append(dict(method=method,frames=k,alpha=alpha,validation_xy_mm=err*1000,validation_accuracy=acc))
            axy=min(candidates,key=lambda r:(r[1],r[0]))[0];acl=max(candidates,key=lambda r:(r[2],r[0]))[0]
            tick=time.perf_counter();reg=ridge_fit(x,y,axy);clf=ridge_fit(x,tr['label'],acl);train_times.append(dict(method=method,frames=k,alpha_xy=axy,alpha_class=acl,fit_seconds=time.perf_counter()-tick))
            for sn in ('validation','test'):
                d=data[sn];tick=time.perf_counter();q=features(d,k,static);xy=np.clip(reg(q),norm['xy_min'],norm['xy_max']);lab=(clf(q)>0.5).astype(int)
                # No dynamic observations for static-only, so paired training labels force chance performance.
                if static:lab=np.zeros(len(q),int)
                ms=(time.perf_counter()-tick)*1000/len(q);emit(method,sn,k,d,lab,xy,None,ms)
        x=features(tr,k)
        for sn in ('validation','test'):
            d=data[sn];tick=time.perf_counter();q=features(d,k);idx=nearest(x,q);lab=tr['label'][idx];xy=tr['xy'][idx];fut=tr['z'][idx,k:] if k<223 else None;ms=(time.perf_counter()-tick)*1000/len(q);emit('Nearest demonstration',sn,k,d,lab,xy,fut,ms)
            # Persistence is a forecast-only control: no context estimates.
            if k<223:
                for i in range(len(d['key'])):
                    r=float(np.sqrt(np.mean((d['z'][i,k-1:k]-d['z'][i,k:])**2)))
                    rows.append(dict(method='Persistence',split=sn,trajectory=d['key'][i],pair_id=int(d['pair'][i]),frames=k,true_label=int(d['label'][i]),pred_label=None,correct=None,xy_error_mm=None,future_nrmse=r,precontact=int(k<=d['contact'][i]),first_edge_contact_frame=int(d['contact'][i]),inference_ms=None))
        print('BASELINES',k,'frames complete',flush=True)
    # Import the frozen CESN predictions, never refit or optimize them.
    for sn in ('validation','test'):
        with (PAPER/f'{sn}_per_trajectory.csv').open() as f:
            for r in csv.DictReader(f):
                rows.append(dict(method='CESN-Vision',split=sn,trajectory=r['trajectory'],pair_id=int(r['pair_id']),frames=int(r['frames']),true_label=int(r['true_label']),pred_label=int(r['pred_label']),correct=int(r['correct']),xy_error_mm=float(r['xy_error_mm']),future_nrmse=float(r['future_rmse'])/norm['dynamic_scale'] if r['future_rmse'] else None,precontact=int(r['precontact']),first_edge_contact_frame=int(r['first_edge_contact_frame']),inference_ms=float(r['amortized_inference_ms'])))
    for sn in ('validation','test'):
        for k in FRAMES:
            rr=[r for r in rows if r['method']=='CESN-Vision' and r['split']==sn and r['frames']==k];assert {r['trajectory'] for r in rr}==set(data[sn]['key']) and len(rr)==len(data[sn]['key'])
    def avg(rr,key):
        a=[r[key] for r in rr if r[key] is not None];return float(np.mean(a)) if a else None
    summary=[]
    for method,sn,k in sorted({(r['method'],r['split'],r['frames']) for r in rows}):
        group=[r for r in rows if (r['method'],r['split'],r['frames'])==(method,sn,k)]
        for subset,rr in [('all',group),('precontact',[r for r in group if r['precontact']])]:
            if not rr:continue
            def pct(rr):
                x=avg(rr,'correct');return None if x is None else x*100
            summary.append(dict(method=method,split=sn,frames=k,observed_percent=100*k/223,subset=subset,count=len(rr),accuracy_pct=pct(rr),safe_accuracy_pct=pct([r for r in rr if r['true_label']==0]),pain_recall_pct=pct([r for r in rr if r['true_label']==1]),xy_mean_mm=avg(rr,'xy_error_mm'),future_nrmse=avg(rr,'future_nrmse'),inference_ms=avg(rr,'inference_ms')))
    write(OUT/'per_trajectory.csv',rows);write(OUT/'comparison.csv',summary);write(OUT/'validation_tuning.csv',tuning);write(OUT/'fit_times.csv',train_times)
    (OUT/'settings.json').write_text(json.dumps(dict(alphas=ALPHAS,frames=FRAMES,split=manifest['split'],latent_sha256=manifest['latent_sha256'],static_intent='always SAFE (chance on balanced pairs)',prefix_features='static and flattened observed dynamic prefix; equal block RMS weighting; original train normalization',nearest='1-NN same features',timing='CPU amortized baselines versus existing GPU CESN; not directly comparable',ridge_selection='validation accuracy and mean XY error independently; no test tuning',ridge_xy='clipped to training bounding box'),indent=2))
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    for sn in ('validation','test'):
        for metric,yl,subset in [('accuracy_pct','Intent accuracy (%)','all'),('pain_recall_pct','Precontact PAIN recall (%)','precontact'),('xy_mean_mm','Mean XY error (mm)','all'),('future_nrmse','Future latent NRMSE','all')]:
            fig,ax=plt.subplots(figsize=(3.5,2.6))
            for method in ('CESN-Vision','Static ridge','Prefix ridge','Nearest demonstration','Persistence'):
                rr=[r for r in summary if r['method']==method and r['split']==sn and r['subset']==subset and r[metric] is not None]
                if rr:ax.plot([r['observed_percent'] for r in rr],[r[metric] for r in rr],marker='o',ms=2.5,label=method)
            if metric.endswith('_pct'):ax.set_ylim(-3,103)
            ax.set(xlabel='Observed trajectory (%)',ylabel=yl);ax.grid(alpha=.2);ax.legend(fontsize=6.5,loc='best');fig.tight_layout()
            for ext in ('png','pdf'):fig.savefig(FIG/f'{sn}_{metric}.{ext}',dpi=300)
            plt.close(fig)
    print('BASELINES COMPLETE',OUT,FIG,flush=True)
if __name__=='__main__':main()

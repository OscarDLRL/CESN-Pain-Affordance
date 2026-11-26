#!/usr/bin/env python3
"""Compare context estimators on frozen FINAL162. No fitting, PNG only."""
import argparse, csv, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from CLEAVER_OPTIMIZED.cesn import model as b

ROOT=Path.cwd(); BASE=ROOT/'CLEAVER_OPTIMIZED/outputs/final162'
AE=BASE/'autoencoder_lightweight128_matched'; LAT=AE/'cleaver_lightweight_latent128.h5'; CESN=BASE/'CESN_Vision'
OUT=ROOT/'paper/results/final162/context_inversion'; FIG=ROOT/'paper/figures/final162/context_inversion'
METHODS=('L-BFGS','Dense grid search','Nearest demonstration')

def tensor(x,device): return torch.as_tensor(x,dtype=torch.float32,device=device)
def write(path,rows):
    if not rows:return
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def sync(device): b.sync(device)

def load(device):
    manifest=json.loads((CESN/'manifest.json').read_text()); assert b.digest(LAT)==manifest['latent_sha256']
    ck=torch.load(CESN/'cesn.pt',map_location='cpu',weights_only=False)
    assert ck['parameters']==manifest['parameters']==dict(reservoir=1500,radius=.6,leak=.015,gain=1.,ridge=10.,seed=42)
    data,split=b.load(LAT,AE/'split.json'); assert split==ck['split']==manifest['split']; assert [len(data[k]['keys']) for k in ('train','validation','test')]==[108,18,36]
    info=ck['normalization']; assert info==json.loads((CESN/'normalization.json').read_text())
    lo=tensor(info['xy_min'],device); hi=tensor(info['xy_max'],device)
    for d in data.values():
        d['pca']=tensor((d['static']-np.asarray(info['static_mean']))/info['static_scale'],device)
        d['yn']=tensor((d['z']-np.asarray(info['dynamic_mean']))/info['dynamic_scale'],device)
        d['xyn']=2*(tensor(d['xy'],device)-lo)/(hi-lo)-1
        d['outside_bounds']=np.any((d['xy']<np.asarray(info['xy_min'])-1e-6)|(d['xy']>np.asarray(info['xy_max'])+1e-6),axis=1)
    model=b.CESN(tuple(ck[k].to(device) for k in ('W','Wc','Wf','bias')),128,True,ck['leak'],(1,1,1,1,True,False,False))
    model.readout=ck['readout'].to(device); assert model.D==1505
    return model,data,lo,hi,info,manifest

@torch.no_grad()
def grid_search(model,d,ids,k,grid,chunk):
    # Same teacher-forced prefix objective as L-BFGS, both class hypotheses.
    obs=d['yn'][ids,:k]; static=d['pca'][ids]; B=len(ids); device=obs.device
    best=torch.full((B,),float('inf'),device=device); lab=torch.zeros(B,dtype=torch.long,device=device); xy=torch.zeros(B,2,device=device)
    for c in (0,1):
        for points in grid.split(chunk):
            R=len(points); yy=obs[:,None].expand(B,R,k,128).reshape(B*R,k,128)
            ss=static[:,None].expand(B,R,128).reshape(B*R,128)
            gg=points[None].expand(B,R,2).reshape(B*R,2)
            ctx=model.context(torch.full((B*R,),c,device=device,dtype=torch.long),ss,gg)
            loss=(model.predict(ctx,k,yy)-yy).square().mean((1,2)).reshape(B,R)
            v,j=loss.min(1); better=v<best; best=torch.where(better,v,best); lab[better]=c; xy[better]=points[j[better]]
    return lab,xy

@torch.no_grad()
def feature(d,k,ids=None):
    s=d['pca'] if ids is None else d['pca'][ids]; z=d['yn'] if ids is None else d['yn'][ids]
    return torch.cat([s/np.sqrt(128),z[:,:k].reshape(len(s),-1)/np.sqrt(128*k)],1)

@torch.no_grad()
def nearest(tr,d,ids,k,train_features):
    q=feature(d,k,ids); distances=(q[:,None]-train_features[None]).square().sum(-1); j=distances.argmin(1)
    return torch.as_tensor(tr['label'],device=q.device)[j],tr['xyn'][j]

@torch.no_grad()
def score(model,d,ids,k,lab,xy,lo,hi,ms,method,sn):
    y=d['yn'][ids]; obs=y[:,:k]; ctx=model.context(lab,d['pca'][ids],xy)
    prefix=model.predict(ctx,k,obs); future=model.predict(ctx,y.shape[1],obs)[:,k:]
    assert torch.isfinite(prefix).all() and torch.isfinite(future).all()
    pr=(prefix-obs).square().mean((1,2)).sqrt().cpu().numpy()
    fr=(future-y[:,k:]).square().mean((1,2)).sqrt().cpu().numpy() if k<y.shape[1] else [None]*len(ids)
    world=(lo+(xy+1)*.5*(hi-lo)).cpu().numpy(); lab=lab.cpu().numpy(); rows=[]
    for j,i in enumerate(ids):
        rows.append(dict(method=method,split=sn,trajectory=d['keys'][i],pair_id=int(d['pair'][i]),frames=k,true_label=int(d['label'][i]),pred_label=int(lab[j]),correct=int(lab[j]==d['label'][i]),xy_error_mm=float(np.linalg.norm(world[j]-d['xy'][i])*1000),prefix_nrmse=float(pr[j]),future_nrmse=None if fr[j] is None else float(fr[j]),precontact=int(k<=d['contact'][i]),inference_ms=ms))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--grid-mm',type=float,default=5.);ap.add_argument('--grid-batch',type=int,default=128);ap.add_argument('--device',default='cuda');a=ap.parse_args(); assert a.grid_mm>0 and a.grid_batch>0
    device=torch.device(a.device);torch.manual_seed(42);torch.set_float32_matmul_precision('high')
    model,data,lo,hi,info,manifest=load(device);frames=manifest['frames']; assert frames==[1,2,3,5,10,12,23,45,67,112,157,223]
    starts=tensor(np.random.default_rng(42).uniform(-.95,.95,(9,2)),device);starts[0]=0
    axes=[torch.linspace(-1,1,max(2,int(np.ceil(float(w)*1000/a.grid_mm-1e-5))+1),device=device) for w in hi-lo];grid=torch.cartesian_prod(*axes)
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True); cache=OUT/'cache';cache.mkdir(exist_ok=True)
    sig=dict(latent_sha256=b.digest(LAT),checkpoint_sha256=b.digest(CESN/'cesn.pt'),script_sha256=b.digest(Path(__file__)),core_sha256=b.digest(Path(b.__file__)),device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',torch_version=str(torch.__version__),grid_mm=a.grid_mm,grid_batch=a.grid_batch,grid_points=len(grid),batch=3,frames=frames,nearest_forecast='frozen CESN conditioned on retrieved intent/goal and query static latent, not retrieved future')
    sp=OUT/'settings.json'
    if sp.exists(): assert json.loads(sp.read_text())==sig,'Settings changed; use a new output directory or remove context_inversion cache intentionally.'
    else:b.atomic_json(sp,sig)
    print('Grid candidates per query:',2*len(grid),'| no training',flush=True)
    rows=[]; tr=data['train']; scale=info['dynamic_scale']
    for k in frames:
        train_features=feature(tr,k)
        for sn in ('validation','test'):
            d=data[sn]
            for i in range(0,len(d['keys']),3):
                ids=list(range(i,min(i+3,len(d['keys'])))); fp=cache/f'{sn}_{k:03d}_{i:03d}.json'
                if fp.exists(): rows.extend(json.loads(fp.read_text())); continue
                rr=[]
                # Existing L-BFGS implementation, same seed/restarts and timing boundary.
                result=b.invert(model,d,ids,k,starts,SimpleNamespace(max_iter=40),lo,hi,scale)
                for r in result:
                    rr.append(dict(method='L-BFGS',split=sn,trajectory=r['trajectory'],pair_id=r['pair_id'],frames=k,true_label=r['true_label'],pred_label=r['pred_label'],correct=r['correct'],xy_error_mm=r['xy_error_mm'],prefix_nrmse=r['prefix_rmse']/scale,future_nrmse=None if r['future_rmse'] is None else r['future_rmse']/scale,precontact=r['precontact'],inference_ms=r['amortized_inference_ms']))
                for method in METHODS[1:]:
                    sync(device);tick=time.perf_counter()
                    lab,xy=grid_search(model,d,ids,k,grid,a.grid_batch) if method==METHODS[1] else nearest(tr,d,ids,k,train_features)
                    sync(device);ms=(time.perf_counter()-tick)*1000/len(ids)
                    rr.extend(score(model,d,ids,k,lab,xy,lo,hi,ms,method,sn))
                b.atomic_json(fp,rr);rows.extend(rr)
            print(sn,k,'frames completed',flush=True)
        write(OUT/'per_trajectory.csv',rows)
    summary=[]
    def avg(rr,key):
        vals=[r[key] for r in rr if r[key] is not None];return float(np.mean(vals)) if vals else None
    for sn in ('validation','test'):
        for method in METHODS:
            for k in frames:
                allr=[r for r in rows if r['split']==sn and r['method']==method and r['frames']==k];assert len(allr)==len(data[sn]['keys'])
                pain=[r for r in allr if r['precontact'] and r['true_label']==1]
                summary.append(dict(split=sn,method=method,frames=k,observed_percent=100*k/223,count=len(allr),accuracy_pct=100*avg(allr,'correct'),precontact_pain_count=len(pain),pain_recall_pct=100*avg(pain,'correct') if pain else None,xy_mean_mm=avg(allr,'xy_error_mm'),prefix_nrmse=avg(allr,'prefix_nrmse'),future_nrmse=avg(allr,'future_nrmse'),inference_ms=avg(allr,'inference_ms')))
    write(OUT/'comparison.csv',summary)
    plt.rcParams.update({'font.size':9})
    metrics=[('accuracy_pct','Intent accuracy (%)'),('pain_recall_pct','Precontact PAIN recall (%)'),('xy_mean_mm','Mean XY error (mm)'),('prefix_nrmse','Prefix latent NRMSE'),('future_nrmse','Future latent NRMSE'),('inference_ms','Inversion time (ms/trajectory)')]
    for sn in ('validation','test'):
        fig,axs=plt.subplots(2,3,figsize=(10.4,6.0),layout='constrained')
        for ax,(metric,label) in zip(axs.flat,metrics):
            for method in METHODS:
                rr=[r for r in summary if r['split']==sn and r['method']==method and r[metric] is not None]
                ax.plot([r['observed_percent'] for r in rr],[r[metric] for r in rr],marker='o',ms=3,label=method)
            if metric in ('accuracy_pct','pain_recall_pct'):ax.set_ylim(-3,103);ax.set_xlim(0,15)
            if metric=='inference_ms':ax.set_yscale('log')
            ax.set(xlabel='Observed trajectory (%)',ylabel=label);ax.grid(alpha=.2)
        axs[0,0].legend(fontsize=7,loc='lower right');fig.savefig(FIG/f'{sn}_context_inversion.png',dpi=300);plt.close(fig)
    print('CONTEXT INVERSION COMPLETE',OUT,FIG,flush=True)
if __name__=='__main__':main()

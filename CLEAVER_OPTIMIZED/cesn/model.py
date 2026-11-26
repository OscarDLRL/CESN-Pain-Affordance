"""Four matched CESN ablations. No explicit time or future observations in inversion."""
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import h5py
import numpy as np
import torch
from scipy.sparse.linalg import eigs

EXPS = [('E20_Raw128_Absolute',128,True),('E21_NoStatic_Increment',0,True),('E22_Raw128_Increment',128,True),('E23_NoStatic_Absolute',0,True)]
OPTIONS = {k:(1,1,1,1,True,False,False) for k in ('E20','E21','E22','E23')}

def atomic_json(path, obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def atomic_pt(path,obj):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)

def csv_write(path, rows):
    if not rows:return
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load(lat, split_path):
    split=json.loads(split_path.read_text()); names=('train','validation','test')
    assert all(k in split for k in names)
    sets=[set(map(int,split[k])) for k in names]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    rows=[]; contacts={}; pairs={}
    with h5py.File(lat,'r') as f:
        for key in sorted(f):
            if not key.startswith('trajectory_'):continue
            g=f[key]; p=int(g.attrs['pair_id']); lab=int(g.attrs['pain_label'])
            s=np.asarray(g['z_static'],np.float32); z=np.asarray(g['z_dynamic'],np.float32)
            xy=np.asarray(g['target_world'],np.float32)[:2]; imp=np.asarray(g['contact_impulses'])
            assert s.shape==(128,) and z.ndim==2 and z.shape[0]==223 and xy.shape==(2,)
            assert np.isfinite(s).all() and np.isfinite(z).all() and np.isfinite(xy).all()
            assert lab in (0,1) and imp.ndim==2 and imp.shape[0]==len(z) and imp.shape[1]>=2 and np.isfinite(imp).all()
            hit=np.flatnonzero(imp[:,:2].sum(1)>1e-4)
            assert bool(len(hit))==bool(lab), key
            if lab:contacts[p]=int(hit[0])
            assert (p,lab) not in pairs; pairs[p,lab]=xy
            rows.append((key,p,lab,s,z,xy))
    assert {r[1] for r in rows}==set.union(*sets)
    for p in set.union(*sets): assert np.allclose(pairs[p,0],pairs[p,1],atol=1e-7,rtol=0)
    assert len(rows)==2*sum(map(len,sets)) and all(sets)
    result={}
    for name,ids in zip(names,sets):
        rr=[r for r in rows if r[1] in ids]
        result[name]=dict(keys=[r[0] for r in rr],pair=np.array([r[1] for r in rr]),label=np.array([r[2] for r in rr]),static=np.stack([r[3] for r in rr]),z=np.stack([r[4] for r in rr]),xy=np.stack([r[5] for r in rr]),contact=np.array([contacts[r[1]] for r in rr]))
    return result,split

def normalize(data,device):
    tr=data['train']; s=tr['static'].astype(np.float64); sm=s.mean(0)
    _,sv,v=np.linalg.svd(s-sm,full_matrices=False); pc=v[:16]; scores=(s-sm)@pc.T
    ps=max(float(scores[:,0].std()),1e-6); zm=tr['z'].astype(np.float64).mean((0,1)); zs=max(float((tr['z']-zm).std()),1e-6)
    raw_std=(s-sm).std(0); raw_floor=max(float(raw_std.max())*.01,1e-6); raw_den=np.maximum(raw_std,raw_floor)
    standardized=(s-sm)/raw_den; raw_gain=float(np.sqrt(np.mean(np.sum(scores**2,axis=1)))/ps/max(np.sqrt(np.mean(np.sum(standardized**2,axis=1))),1e-6))
    lo=tr['xy'].min(0); hi=tr['xy'].max(0); assert np.all(hi>lo)
    def t(x):return torch.as_tensor(x,dtype=torch.float32,device=device)
    for d in data.values():
        d['pca16']=t((d['static']-sm)@pc.T/ps); d['pca']=d['pca16']; d['raw_global']=t((d['static']-sm)/ps); d['raw_feature']=t((d['static']-sm)/raw_den*raw_gain); d['yn']=t((d['z']-zm)/zs); d['xyn']=t(2*(d['xy']-lo)/(hi-lo)-1)
        d['outside_bounds']=np.any((d['xy']<lo-1e-6)|(d['xy']>hi+1e-6),axis=1)
    info=dict(raw_feature_floor=raw_floor,raw_feature_denominator=raw_den.tolist(),raw_feature_gain=raw_gain,static_mean=sm.tolist(),pca_components=pc.tolist(),pca_scale=ps,pca_whitening=False,pca_component_std=scores.std(0).tolist(),pca_explained_per_component=(sv[:16]**2/(sv**2).sum()).tolist(),pca_explained=float((sv[:16]**2).sum()/(sv**2).sum()),dynamic_mean=zm.tolist(),dynamic_scale=zs,xy_min=lo.tolist(),xy_max=hi.tolist())
    return info,t(lo),t(hi)

def common_weights(n,seed,radius,device):
    rng=np.random.default_rng(seed); w=rng.uniform(-1,1,(n,n)).astype(np.float32); w*=rng.random((n,n))<.05
    if n<=32: rad=float(np.max(np.abs(np.linalg.eigvals(w))))
    else:
        try:rad=float(np.max(np.abs(eigs(w,k=1,which='LM',v0=np.ones(n),tol=1e-5,maxiter=5000,return_eigenvectors=False))))
        except Exception:rad=float(np.max(np.abs(np.linalg.eigvals(w))))
    if rad<=0:raise RuntimeError('Zero reservoir spectral radius')
    w*=radius/rad
    # Shared columns: class2, PCA16, XY2; shared feedback weights for E03/E04.
    wc=rng.uniform(-.5,.5,(20,n)).astype(np.float32); wf=rng.uniform(-.5,.5,(128,n)).astype(np.float32)/math.sqrt(128)
    bias=rng.uniform(-.1,.1,n).astype(np.float32)
    return tuple(torch.tensor(a,device=device) for a in (w,wc,wf,bias))

class CESN:
    def __init__(self,common,pca,feedback,leak,options,incremental=False):
        self.incremental=incremental
        self.W,self.Wc,self.Wf,self.bias=common; self.pca=pca; self.feedback=feedback; self.leak=leak
        self.Wc=self.Wc[list(range(2+pca))+[18,19]] if pca<=16 else self.Wc
        sg,ig,xg,fg,sr,so,interaction=options
        self.Wc=self.Wc.clone(); self.Wc[:2]*=ig; self.Wc[2:2+pca]*=sg if sr else 0; self.Wc[-2:]*=xg
        self.Wf=self.Wf*fg; self.static_out=so; self.interaction=interaction
        self.n=len(self.bias); self.C=pca+4 if so else 4; self.readout=None
        # Fixed projection uses the same random static channel, with bounded modulation.
        self.P=self.Wc[2:2+pca].clone()
        self.D=1+self.C+self.n*(2 if interaction else 1)
    def context(self,label,pca,xy):
        one=torch.nn.functional.one_hot(label.long(),2).float()
        return torch.cat([one,pca[:,:self.pca],xy],dim=-1)
    def context_out(self,c):
        return c if self.static_out else torch.cat([c[:,:2],c[:,-2:]],-1)
    def modulation(self,c):
        return torch.tanh(c[:,2:-2]@self.P) if self.interaction else None
    def features(self,c,y):
        B,T,D=y.shape; r=c.new_zeros(B,self.n); prev=c.new_zeros(B,D); base=c@self.Wc+self.bias; ones=c.new_ones(B,1); f=[]; co=self.context_out(c); mod=self.modulation(c)
        for t in range(T):
            drive=base+r@self.W+prev@self.Wf
            r=(1-self.leak)*r+self.leak*torch.tanh(drive)
            f.append(torch.cat([ones,co,r,r*mod] if mod is not None else [ones,co,r],-1)); prev=y[:,t]
        return torch.stack(f,1)
    def predict(self,c,T,obs=None):
        B=len(c); r=c.new_zeros(B,self.n); prev=c.new_zeros(B,self.Wf.shape[0]); base=c@self.Wc+self.bias
        q=self.readout; constant=q[0]+self.context_out(c)@q[1:1+self.C]; qr=q[1+self.C:1+self.C+self.n]; mod=self.modulation(c); preds=[]
        for t in range(T):
            drive=base+r@self.W+prev@self.Wf
            r=(1-self.leak)*r+self.leak*torch.tanh(drive)
            y=constant+r@qr
            if mod is not None:y=y+(r*mod)@q[-self.n:]
            if self.incremental: y=y+prev
            preds.append(y); prev=obs[:,t] if obs is not None and t<obs.shape[1] else y
        return torch.stack(preds,1)
    @torch.no_grad()
    def fit(self,d,ridge):
        y=d['yn']; c=self.context(torch.tensor(d['label'],device=y.device),d['pca'],d['xyn']); D=self.D
        targets=torch.cat([y[:,:1],y[:,1:]-y[:,:-1]],1) if self.incremental else y
        G=torch.zeros((D,D),device=y.device,dtype=torch.float64); H=torch.zeros((D,y.shape[-1]),device=y.device,dtype=torch.float64)
        for i in range(0,len(y),8):
            f=self.features(c[i:i+8],y[i:i+8]).reshape(-1,D).double(); target=targets[i:i+8].reshape(-1,y.shape[-1]).double(); G+=f.T@f; H+=f.T@target
        G.diagonal().add_(ridge); G[0,0]-=ridge; self.readout=torch.linalg.solve(G,H).float(); assert torch.isfinite(self.readout).all()
    @torch.no_grad()
    def oracle(self,d,scale):
        y=d['yn']; c=self.context(torch.tensor(d['label'],device=y.device),d['pca'],d['xyn']); out={}
        for name,obs in [('teacher_forced',y),('autonomous',None)]:
            p=self.predict(c,y.shape[1],obs); out[name+'_rmse']=float((p.double()-y.double()).square().mean().sqrt()*scale); out[name+'_max_abs_normalized']=float(p.abs().max()); out[name+'_nonfinite_count']=int((~torch.isfinite(p)).sum())
        return out

def grouped_validation(data,split):
    from scipy.spatial import Delaunay
    a,b=data['train'],data['validation']; pool={k:(a[k]+b[k] if k=='keys' else np.concatenate([a[k],b[k]],axis=0)) for k in a}
    positions=np.round(pool['xy'],5); unique=np.unique(positions,axis=0); candidates=[]
    for pos in unique:
        mask=np.all(positions==pos,axis=1); rest=np.unique(positions[~mask],axis=0)
        if len(rest)<3:continue
        try: inside=Delaunay(rest).find_simplex(pos,tol=1e-7)>=0
        except Exception:continue
        if inside:candidates.append((abs(int(mask.sum())-12),-float(np.min(np.linalg.norm(rest-pos,axis=1))),tuple(pos),mask))
    if not candidates:raise RuntimeError('No interior held-out goal available; use --validation original')
    mask=sorted(candidates,key=lambda x:x[:3])[0][3]
    for name,sel in [('train',~mask),('validation',mask)]:
        data[name]={k:([v[i] for i in np.flatnonzero(sel)] if k=='keys' else v[sel]) for k,v in pool.items()}; split[name]=sorted(set(map(int,data[name]['pair'])))
    assert not set(split['train'])&set(split['validation'])
    return data,split

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)

def invert(model,d,ids,n,starts,args,lo,hi,scale):
    device=lo.device; B=len(ids); R=len(starts); T=d['yn'].shape[1]
    obs=d['yn'][ids,:n]; static=d['pca'][ids]; full=d['yn'][ids]
    labs=torch.arange(2,device=device)[None,:,None].expand(B,2,R).reshape(-1)
    S=static.shape[-1]; ss=static[:,None,None,:].expand(B,2,R,S).reshape(B*2*R,S)
    yy=obs[:,None,None,:,:].expand(B,2,R,n,obs.shape[-1]).reshape(-1,n,obs.shape[-1])
    raw=torch.atanh(starts.clamp(-.999,.999))[None,None,:,:].expand(B,2,R,2).reshape(-1,2).clone().requires_grad_(True)
    opt=torch.optim.LBFGS([raw],lr=1.,max_iter=args.max_iter,max_eval=2*args.max_iter,history_size=20,tolerance_grad=1e-7,tolerance_change=1e-9,line_search_fn='strong_wolfe')
    best_loss=torch.full((len(raw),),float('inf'),device=device); best_xy=torch.zeros_like(raw); calls=0
    def losses():
        xy=raw.tanh(); c=model.context(labs,ss,xy); p=model.predict(c,n,yy if model.feedback else None); return (p-yy).square().mean((1,2)),xy
    def remember(loss,xy):
        with torch.no_grad():
            better=loss.detach()<best_loss; best_loss[better]=loss.detach()[better]; best_xy[better]=xy.detach()[better]
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True); loss,xy=losses(); remember(loss,xy); total=loss.sum(); total.backward(); calls+=1; return total
    sync(device); tick=time.perf_counter(); opt.step(closure)
    with torch.no_grad():
        loss,xy=losses(); remember(loss,xy); calls+=1
        bl=best_loss.view(B,2,R); bx=best_xy.view(B,2,R,2); v,j=bl.min(-1); x=bx.gather(2,j[:,:,None,None].expand(B,2,1,2)).squeeze(2); pred=v.argmin(1); idx=torch.arange(B,device=device); chosen=x[idx,pred]
        sync(device); elapsed=(time.perf_counter()-tick)*1000
        c=model.context(pred,static,chosen); forecast=model.predict(c,T,obs if model.feedback else None)
        if n<T:future=(forecast[:,n:].double()-full[:,n:].double()).square().mean((1,2)).sqrt()*scale
        else:future=torch.full((B,),float('nan'),device=device)
        world=lo+(chosen+1)*.5*(hi-lo); world=world.cpu().numpy(); vv=v.cpu().numpy(); pl=pred.cpu().numpy(); fut=future.cpu().numpy()
    assert np.isfinite(vv).all() and np.isfinite(world).all()
    rows=[]
    for j,i in enumerate(ids):
        lab=int(d['label'][i]); good,bad=map(float,(vv[j,lab],vv[j,1-lab])); tie=abs(good-bad)<=1e-10*max(1,good,bad)
        rows.append(dict(trajectory=d['keys'][i],pair_id=int(d['pair'][i]),frames=n,observed_percent=100*n/T,true_label=lab,pred_label=int(pl[j]),correct=int(lab==pl[j]),tie=int(tie),true_x_m=float(d['xy'][i,0]),true_y_m=float(d['xy'][i,1]),pred_x_m=float(world[j,0]),pred_y_m=float(world[j,1]),xy_error_mm=float(np.linalg.norm(world[j]-d['xy'][i])*1000),future_rmse=float(fut[j]) if np.isfinite(fut[j]) else None,future_nonfinite=int(n<T and not np.isfinite(fut[j])),prefix_rmse=float(math.sqrt(min(vv[j]))*scale),margin=(bad-good)/max(good+bad,1e-12),safe_loss=float(vv[j,0]),pain_loss=float(vv[j,1]),first_edge_contact_frame=int(d['contact'][i]),precontact=int(n<=d['contact'][i]),outside_train_xy_bounds=int(d['outside_bounds'][i]),batch_inference_ms=elapsed,amortized_inference_ms=elapsed/B,closure_calls=calls,candidates_per_trajectory=2*R))
    return rows

def summarize(rows):
    out=[]
    def mean(rr,k):
        a=[float(r[k]) for r in rr if r[k] is not None];return float(np.mean(a)) if a else None
    for n in sorted({r['frames'] for r in rows}):
        allr=[r for r in rows if r['frames']==n]
        for subset,rr in [('all',allr),('precontact',[r for r in allr if r['precontact']])]:
            if not rr:continue
            safe=[r for r in rr if r['true_label']==0]; pain=[r for r in rr if r['true_label']==1]
            out.append(dict(frames=n,observed_percent=rr[0]['observed_percent'],subset=subset,count=len(rr),accuracy_pct=100*mean(rr,'correct'),safe_accuracy_pct=100*mean(safe,'correct') if safe else None,pain_recall_pct=100*mean(pain,'correct') if pain else None,tie_pct=100*mean(rr,'tie'),xy_mean_mm=mean(rr,'xy_error_mm'),xy_median_mm=float(np.median([r['xy_error_mm'] for r in rr])),xy_p90_mm=float(np.quantile([r['xy_error_mm'] for r in rr],.9)),xy_within_10mm_pct=100*float(np.mean([r['xy_error_mm']<=10 for r in rr])),future_rmse=mean(rr,'future_rmse'),future_nonfinite_count=sum(r['future_nonfinite'] for r in rr),prefix_rmse=mean(rr,'prefix_rmse'),mean_margin=mean(rr,'margin'),amortized_inference_ms=mean(rr,'amortized_inference_ms')))
    return out

def plot(out,rows):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for split in sorted({r['split'] for r in rows}):
        for subset in ('all','precontact'):
            for metric,ylabel in [('accuracy_pct','Intent accuracy (%)'),('pain_recall_pct','PAIN recall (%)'),('xy_mean_mm','Mean XY error (mm)'),('future_rmse','Future latent RMSE'),('amortized_inference_ms','Amortized inversion (ms/trajectory)'),('mean_margin','Signed classification margin')]:
                fig,ax=plt.subplots(figsize=(7,4))
                for name,_,_ in EXPS:
                    rr=[r for r in rows if r['experiment']==name and r['split']==split and r['subset']==subset and r[metric] is not None]
                    if rr:ax.plot([r['observed_percent'] for r in rr],[r[metric] for r in rr],marker='o',ms=3,label=name.split('_',1)[1])
                if metric in ('accuracy_pct','pain_recall_pct'):ax.set_ylim(-3,103)
                if metric=='accuracy_pct':ax.axhline(50,color='gray',ls='--',lw=.8)
                ax.set_xlabel('Observed trajectory (%)');ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'{split}_{subset}_{metric}.png',dpi=180);plt.close(fig)

def latent_geometry(data,rgb,out,scale):
    """Matched physical conditions, train and validation only. No model inputs."""
    from itertools import combinations
    from scipy.spatial import Delaunay
    import matplotlib.pyplot as plt
    rows=[]; interp=[]; pool=[]
    with h5py.File(rgb,'r') as h:
        for split in ('train','validation'):
            d=data[split]
            for i,key in enumerate(d['keys']):
                g=h[key]; assert int(g.attrs['pair_id'])==int(d['pair'][i])
                pool.append(dict(split=split,key=key,pair=int(d['pair'][i]),label=int(d['label'][i]),start=int(g.attrs['start_id']),obj=int(g.attrs['object_id']),xy=d['xy'][i].astype(float),robot=np.asarray(g['tcp_xyz'][0])[:2],z=d['z'][i].astype(float)/scale))
    def rms(a):return float(np.sqrt(np.mean(a*a)))
    for a,b in combinations(pool,2):
        if a['pair']==b['pair'] and a['label']!=b['label']: kind='different_intent_same_pair'; dist=0.
        elif a['label']==b['label'] and a['start']==b['start'] and a['obj']!=b['obj']:kind='different_goal_same_start_intent';dist=float(np.linalg.norm(a['xy']-b['xy'])*1000)
        elif a['label']==b['label'] and a['obj']==b['obj'] and a['start']!=b['start']:kind='different_start_same_goal_intent';dist=float(np.linalg.norm(a['robot']-b['robot'])*1000)
        else:continue
        split='train' if a['split']==b['split']=='train' else ('validation' if a['split']==b['split']=='validation' else 'cross')
        for k in (5,10,23,67,223):
            x,y=a['z'][:k],b['z'][:k]; rows.append(dict(kind=kind,split=split,a=a['key'],b=b['key'],frames=k,distance_mm=dist,absolute_rms=rms(x-y),increment_rms=rms(np.diff(x,axis=0)-np.diff(y,axis=0))))
    for a in pool:
        if a['split']!='validation':continue
        tr=[b for b in pool if b['split']=='train' and b['start']==a['start'] and b['label']==a['label']]
        if len(tr)<3:continue
        pts=np.stack([b['xy'] for b in tr]);
        try:
            tri=Delaunay(pts);j=int(tri.find_simplex(a['xy'],tol=1e-7))
        except Exception:continue
        if j<0:continue
        bary=tri.transform[j,:2]@(a['xy']-tri.transform[j,2]);weights=np.r_[bary,1-bary.sum()];z=np.einsum('i,itd->td',weights,np.stack([tr[i]['z'] for i in tri.simplices[j]]));near=tr[int(np.linalg.norm(pts-a['xy'],axis=1).argmin())]['z']
        interp.append(dict(trajectory=a['key'],label=a['label'],absolute_interp_rms=rms(z-a['z']),absolute_nearest_rms=rms(near-a['z']),increment_interp_rms=rms(np.diff(z-a['z'],axis=0)),increment_nearest_rms=rms(np.diff(near-a['z'],axis=0)),nearest_mm=float(np.linalg.norm(pts-a['xy'],axis=1).min()*1000)))
    csv_write(out/'latent_matched_distances.csv',rows);csv_write(out/'latent_spatial_interpolation.csv',interp)
    summary=[]
    for kind,split,k in sorted({(r['kind'],r['split'],r['frames']) for r in rows}):
        rr=[r for r in rows if (r['kind'],r['split'],r['frames'])==(kind,split,k)];summary.append(dict(kind=kind,split=split,frames=k,count=len(rr),mean_absolute_rms=float(np.mean([r['absolute_rms'] for r in rr])),mean_increment_rms=float(np.mean([r['increment_rms'] for r in rr]))))
    csv_write(out/'latent_distance_summary.csv',summary)
    if interp: print('LATENT INTERPOLATION',len(interp),'covered validation trajectories', {k:float(np.mean([r[k] for r in interp])) for k in ('absolute_interp_rms','absolute_nearest_rms','increment_interp_rms','increment_nearest_rms')},flush=True)
    else:print('LATENT INTERPOLATION: no matched validation points inside local hull',flush=True)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for kind in sorted({r['kind'] for r in rows}):
        rr=[r for r in summary if r['kind']==kind and r['split']=='train'];
        for a,key in zip(ax,('mean_absolute_rms','mean_increment_rms')):a.plot([r['frames'] for r in rr],[r[key] for r in rr],marker='o',label=kind);a.set(xlabel='Observed frames',ylabel=key);a.legend(fontsize=7);a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'latent_distances.png',dpi=160);plt.close(fig)


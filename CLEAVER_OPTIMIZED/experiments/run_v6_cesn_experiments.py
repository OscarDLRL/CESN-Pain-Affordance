"""Matched V6 CESN experiments. Run from repo root. No AE training."""
import argparse, json, math, inspect
from pathlib import Path
from types import SimpleNamespace
import h5py, numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from CLEAVER_OPTIMIZED.experiments import train_v6_compact128 as v
from CLEAVER_OPTIMIZED.experiments.run_v6_shared_encoder import SharedV6
from CLEAVER_OPTIMIZED.experiments.run_v6_residual_ablation import reconstruct
from CLEAVER_OPTIMIZED.experiments.run_dimensions import normalize
from CLEAVER_OPTIMIZED.experiments.run_forward_lowleak import make_model
from CLEAVER_OPTIMIZED.experiments.run_context_inversion_comparison import grid_search, feature, nearest
from CLEAVER_OPTIMIZED.cesn import model as b
from CLEAVER_OPTIMIZED.experiments.run_v6_object_loss import projected_mask
ROOT=Path.cwd(); BASE=ROOT/'CLEAVER_OPTIMIZED/outputs'; OLD=BASE/'final162'
DATA=ROOT/'datasets/cleaver_extended_workspace_162_complete/cleaver_same_goal_162.h5'
SPLIT=ROOT/'CLEAVER_OPTIMIZED/split.json'
FRAMES=[1,2,3,5,10,12,23,45,67,112,157,223]
MODES={'dual_encoder':OLD/'v6_residual_ablation/V6-AL.pt','shared_encoder':OLD/'v6_shared_encoder/V6-SE.pt'}
DEV=torch.device('cuda'); N=1500; STEP=N*N+N*128+N*128; CONTEXT=132*N+4*128

def t(x):return torch.as_tensor(x,device=DEV,dtype=torch.float32)
def rgb(x):return v.device_rgb(x)
def save_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(p,index=False)
def decode(a,x,s,z):return reconstruct(a,x,z,s,None,True)['frame_recon']

def load_ae(arch):
    ck=torch.load(MODES[arch],map_location='cpu',weights_only=False)
    assert ck['reconstruction']=='alpha'
    m=SharedV6('V6-SE') if arch=='shared_encoder' else v.V6Compact128()
    m.load_state_dict(ck['model_state_dict'],strict=True); m=m.cuda().eval().requires_grad_(False)
    sp=json.loads(SPLIT.read_text()); assert all(sorted(ck[k])==sorted(sp[s]) for k,s in [('final_train_pairs','train'),('validation_pairs','validation'),('test_pairs','test')])
    return m

@torch.no_grad()
def export(a,source,dest,batch):
    if dest.exists():return
    tmp=dest.with_suffix('.tmp')
    with h5py.File(source,'r') as f,h5py.File(tmp,'w') as o:
        keys=sorted(k for k in f if k.startswith('trajectory_')); assert keys,source
        for i,key in enumerate(keys):
            g=f[key]; im=g['rgb']; x=rgb(im[:1]); s=a.encode_static(x); zz=[]
            for j in range(0,len(im),batch): zz.append(a.encode_dynamic(rgb(im[j:j+batch])-x).cpu().numpy())
            h=o.create_group(key)
            for k,val in g.attrs.items():h.attrs[k]=val
            h.attrs['pain_label']=int(g.attrs.get('label',g.attrs.get('pain_label',-1)))
            for k in ('target_world','contact_impulses'):h.create_dataset(k,data=g[k][()])
            h.create_dataset('z_static',data=s[0].cpu().numpy());h.create_dataset('z_dynamic',data=np.concatenate(zz))
            if i%20==0:print('EXPORT',dest.name,i+1,flush=True)
    tmp.replace(dest)

def external(lat,info):
    rows=[]; contacts={}
    with h5py.File(lat,'r') as f:
        for key in sorted(f):
            g=f[key]; p=int(g.attrs['pair_id']); lab=int(g.attrs['pain_label']); hit=np.flatnonzero(g['contact_impulses'][:,:2].sum(1)>1e-4)
            assert lab in (0,1) and bool(len(hit))==bool(lab),(key,lab)
            if lab:contacts[p]=int(hit[0])
            rows.append((key,p,lab,g['z_static'][:],g['z_dynamic'][:],g['target_world'][:2]))
    assert rows and all(r[1] in contacts for r in rows)
    d=dict(keys=[r[0] for r in rows],pair=np.array([r[1] for r in rows]),label=np.array([r[2] for r in rows]),static=np.stack([r[3] for r in rows]),z=np.stack([r[4] for r in rows]),xy=np.stack([r[5] for r in rows]),contact=np.array([contacts[r[1]] for r in rows]))
    lo=np.array(info['xy_min']); hi=np.array(info['xy_max']); d.update(pca=t((d['static']-info['static_mean'])/info['static_scale']),yn=t((d['z']-info['dynamic_mean'])/info['dynamic_scale']),xyn=t(2*(d['xy']-lo)/(hi-lo)-1),outside_bounds=np.any((d['xy']<lo-1e-6)|(d['xy']>hi+1e-6),1));return d

def fit(data,folder,nofb=False):
    weights=b.common_weights(N,42,1.,DEV); m=make_model(weights,N,.6,.015,1.,42,DEV)
    if nofb:m.Wf.zero_();m.feedback=False
    m.fit(data['train'],10.)
    b.atomic_pt(folder/('cesn_no_feedback.pt' if nofb else 'cesn.pt'),{k:getattr(m,k).cpu() for k in ('W','Wc','Wf','bias','readout')});return m

@torch.no_grad()
def visual_cost(a):
    counts=[0]; hooks=[]
    def hook(m,inp,out):
        if isinstance(m,torch.nn.Conv2d):counts[0]+=out.numel()*m.in_channels//m.groups*math.prod(m.kernel_size)
        elif isinstance(m,torch.nn.Linear):counts[0]+=out.numel()*m.in_features
    for m in a.modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):hooks.append(m.register_forward_hook(hook))
    x=torch.zeros(1,3,256,256,device=DEV); s=a.encode_static(x); static=counts[0];counts[0]=0;z=a.encode_dynamic(x); enc=counts[0];counts[0]=0;decode(a,x,s,z);dec=counts[0]
    for h in hooks:h.remove()
    return dict(static_encoder_macs=static,dynamic_encoder_macs=enc,decoder_macs=dec,parameters=sum(p.numel() for p in a.parameters()),inference_parameters=sum(p.numel() for p in a.parameters())-sum(p.numel() for p in a.static_aux_decoder.parameters())-sum(p.numel() for p in a.motion_centroid_head.parameters()))

def infer(m,d,ids,k,starts,lo,hi,scale,method,grid,tr,args):
    if method=='L-BFGS':
        rr=b.invert(m,d,ids,k,starts,SimpleNamespace(max_iter=args.max_iter),lo,hi,scale)
        lab=torch.tensor([r['pred_label'] for r in rr],device=DEV);xy=2*(t([[r['pred_x_m'],r['pred_y_m']] for r in rr])-lo)/(hi-lo)-1
        ev=np.array([r['closure_calls']*r['candidates_per_trajectory'] for r in rr]); mac=ev*(CONTEXT+k*STEP); backward=np.array([(r['closure_calls']-1)*r['candidates_per_trajectory']*(CONTEXT+k*(N*N+N*128)) for r in rr]);return lab,xy,ev,mac,backward
    if method=='Grid':
        lab,xy=grid_search(m,d,ids,k,grid,args.grid_batch);ev=np.full(len(ids),2*len(grid));return lab,xy,ev,ev*(CONTEXT+k*STEP),np.zeros(len(ids),dtype=np.int64)
    lab,xy=nearest(tr,d,ids,k,feature(tr,k));return lab,xy,np.zeros(len(ids),dtype=np.int64),np.zeros(len(ids),dtype=np.int64),np.zeros(len(ids),dtype=np.int64)

@torch.no_grad()
def rgb_error(a,d,ids,p,k,source,info,batch):
    # Common RGB metric on every fifth future frame, independent of latent basis.
    ts=np.arange(k,p.shape[1],5); out=[]
    if not len(ts):return [(None,None)]*len(ids)
    for j,i in enumerate(ids):
        g=source[d['keys'][i]]; x=rgb(g['rgb'][:1]); s=t(d['static'][i:i+1]); err=[]; roi=[]
        for tt in np.array_split(ts,max(1,math.ceil(len(ts)/batch))):
            z=p[j,tt]*info['dynamic_scale']+t(info['dynamic_mean']); y=decode(a,x.expand(len(tt),-1,-1,-1),s.expand(len(tt),-1),z); target=rgb(g['rgb'][tt]);e=(y-target).square();err.extend(e.mean((1,2,3)).cpu().tolist())
            if 'object_xyz' in g and 'object_quat' in g:
                mask=t(np.stack([projected_mask(g['object_xyz'][int(u)],g['object_quat'][int(u)]) for u in tt]))[:,None]
                roi.extend(((e*mask).sum((1,2,3))/(3*mask.sum((1,2,3)).clamp_min(1))).cpu().tolist())
        out.append((float(np.mean(err)),float(np.mean(roi)) if roi else None))
    return out

@torch.no_grad()
def rollout(m,a,c,obs,T,x,s,info,alpha,period):
    B=len(c);r=c.new_zeros(B,N);prev=c.new_zeros(B,128);base=c@m.Wc+m.bias;q=m.readout;constant=q[0]+m.context_out(c)@q[1:5];qr=q[5:];out=[]; calls=0
    mu=t(info['dynamic_mean']); scale=info['dynamic_scale']
    for j in range(T):
        r=(1-m.leak)*r+m.leak*torch.tanh(base+r@m.W+prev@m.Wf);y=constant+r@qr;out.append(y)
        if j<obs.shape[1]:prev=obs[:,j]
        else:
            prev=y
            if alpha and (j-obs.shape[1]+1)%period==0 and j<T-1:
                z=a.encode_dynamic(decode(a,x,s,y*scale+mu)-x);prev=(1-alpha)*y+alpha*(z-mu)/scale;calls+=1
    return torch.stack(out,1),calls

def summary(rows):
    df=pd.DataFrame(rows); result=[]
    for (sn,method,k),g in df.groupby(['split','method','frames']):
        pain=g[(g.true_label==1)&(g.precontact==1)];r=dict(split=sn,method=method,frames=k,accuracy_pct=100*g.correct.mean(),pain_recall_pct=100*pain.correct.mean() if len(pain) else np.nan)
        for col in ['xy_error_mm','future_nrmse','rgb_mse','object_roi_mse','forward_macs','backward_matrix_macs_est','forecast_macs','observation_encoder_macs','candidate_evaluations','nearest_scalar_ops','estimated_arithmetic_ops']:
            if col in g:r[col]=g[col].mean()
        result.append(r)
    return result

def plots(folder,rows):
    df=pd.DataFrame(rows)
    for sn,g in df.groupby('split'):
        fig,axs=plt.subplots(2,3,figsize=(12,7)); cols=['accuracy_pct','pain_recall_pct','xy_error_mm','future_nrmse','rgb_mse','estimated_arithmetic_ops']; labels=['Intent accuracy (%)','Precontact PAIN recall (%)','XY error (mm)','Future latent NRMSE','Future RGB MSE','Estimated inversion / rollout ops']
        for ax,col,label in zip(axs.flat,cols,labels):
            for name,h in g.groupby('method'):
                h=h.sort_values('frames');ax.plot(h.frames,h[col],marker='.',label=name)
            ax.set(xlabel='Observed frames',ylabel=label);ax.grid(alpha=.2)
            if col=='estimated_arithmetic_ops':ax.set_yscale('symlog',linthresh=1e6)
        axs[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(folder/f'{sn}.png',dpi=300);plt.close(fig)

def run_arch(arch,args):
    out=BASE/arch
    if args.smoke:out=out/'smoke'
    for p in ['autoencoder','latents','cesn_one_shot/models','cesn_one_shot/context_inversion','cesn_one_shot/offgrid_interpolation','cesn_feedback/internal_feedback','cesn_feedback/periodic_observation']:(out/p).mkdir(parents=True,exist_ok=True)
    a=load_ae(arch);cost=visual_cost(a); sources={'main':DATA}
    for path in args.offgrid:
        pp=Path(path);name=pp.stem
        if name in sources:name=pp.parent.name+'_'+name
        assert name not in sources,name
        sources[name]=pp
    assert len(sources)==1+len(args.offgrid),'Offgrid dataset stems must be unique'
    sig=dict(checkpoint=b.digest(MODES[arch]),split=b.digest(SPLIT),script=b.digest(Path(__file__)),datasets={k:b.digest(p) for k,p in sources.items()},settings={k:v for k,v in vars(args).items() if k!='arch'},cost=cost,dependencies={Path(inspect.getfile(x)).name:b.digest(Path(inspect.getfile(x))) for x in (v,b,SharedV6,reconstruct,normalize,make_model,grid_search,projected_mask)},cost_convention='dense Conv2d/Linear and CESN matrix products; backward MACs estimated for fixed-weight context gradients through recurrence/readout (not full autograd profiler); 2 ops/MAC, nearest distance scalar ops; excludes elementwise nonlinearities, optimizer bookkeeping, memory traffic and RGB scoring; no timing')
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==sig,'Run settings changed; use a fresh directory'
    else:b.atomic_json(mp,sig)
    b.atomic_json(out/'autoencoder/checkpoint_reference.json',dict(path=str(MODES[arch].relative_to(ROOT)),sha256=sig['checkpoint'],reconstruction='alpha',mode=arch))
    for name,source in sources.items():export(a,source,out/'latents'/f'{name}.h5',args.batch_size)
    data,sp=b.load(out/'latents/main.h5',SPLIT);info,lo,hi=normalize(data,DEV);b.atomic_json(out/'cesn_one_shot/models/normalization.json',info);b.atomic_json(out/'cesn_one_shot/models/split.json',sp)
    m=fit(data,out/'cesn_one_shot/models');nf=fit(data,out/'cesn_one_shot/models',True);tr=data['train'];starts=t(np.random.default_rng(42).uniform(-.95,.95,(9,2)));starts[0]=0
    grid=torch.cartesian_prod(*[torch.linspace(-1,1,max(2,math.ceil(float(w)*1000/args.grid_mm)+1),device=DEV) for w in hi-lo]); frames=[5,23] if args.smoke else FRAMES
    sets={sn:(data[sn],DATA) for sn in ('validation','test')}
    for name,source in sources.items():
        if name!='main':
            ext=external(out/'latents'/f'{name}.h5',info)
            from scipy.spatial import Delaunay
            hull=Delaunay(np.unique(tr['xy'],axis=0)); assert np.all(hull.find_simplex(ext['xy'],tol=1e-6)>=0),f'{name}: outside training hull'
            dist=np.linalg.norm(ext['xy'][:,None]-data['train']['xy'][None],axis=-1).min(1)
            assert np.all(dist>1e-5),f'{name}: includes training goal'
            with h5py.File(DATA,'r') as f:
                known=np.stack([f[key]['requested_start_xyz'][:] for key in tr['keys']])
            with h5py.File(source,'r') as f:
                if all('requested_start_xyz' in f[key] for key in ext['keys']):
                    st=np.stack([f[key]['requested_start_xyz'][:] for key in ext['keys']]); isknown=np.linalg.norm(st[:,None]-known[None],axis=-1).min(1)<1e-5
                else:isknown=None
            if isknown is None:sets[name]=(ext,source)
            else:
                for label,mask in [('known_start',isknown),('new_start',~isknown)]:
                    ii=np.flatnonzero(mask)
                    if len(ii):sets[name+'_'+label]=({k:([val[i] for i in ii] if k=='keys' else val[ii.tolist()]) for k,val in ext.items()},source)
    allone=[]; allfb=[]; alarms=[]
    for sn,(d,path) in sets.items():
        print(arch,sn,flush=True); isext=sn not in ('validation','test'); folder=out/'cesn_one_shot'/('offgrid_interpolation' if isext else 'context_inversion');cache=folder/'cache';cache.mkdir(exist_ok=True); rows=[]; fb=[]; periodic=[]
        size=min(2,len(d['keys'])) if args.smoke else len(d['keys']); T=d['yn'].shape[1]
        with h5py.File(path,'r') as source:
            for k in frames:
                for start in range(0,size,3):
                    ids=list(range(start,min(start+3,size))); fp=cache/f'{sn}_{k}_{start}.json'
                    if fp.exists():chunk=json.loads(fp.read_text());rows.extend(chunk['one']);fb.extend(chunk['feedback']);continue
                    one=[]; internal=[];y=d['yn'][ids];obs=y[:,:k]
                    for method in ('L-BFGS','Grid','Nearest'):
                        lab,xy,ev,mac,back=infer(m,d,ids,k,starts,lo,hi,info['dynamic_scale'],method,grid,tr,args)
                        with torch.no_grad():
                            c=m.context(lab,d['pca'][ids],xy);p=m.predict(c,T,obs);world=lo+(xy+1)*(hi-lo)/2; re=rgb_error(a,d,ids,p,k,source,info,args.batch_size)
                            for j,i in enumerate(ids):
                                base=dict(split=sn,trajectory=d['keys'][i],frames=k,true_label=int(d['label'][i]),pred_label=int(lab[j]),correct=int(lab[j]==d['label'][i]),precontact=int(k<=d['contact'][i]),xy_error_mm=float((world[j]-t(d['xy'][i])).norm()),future_nrmse=float((p[j,k:]-y[j,k:]).square().mean().sqrt()) if k<T else None,rgb_mse=re[j][0],object_roi_mse=re[j][1]);base['xy_error_mm']*=1000
                                near=3*len(tr['keys'])*128*(k+1) if method=='Nearest' else 0
                                one.append(dict(base,method=method,forward_macs=int(mac[j]),candidate_evaluations=int(ev[j]),nearest_scalar_ops=near,backward_matrix_macs_est=int(back[j]),forecast_macs=CONTEXT+T*STEP,observation_encoder_macs=cost['static_encoder_macs']+k*cost['dynamic_encoder_macs'],estimated_arithmetic_ops=2*(int(mac[j])+int(back[j]))+near))
                            if method=='L-BFGS' and k in (5,12,23,45) and k<T:
                                x=rgb(np.stack([source[d['keys'][i]]['rgb'][0] for i in ids]));s=t(d['static'][ids]); variants=[('None',0,1,nf),('Latent',0,1,m),('Visual',1,1,m),('Hybrid .1',.1,1,m),('Hybrid .25',.25,1,m),('Hybrid .5',.5,1,m),('Visual /5',1,5,m),('Visual /10',1,10,m),('Visual /20',1,20,m)]
                                for name,alpha,period,model in variants:
                                    pp,calls=rollout(model,a,c,obs,T,x,s,info,alpha,period); re2=rgb_error(a,d,ids,pp,k,source,info,args.batch_size)
                                    for j,i in enumerate(ids):
                                        rr=one[j].copy();rr.update(method=name,future_nrmse=float((pp[j,k:]-y[j,k:]).square().mean().sqrt()),rgb_mse=re2[j][0],object_roi_mse=re2[j][1],forward_macs=CONTEXT+T*STEP+calls*(cost['decoder_macs']+cost['dynamic_encoder_macs']),candidate_evaluations=0,nearest_scalar_ops=0,backward_matrix_macs_est=0,forecast_macs=CONTEXT+T*STEP,estimated_arithmetic_ops=2*(CONTEXT+T*STEP+calls*(cost['decoder_macs']+cost['dynamic_encoder_macs'])));internal.append(rr)
                    b.atomic_json(fp,dict(one=one,feedback=internal));rows.extend(one);fb.extend(internal)
                print(arch,sn,'prefix',k,flush=True)
            # Streaming replay: batches of real observations arrive at intervals; cold-start inversion, no future leakage.
            if not isext:
                pdir=out/'cesn_feedback/periodic_observation';pc=pdir/'cache';pc.mkdir(exist_ok=True)
                intervals=[5] if args.smoke else args.intervals
                for interval in intervals:
                    for start in range(0,size,3):
                        ids=list(range(start,min(start+3,size))); fp=pc/f'{sn}_{interval}_{start}.json'
                        if fp.exists():periodic.extend(json.loads(fp.read_text()));continue
                        first=[None]*len(ids);total=[0]*len(ids); backward=[0]*len(ids); updates=[0]*len(ids); last=[None]*len(ids)
                        for k in range(interval,min(T,30 if args.smoke else T)+1,interval):
                            active=[j for j,i in enumerate(ids) if first[j] is None and (d['label'][i]==0 or k<=d['contact'][i])]
                            if not active:break
                            use=[ids[j] for j in active];lab,xy,ev,mac,back=infer(m,d,use,k,starts,lo,hi,info['dynamic_scale'],'L-BFGS',grid,tr,args)
                            for l,j in enumerate(active):
                                total[j]+=int(mac[l]);backward[j]+=int(back[l]);updates[j]+=1;last[j]=k
                                if int(lab[l])==1:first[j]=k-1
                        rr=[]
                        for j,i in enumerate(ids):
                            pain=int(d['label'][i]);alarm=first[j];contact=int(d['contact'][i]) if pain else None
                            rr.append(dict(split=sn,trajectory=d['keys'][i],policy='periodic',interval=interval,true_label=pain,alarm_frame=alarm,contact_frame=contact,early_detection=int(pain and alarm is not None and alarm<contact),false_stop=int(not pain and alarm is not None),lead_frames=contact-alarm if pain and alarm is not None else None,updates=updates[j],forward_macs=total[j],backward_matrix_macs_est=backward[j],estimated_arithmetic_ops=2*(total[j]+backward[j]),observation_encoder_macs=cost['static_encoder_macs']+(last[j] or 0)*cost['dynamic_encoder_macs']))
                        b.atomic_json(fp,rr);periodic.extend(rr)
                    print(arch,sn,'periodic',interval,flush=True)
        # Matched one-shot alarm policies, evaluated only once at the stated prefix.
        if not isext:
            for r in rows:
                if r['method']!='L-BFGS' or r['frames'] not in (5,12,23):continue
                i=d['keys'].index(r['trajectory']);k=r['frames'];pain=r['true_label'];contact=int(d['contact'][i]) if pain else None
                alarm=k-1 if r['pred_label']==1 else None
                periodic.append(dict(split=sn,trajectory=r['trajectory'],policy='one_shot',interval=k,true_label=pain,alarm_frame=alarm,contact_frame=contact,early_detection=int(pain and alarm is not None and alarm<contact),false_stop=int(not pain and alarm is not None),lead_frames=contact-alarm if pain and alarm is not None and alarm<contact else None,updates=1,forward_macs=r['forward_macs'],backward_matrix_macs_est=r['backward_matrix_macs_est'],estimated_arithmetic_ops=r['estimated_arithmetic_ops'],observation_encoder_macs=cost['static_encoder_macs']+k*cost['dynamic_encoder_macs']))
        allone.extend(rows);allfb.extend(fb);alarms.extend(periodic)
    for typ,rr in [('context_inversion',[r for r in allone if r['split'] in ('validation','test')]),('offgrid_interpolation',[r for r in allone if r['split'] not in ('validation','test')])]:
        if rr:
            folder=out/'cesn_one_shot'/typ;ss=summary(rr);save_csv(folder/'per_trajectory.csv',rr);save_csv(folder/'comparison.csv',ss);plots(folder,ss)
    if allfb:
        folder=out/'cesn_feedback/internal_feedback';ss=summary(allfb);save_csv(folder/'per_trajectory.csv',allfb);save_csv(folder/'comparison.csv',ss);plots(folder,ss)
    if alarms:
        folder=out/'cesn_feedback/periodic_observation';save_csv(folder/'per_trajectory.csv',alarms);df=pd.DataFrame(alarms);s=[]
        for (sn,policy,k),g in df.groupby(['split','policy','interval']):
            p=g[g.true_label==1];safe=g[g.true_label==0];s.append(dict(split=sn,policy=policy,interval=k,early_detection_pct=100*p.early_detection.mean(),false_stop_pct=100*safe.false_stop.mean(),lead_frames=p.lead_frames.mean(),forward_macs=g.forward_macs.mean(),backward_matrix_macs_est=g.backward_matrix_macs_est.mean(),estimated_arithmetic_ops=g.estimated_arithmetic_ops.mean(),observation_encoder_macs=g.observation_encoder_macs.mean()))
        save_csv(folder/'comparison.csv',s);df=pd.DataFrame(s)
        for sn,g in df.groupby('split'):
            fig,ax=plt.subplots(2,2,figsize=(9,6));
            for aa,col in zip(ax.flat,['early_detection_pct','false_stop_pct','lead_frames','estimated_arithmetic_ops']):
                for policy,h in g.groupby('policy'):aa.plot(h.interval,h[col],marker='o',label=policy)
                aa.set(xlabel='Prefix / update interval (frames)',ylabel=col);aa.grid(alpha=.2)
                if col=='estimated_arithmetic_ops':aa.set_yscale('log')
            ax[0,0].legend()
            fig.tight_layout();fig.savefig(folder/f'{sn}.png',dpi=300);plt.close(fig)
    print('COMPLETE',arch,out,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arch',choices=['both',*MODES],default='both');p.add_argument('--offgrid',nargs='*',default=[]);p.add_argument('--offgrid-auto',action='store_true');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--grid-mm',type=float,default=5);p.add_argument('--grid-batch',type=int,default=64);p.add_argument('--max-iter',type=int,default=40);p.add_argument('--intervals',nargs='+',type=int,default=[1,5,10,20]);p.add_argument('--smoke',action='store_true');p.add_argument('--check',action='store_true');args=p.parse_args();assert min(args.intervals)>0;assert torch.cuda.is_available();torch.manual_seed(42);torch.set_float32_matmul_precision('high')
    if args.offgrid_auto and not args.check:
        assert not args.offgrid,'Choose explicit --offgrid or --offgrid-auto'
        candidates=[p for p in sorted((ROOT/'datasets').rglob('*.h5')) if any(x in str(p).lower() for x in ('offgrid','off_grid','off-grid','interpolation'))]
        primary=[p for p in candidates if any(x in str(p).lower() for x in ('offgrid','off_grid','off-grid'))]
        for p in primary or candidates:
            with h5py.File(p,'r') as f:
                keys=[k for k in f if k.startswith('trajectory_')]
                if keys and all('rgb' in f[k] and 'target_world' in f[k] and 'contact_impulses' in f[k] and 'pair_id' in f[k].attrs and f[k]['rgb'].shape==(223,256,256,3) for k in keys):args.offgrid.append(str(p.relative_to(ROOT)))
        assert args.offgrid,'No raw off-grid H5 detected; use --offgrid path/to/raw.h5'
        print('OFFGRID DATASETS',args.offgrid,flush=True)
    for arch in MODES if args.arch=='both' else [args.arch]:
        if args.check:
            a=load_ae(arch)
            with h5py.File(DATA,'r') as f,torch.no_grad():
                g=f[sorted(k for k in f if k.startswith('trajectory_'))[0]]; x=rgb(g['rgb'][:1]); y=rgb(g['rgb'][25:26]); ss=a.encode_static(x); zz=a.encode_dynamic(y-x); im=decode(a,x,ss,zz)
                assert ss.shape==zz.shape==(1,128) and im.shape==(1,3,256,256) and torch.isfinite(im).all()
            print('CHECK PASSED',arch,visual_cost(a),flush=True);del a
        else:run_arch(arch,args)
        torch.cuda.empty_cache()
if __name__=='__main__':main()

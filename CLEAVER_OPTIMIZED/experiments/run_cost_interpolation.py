#!/usr/bin/env python3
"""FINAL162 frozen inversion cost and truly off-grid evaluation. Run as a module."""
import argparse, csv, hashlib, inspect, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'CLEAVER_OPTIMIZED/outputs/final162'
RAW = ROOT/'datasets/cleaver_extended_workspace_162_complete/cleaver_same_goal_162.h5'
EXTERNAL = ROOT/'datasets/cleaver_offgrid_48'
FRAMES = [1,2,3,5,10,12,23,45,67,112,157,223]
OBJECTS = np.array([[1.7,2.4],[2.6,8.1],[6.7,7.3],[7.1,3.4]])/100
NEW_STARTS = np.array([[-9.2,-1.1,35],[-1.4,-8.7,35],[-7.1,-10.4,35]])/100
METHODS = ('L-BFGS','Dense grid search','Nearest demonstration')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def make_plan():
    import h5py
    from scipy.spatial import Delaunay
    split=json.loads((BASE/'CESN_Vision/split_used.json').read_text())
    train=set(split['train']); all_obj={}; all_start={}; train_obj={}; offsets=[]
    with h5py.File(RAW,'r') as h:
        for key in sorted(h):
            if not key.startswith('trajectory_'):continue
            g=h[key];o=int(g.attrs['object_id']);r=int(g.attrs['start_id']);p=int(g.attrs['pair_id'])
            all_obj[o]=g['knife_xy'][:];all_start[r]=g['requested_start_xyz'][:]
            offsets.append(g['target_world'][:2]-g['knife_xy'][:])
            if p in train:train_obj[o]=g['knife_xy'][:]
    known=np.array([all_start[i] for i in (0,4,8)]); starts=np.vstack([known,NEW_STARTS]); delta=np.asarray(offsets[0]);assert np.allclose(offsets,delta,atol=1e-7)
    training=np.array(list(train_obj.values())); original=np.array(list(all_obj.values()));robot=np.array(list(all_start.values()))
    assert np.all(Delaunay(training).find_simplex(OBJECTS,tol=1e-9)>=0)
    assert np.all(np.min(np.linalg.norm(OBJECTS[:,None]-original[None],axis=2),axis=1)>.01)
    assert np.all(Delaunay(robot[:,:2]).find_simplex(NEW_STARTS[:,:2],tol=1e-9)>=0)
    assert np.all(np.min(np.linalg.norm(NEW_STARTS[:,None]-robot[None],axis=2),axis=1)>.01)
    assert len({tuple(x) for x in starts})==6
    ids=[o*9+r for o in range(4) for r in range(6)]
    return dict(objects_m=OBJECTS.tolist(),goals_m=(OBJECTS+delta).tolist(),starts_m=starts.tolist(),pair_ids=ids,train_objects_m=training.tolist(),original_objects_m=original.tolist(),original_starts_m=robot.tolist(),known_start_ids=[0,4,8],new_start_slots=[3,4,5],source_split=split)

def generate(resume):
    import h5py
    from Data_makers import build_cleaver_dataset_162_fixed as gen
    plan=make_plan(); path=EXTERNAL/'plan.json'; data=EXTERNAL/'cleaver_same_goal_162.h5'
    if EXTERNAL.exists():
        if not resume:raise RuntimeError('External dataset exists; use generate --resume')
        assert json.loads(path.read_text())==plan,'Plan changed'
    else:
        assert not resume,'No external dataset to resume'
    print('Off-grid cleaver positions (cm):',OBJECTS*100,flush=True);print('Robot starts (cm):',np.array(plan['starts_m'])*100,flush=True)
    # gen.main creates its output directory; plan is written beforehand only on resume or afterwards.
    gen.START_CENTER=np.array([-.05,-.05,.35]);gen.START_OFFSETS=[tuple(p[:2]-gen.START_CENTER[:2]) for p in np.array(plan['starts_m'])]
    gen.OBJECT_OFFSETS=[tuple(p-gen.v4.ORIGIN[:2]) for p in OBJECTS]
    assert np.allclose(np.array(gen.OBJECT_OFFSETS)+gen.v4.GOAL[:2],plan['goals_m'],atol=1e-7)
    if not EXTERNAL.exists():
        EXTERNAL.mkdir(parents=True);atomic_json(path,plan)
    # Underlying generator allows a precreated directory with --resume, and opens a new HDF5 in append mode.
    argv=sys.argv[:];sys.argv=[argv[0],'--out',str(EXTERNAL),'--resume','--pairs',*map(str,plan['pair_ids'])]
    try:gen.main()
    finally:sys.argv=argv
    with h5py.File(data,'r+') as h:
        keys={k for k in h if k.startswith('trajectory_')}; expected={f'trajectory_{2*p+l:03d}' for p in plan['pair_ids'] for l in (0,1)}
        assert keys<=expected
        for p in plan['pair_ids']:
            for l in (0,1):
                name=f'trajectory_{2*p+l:03d}'
                if name not in h:continue
                g=h[name]; assert g['rgb'].shape==(223,256,256,3) and g['rgb'].dtype==np.uint8
                assert (int(g.attrs['pair_id']),int(g.attrs['object_id']),int(g.attrs['start_id']),int(g.attrs['label']))==(p,p//9,p%9,l)
                assert np.allclose(g['knife_xy'][:],plan['objects_m'][p//9],atol=1e-7)
                assert np.allclose(g['target_world'][:2],plan['goals_m'][p//9],atol=1e-7)
                assert np.allclose(g['requested_start_xyz'][:],plan['starts_m'][p%9],atol=1e-7)
        h.attrs.update(dataset_name='cleaver_offgrid_48',requested_trajectories=48,object_locations=4,robot_starts=6,accepted_pairs=len(keys)//2,failed_pairs=24-len(keys)//2)
    missing=sorted(expected-keys);atomic_json(EXTERNAL/'summary.json',dict(requested_pairs=24,accepted_pairs=len(keys)//2,accepted_trajectories=len(keys),failed_pairs=24-len(keys)//2,missing=missing))
    if missing:raise RuntimeError(f'Off-grid feasibility failed: missing {missing}. Inspect pairs/ reports; do not choose replacements based on model performance.')
    audit_external(plan);print('OFFGRID48 COMPLETE',flush=True)

def audit_external(plan):
    import h5py
    path=EXTERNAL/'cleaver_same_goal_162.h5'; ids=plan['pair_ids']
    with h5py.File(path,'r') as h:
        assert set(h)=={f'trajectory_{2*p+l:03d}' for p in ids for l in (0,1)}
        for p in ids:
            a,b=[h[f'trajectory_{2*p+l:03d}'] for l in (0,1)]
            assert np.allclose(a['qpos'][0],b['qpos'][0],atol=1e-5,rtol=0)
            assert np.allclose(a['target_world'][:],b['target_world'][:],atol=1e-7,rtol=0)
            for l,g in enumerate((a,b)):
                assert g['rgb'].shape==(223,256,256,3) and g['rgb'].dtype==np.uint8
                assert int(g.attrs['label'])==l and int(g.attrs['pair_id'])==p
                assert np.allclose(g['knife_xy'][:],plan['objects_m'][p//9],atol=1e-7)
                assert np.allclose(g['target_world'][:2],plan['goals_m'][p//9],atol=1e-7)
                assert np.allclose(g['requested_start_xyz'][:],plan['starts_m'][p%9],atol=1e-7)
                imp=g['contact_impulses'][:];assert np.isfinite(imp).all() and bool(np.any(imp[:,:2].sum(1)>1e-4))==bool(l)
                assert np.isfinite(g['qpos'][:]).all()
            assert json.loads((EXTERNAL/'pairs'/f'pair_{p:03d}'/'summary.json').read_text())['accepted']
    return path

def external_latents(device,batch,plan):
    import torch,h5py
    from CLEAVER_OPTIMIZED.autoencoder.lightweight_variable import LightweightAutoencoderVariable
    raw=audit_external(plan); ae=BASE/'autoencoder_lightweight128_matched/autoencoder.pt'; lat=EXTERNAL/'latents128.h5'
    sig=dict(raw_sha256=digest(raw),ae_sha256=digest(ae),encoder='lightweight_variable128',precision='cuda_amp_fp16' if device.type=='cuda' else 'float32')
    marker=EXTERNAL/'latent_settings.json'
    if lat.exists() and marker.exists():
        assert json.loads(marker.read_text())==sig,'External latent identity mismatch';return lat
    net=LightweightAutoencoderVariable(128).to(device).eval();net.load_state_dict(torch.load(ae,map_location='cpu',weights_only=False)['model_state_dict'])
    def rgb(a):return torch.as_tensor(np.asarray(a),device=device).permute(0,3,1,2).float()/255
    tmp=lat.with_suffix('.tmp')
    with h5py.File(raw,'r') as h,h5py.File(tmp,'w') as out,torch.no_grad():
        for i,key in enumerate(sorted(h)):
            g=h[key];x=rgb(g['rgb'][0:1]);chunks=[]
            with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                s=net.encode_static(x)
                for t in range(0,223,batch):chunks.append(net.encode_dynamic(rgb(g['rgb'][t:t+batch])-x).float().cpu().numpy())
            dst=out.create_group(key)
            for k,v in g.attrs.items():dst.attrs[k]=v
            dst.create_dataset('z_static',data=s[0].float().cpu().numpy());dst.create_dataset('z_dynamic',data=np.concatenate(chunks))
            for name in ('target_world','contact_impulses'):dst.create_dataset(name,data=g[name][:])
            print('OFFGRID LATENT',i+1,'/48',flush=True)
    tmp.replace(lat);atomic_json(marker,sig);return lat

def read_external(lat,info,device):
    import torch,h5py
    keys=[];pairs=[];labels=[];s=[];z=[];xy=[];hits={}
    with h5py.File(lat,'r') as h:
        for key in sorted(h):
            g=h[key];p=int(g.attrs['pair_id']);l=int(g.attrs['label']);keys.append(key);pairs.append(p);labels.append(l);s.append(g['z_static'][:]);z.append(g['z_dynamic'][:]);xy.append(g['target_world'][:2]);
            if l:hits[p]=int(np.flatnonzero(g['contact_impulses'][:,:2].sum(1)>1e-4)[0])
    s=np.stack(s);z=np.stack(z);xy=np.stack(xy);assert s.shape==(48,128) and z.shape==(48,223,128) and np.isfinite(s).all() and np.isfinite(z).all()
    lo=np.array(info['xy_min']);hi=np.array(info['xy_max']);assert np.all(xy>lo) and np.all(xy<hi)
    def t(x):return torch.as_tensor(x,dtype=torch.float32,device=device)
    return dict(keys=keys,pair=np.array(pairs),label=np.array(labels),static=s,z=z,xy=xy,contact=np.array([hits[p] for p in pairs]),pca=t((s-np.array(info['static_mean']))/info['static_scale']),yn=t((z-np.array(info['dynamic_mean']))/info['dynamic_scale']),xyn=t(2*(xy-lo)/(hi-lo)-1),outside_bounds=np.zeros(len(keys),dtype=bool))

def cost(k,method,grid_points,train_n,calls=None):
    # Dense GEMM arithmetic estimate (2 FLOPs per multiply-add). No claim of measured hardware FLOPs.
    N,D,C,P=1500,128,132,4
    f=2*(C*N+P*D+k*(N*N+2*N*D))
    # Frozen weights, teacher-forced feedback; no gradient w.r.t. observations or weights.
    back=2*(C*N+P*D+k*N*D+max(k-1,0)*N*N)
    nf=nb=nd=0;df=128*(k+1)
    if method==METHODS[0]:
        assert calls is not None and calls>=2;nf=18*calls;nb=18*(calls-1);fl=nf*f+nb*back
    elif method==METHODS[1]:nf=2*grid_points;fl=nf*f
    else:nd=train_n;fl=nd*(3*df-1)
    rollout=2*(C*N+P*D+223*(N*N+2*N*D))
    return dict(candidate_forward=nf,candidate_backward=nb,forward_reservoir_updates=nf*k,backward_reservoir_steps=nb*k,distance_candidates=nd,feature_dim=df,estimated_search_flops=int(fl),estimated_search_plus_rollout_flops=int(fl+rollout))

def evaluate(args):
    import torch
    from CLEAVER_OPTIMIZED.experiments import run_context_inversion_comparison as c
    from CLEAVER_OPTIMIZED.cesn import model as b
    device=torch.device(args.device);torch.manual_seed(42);torch.set_float32_matmul_precision('high')
    model,data,lo,hi,info,manifest=c.load(device)
    # Confirm the exact implementation used to interpret closure_calls: final forward counted, no backward.
    src=inspect.getsource(b.invert);assert 'total.backward(); calls+=1' in src and 'remember(loss,xy); calls+=1' in src,'Inspect invert counter before using arithmetic estimates'
    assert model.n==1500 and model.C==4 and model.D==1505 and model.Wc.shape==(132,1500)
    assert b.digest(Path(b.__file__))==manifest['core_sha256'],'Frozen CESN implementation changed'
    starts=c.tensor(np.random.default_rng(42).uniform(-.95,.95,(9,2)),device);starts[0]=0
    axes=[torch.linspace(-1,1,max(2,int(np.ceil(float(w)*1000/args.grid_mm-1e-5))+1),device=device) for w in hi-lo];grid=torch.cartesian_prod(*axes)
    tr=data['train']; split_names=['validation','test'] if args.mode=='existing' else ['offgrid_known_start','offgrid_new_start']
    out=ROOT/'paper/results/final162'/('context_inversion_cost' if args.mode=='existing' else 'offgrid_interpolation');figdir=ROOT/'paper/figures/final162'/out.name;cache=BASE/'evaluation_cache'/out.name
    out.mkdir(parents=True,exist_ok=True);figdir.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    ext_hash=None
    if args.mode=='external':
        plan=make_plan();assert json.loads((EXTERNAL/'plan.json').read_text())==plan
        lat=external_latents(device,args.encode_batch,plan);ext_hash=digest(lat);d=read_external(lat,info,device)
        for sn,mask in zip(split_names,(d['pair']%9<3,d['pair']%9>=3)):
            ix=np.flatnonzero(mask);data[sn]={k:([v[i] for i in ix] if k=='keys' else v[ix.tolist()] if torch.is_tensor(v) else v[ix]) for k,v in d.items()};assert len(ix)==24
    settings=dict(mode=args.mode,model_sha256=digest(BASE/'CESN_Vision/cesn.pt'),latent_sha256=manifest['latent_sha256'],external_latent_sha256=ext_hash,script_sha256=digest(Path(__file__)),grid_mm=args.grid_mm,grid_points=len(grid),grid_batch=args.grid_batch,frames=FRAMES,device=str(device),cost_definition='Estimated dense matrix arithmetic and nearest distance arithmetic only. Excludes encoding, feature setup, elementwise reservoir operations, loss/selection, optimizer vector algebra, memory and hardware effects. L-BFGS calls includes final no-grad forward. Shared final CESN rollout reported separately.',batch=3,nearest_control='retrieved_suffix_nrmse is a separate retrieval-only control, not the CESN-conditioned future metric plotted')
    marker=out/'settings.json'
    if marker.exists():assert json.loads(marker.read_text())==settings,'Evaluation identity changed; do not mix cache'
    else:atomic_json(marker,settings)
    rows=[];scale=info['dynamic_scale']
    for k in FRAMES:
        tf=c.feature(tr,k)
        for sn in split_names:
            d=data[sn]
            for i in range(0,len(d['keys']),3):
                ids=list(range(i,min(i+3,len(d['keys']))));path=cache/f'{sn}_{k:03d}_{i:03d}.json'
                if path.exists():rows.extend(json.loads(path.read_text()));continue
                rr=[]
                res=b.invert(model,d,ids,k,starts,SimpleNamespace(max_iter=40),lo,hi,scale)
                for r in res:
                    rr.append(dict(method=METHODS[0],split=sn,trajectory=r['trajectory'],pair_id=r['pair_id'],frames=k,true_label=r['true_label'],pred_label=r['pred_label'],correct=r['correct'],true_x_m=r['true_x_m'],true_y_m=r['true_y_m'],pred_x_m=r['pred_x_m'],pred_y_m=r['pred_y_m'],xy_error_mm=r['xy_error_mm'],prefix_nrmse=r['prefix_rmse']/scale,future_nrmse=None if r['future_rmse'] is None else r['future_rmse']/scale,precontact=r['precontact'],retrieved_suffix_nrmse=None,**cost(k,METHODS[0],len(grid),len(tr['keys']),r['closure_calls'])))
                for method in METHODS[1:]:
                    retrieved=None
                    if method==METHODS[1]:
                        lab,xy=c.grid_search(model,d,ids,k,grid,args.grid_batch)
                    else:
                        # Same 1-NN objective as the existing comparison; keep selected index for a separate retrieval-only control.
                        q=c.feature(d,k,ids);j=(q[:,None]-tf[None]).square().sum(-1).argmin(1)
                        lab=torch.as_tensor(tr['label'],device=device)[j];xy=tr['xyn'][j]
                        if k<223:
                            retrieved=(tr['yn'][j,k:]-d['yn'][ids,k:]).square().mean((1,2)).sqrt().cpu().numpy()
                    pred=(lo+(xy+1)*.5*(hi-lo)).cpu().numpy();sc=c.score(model,d,ids,k,lab,xy,lo,hi,0.,method,sn)
                    for j,r in enumerate(sc):
                        r.pop('inference_ms');r.update(true_x_m=float(d['xy'][ids[j],0]),true_y_m=float(d['xy'][ids[j],1]),pred_x_m=float(pred[j,0]),pred_y_m=float(pred[j,1]),retrieved_suffix_nrmse=None if retrieved is None else float(retrieved[j]),**cost(k,method,len(grid),len(tr['keys'])));rr.append(r)
                # CSV keys have the same set, order may differ.
                atomic_json(path,rr);rows.extend(rr)
            print(sn,k,'completed',flush=True)
        write_csv(out/'per_trajectory.csv',rows)
    summarize(rows,split_names,out,figdir,args.mode)
    print('COST AND INVERSION COMPLETE',out,figdir,flush=True)

def summarize(rows,splits,out,figdir,mode):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    def avg(rr,key):
        vals=[r[key] for r in rr if r[key] is not None];return float(np.mean(vals)) if vals else None
    summary=[];agreement=[]
    for sn in splits:
        for k in FRAMES:
            for m in METHODS:
                rr=[r for r in rows if (r['split'],r['frames'],r['method'])==(sn,k,m)];assert rr
                pain=[r for r in rr if r['precontact'] and r['true_label']==1]
                summary.append(dict(split=sn,method=m,frames=k,observed_percent=100*k/223,count=len(rr),accuracy_pct=100*avg(rr,'correct'),pain_recall_pct=100*avg(pain,'correct') if pain else None,precontact_pain_count=len(pain),xy_mean_mm=avg(rr,'xy_error_mm'),prefix_nrmse=avg(rr,'prefix_nrmse'),future_nrmse=avg(rr,'future_nrmse'),retrieved_suffix_nrmse=avg(rr,'retrieved_suffix_nrmse'),**{key:avg(rr,key) for key in ('candidate_forward','candidate_backward','forward_reservoir_updates','backward_reservoir_steps','distance_candidates','estimated_search_flops','estimated_search_plus_rollout_flops')}))
            a={r['trajectory']:r for r in rows if (r['split'],r['frames'],r['method'])==(sn,k,METHODS[0])};g={r['trajectory']:r for r in rows if (r['split'],r['frames'],r['method'])==(sn,k,METHODS[1])};assert a.keys()==g.keys()
            dist=[1000*np.hypot(a[x]['pred_x_m']-g[x]['pred_x_m'],a[x]['pred_y_m']-g[x]['pred_y_m']) for x in a]
            diff=[a[x]['prefix_nrmse']-g[x]['prefix_nrmse'] for x in a]
            agreement.append(dict(split=sn,frames=k,count=len(a),intent_agreement_pct=100*np.mean([a[x]['pred_label']==g[x]['pred_label'] for x in a]),mean_estimate_distance_mm=float(np.mean(dist)),mean_prefix_nrmse_difference=float(np.mean(diff)),lbfgs_better_pct=100*np.mean(np.array(diff)<-1e-6),grid_better_pct=100*np.mean(np.array(diff)>1e-6)))
    write_csv(out/'comparison.csv',summary);write_csv(out/'agreement.csv',agreement)
    plt.rcParams.update({'font.size':9});metrics=[('accuracy_pct','Intent accuracy (%)'),('pain_recall_pct','Precontact PAIN recall (%)'),('xy_mean_mm','Mean XY error (mm)'),('prefix_nrmse','Prefix latent NRMSE'),('future_nrmse','Future latent NRMSE'),('estimated_search_flops','Estimated search arithmetic (GFLOPs)')]
    styles=[('o','-'),('x','--'),('^',':')]
    for sn in splits:
        fig,axs=plt.subplots(2,3,figsize=(10.4,6),layout='constrained')
        for ax,(metric,label) in zip(axs.flat,metrics):
            for m,(mark,ls) in zip(METHODS,styles):
                rr=[r for r in summary if r['split']==sn and r['method']==m and r[metric] is not None];ax.plot([r['observed_percent'] for r in rr],[r[metric]/1e9 if metric=='estimated_search_flops' else r[metric] for r in rr],marker=mark,linestyle=ls,ms=3,label=m)
            if metric in ('accuracy_pct','pain_recall_pct'):ax.set_ylim(-3,103);ax.set_xlim(0,15)
            if metric=='estimated_search_flops':ax.set_yscale('log')
            ax.set(xlabel='Observed trajectory (%)',ylabel=label);ax.grid(alpha=.2)
        axs[0,0].legend(fontsize=7,loc='lower right');fig.savefig(figdir/f'{sn}_context_inversion.png',dpi=300);plt.close(fig)
    if mode=='external':
        plan=make_plan();fig,axs=plt.subplots(1,2,figsize=(7,3.5),layout='constrained')
        from scipy.spatial import ConvexHull
        for ax,old,train,new,title in [(axs[0],plan['original_objects_m'],plan['train_objects_m'],plan['objects_m'],'Cleaver positions'),(axs[1],plan['original_starts_m'],plan['original_starts_m'],plan['starts_m'][3:],'Robot start positions')]:
            old=np.array(old)[:,:2]*100;train=np.array(train)[:,:2]*100;new=np.array(new)[:,:2]*100;h=ConvexHull(train);poly=train[np.r_[h.vertices,h.vertices[0]]];ax.plot(poly[:,0],poly[:,1],':',label='Training hull');ax.scatter(old[:,0],old[:,1],marker='o',facecolors='none',edgecolors='0.5',label='Original');ax.scatter(train[:,0],train[:,1],marker='o',label='Training');ax.scatter(new[:,0],new[:,1],marker='x',label='New off-grid');ax.set(title=title,xlabel='X (cm)',ylabel='Y (cm)',aspect='equal');ax.grid(alpha=.2);ax.legend(fontsize=6)
        fig.savefig(figdir/'offgrid_workspace.png',dpi=300);plt.close(fig)

def main():
    import os
    os.chdir(ROOT)
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','generate','evaluate']);p.add_argument('--resume',action='store_true');p.add_argument('--mode',choices=['existing','external'],default='existing');p.add_argument('--device',default='cuda');p.add_argument('--grid-mm',type=float,default=5.);p.add_argument('--grid-batch',type=int,default=128);p.add_argument('--encode-batch',type=int,default=128);a=p.parse_args();assert a.grid_mm>0 and min(a.grid_batch,a.encode_batch)>0
    if a.action=='plan':print(json.dumps(make_plan(),indent=2))
    elif a.action=='generate':generate(a.resume)
    else:evaluate(a)
if __name__=='__main__':main()

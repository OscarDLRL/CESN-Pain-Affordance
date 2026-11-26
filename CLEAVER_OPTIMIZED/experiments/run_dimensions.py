"""Matched dynamic dimension sweep. Validation only unless --evaluate-test."""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import h5py
from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED.cesn import model as b
from CLEAVER_OPTIMIZED.autoencoder.lightweight_variable import LightweightAutoencoderVariable

def normalize(data,device):
    tr=data['train']; s=tr['static'].astype(np.float64); sm=s.mean(0)
    # Match E20 scalar normalization: largest singular value / sqrt(n), no projection.
    ps=max(float(np.linalg.norm(s-sm,ord=2)/np.sqrt(len(s))),1e-6)
    zm=tr['z'].astype(np.float64).mean((0,1)); zs=max(float((tr['z']-zm).std()),1e-6)
    lo=tr['xy'].min(0); hi=tr['xy'].max(0); assert np.all(hi>lo)
    def t(x):return torch.as_tensor(x,dtype=torch.float32,device=device)
    for d in data.values():
        d['pca']=t((d['static']-sm)/ps);d['yn']=t((d['z']-zm)/zs);d['xyn']=t(2*(d['xy']-lo)/(hi-lo)-1);d['outside_bounds']=np.any((d['xy']<lo-1e-6)|(d['xy']>hi+1e-6),axis=1)
    return dict(static_mean=sm.tolist(),static_scale=ps,dynamic_mean=zm.tolist(),dynamic_scale=zs,xy_min=lo.tolist(),xy_max=hi.tolist()),t(lo),t(hi)

@torch.no_grad()
def landscape(model,d,lo,hi,frames,mm,batch):
    axes=[torch.linspace(-1,1,max(2,math.ceil(float(w)*1000/mm)+1),device=lo.device) for w in hi-lo]
    grid=torch.cartesian_prod(*axes); rows=[]
    for i in range(len(d['keys'])):
        for k in frames:
            obs=d['yn'][i:i+1,:k]; true=d['xyn'][i:i+1]; candidates=torch.cat([grid,true]); losses=[]
            for xy in candidates.split(batch):
                c=model.context(torch.full((len(xy),),int(d['label'][i]),device=lo.device),d['pca'][i:i+1].expand(len(xy),-1),xy)
                y=obs.expand(len(xy),-1,-1); losses.append((model.predict(c,k,y)-y).square().mean((1,2)))
            loss=torch.cat(losses); j=int(loss[:-1].argmin()); est=lo+(grid[j]+1)*.5*(hi-lo)
            rows.append(dict(trajectory=d['keys'][i],frames=k,true_loss=float(loss[-1]),grid_min_loss=float(loss[j]),grid_xy_error_mm=float(torch.linalg.vector_norm(est-(lo+(true[0]+1)*.5*(hi-lo)))*1000),true_better_than_grid=int(loss[-1]<=loss[j]),grid_candidates=len(grid)))
    return rows

@torch.no_grad()
def rgb_metrics(model,d,info,checkpoint,dim,device):
    ae=LightweightAutoencoderVariable(dim).to(device).eval(); ae.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']); mu=torch.tensor(info['dynamic_mean'],device=device,dtype=torch.float32); scale=info['dynamic_scale']; rows=[]
    c=model.context(torch.tensor(d['label'],device=device),d['pca'],d['xyn']); pred=model.predict(c,223)*scale+mu
    ts=[0,5,10,25,75,125,175,222]
    with h5py.File(cfg.RGB,'r') as h:
        for i,key in enumerate(d['keys']):
            rgb=h[key]['rgb']; x=torch.tensor(rgb[0],device=device).permute(2,0,1)[None].float()/255; y=torch.tensor(rgb[ts],device=device).permute(0,3,1,2).float()/255
            static=torch.tensor(d['static'][i:i+1],device=device).expand(len(ts),-1)
            z=torch.tensor(d['z'][i,ts],device=device)
            for name,latent in [('ae',z),('oracle_autonomous',pred[i,ts])]:
                rec=(x+ae.decode_dynamic_full(latent,static)['delta_recon']).clamp(0,1)
                rows.extend(dict(trajectory=key,frame=t,method=name,rgb_mse=float(v)) for t,v in zip(ts,(rec-y).square().flatten(1).mean(1)))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--dimensions',nargs='+',type=int,default=[128,200,256]);p.add_argument('--reservoir',type=int,default=1500);p.add_argument('--ridge',type=float,default=.5);p.add_argument('--radius',type=float,default=.95);p.add_argument('--leak',type=float,default=.25);p.add_argument('--seed',type=int,default=42);p.add_argument('--restarts',type=int,default=9);p.add_argument('--max-iter',type=int,default=40);p.add_argument('--trajectory-batch',type=int,default=3);p.add_argument('--frames',nargs='+',type=int,default=[1,2,3,5,10,12,23,45,67,112,157,223]);p.add_argument('--grid-mm',type=float,default=5);p.add_argument('--grid-batch',type=int,default=128);p.add_argument('--grid-frames',nargs='+',type=int,default=[10,223]);p.add_argument('--device',default='cuda');p.add_argument('--run-name',default='dynamic_dimension_sweep');p.add_argument('--evaluate-test',action='store_true');args=p.parse_args()
    device=torch.device(args.device);torch.manual_seed(args.seed);torch.set_float32_matmul_precision('high');out=cfg.OUT/args.run_name;out.mkdir(parents=True,exist_ok=True)
    summaries=[];oracles=[];land=[];rgbs=[]; b.EXPS=[]
    for dim in args.dimensions:
        ae=cfg.OUT/f'autoencoder_lightweight{dim}_matched';lat=ae/f'cleaver_lightweight_latent{dim}.h5'; data,split=b.load(lat,ae/'split.json');data,split=b.grouped_validation(data,split);assert all(d['z'].shape[-1]==dim for d in data.values());info,lo,hi=normalize(data,device);scale=info['dynamic_scale'];name=f'E{ {128:27,200:28,256:29}[dim]}_Dynamic{dim}';b.EXPS.append((name,128,True));folder=out/name;folder.mkdir(exist_ok=True)
        sig=dict(args=vars(args),split=split,latent_sha256=b.digest(lat),code=b.digest(Path(__file__)),core=b.digest(Path(b.__file__)));mp=folder/'manifest.json'
        if mp.exists():assert json.loads(mp.read_text())==sig,'Use a new --run-name for changed settings'
        else:b.atomic_json(mp,sig)
        b.atomic_json(folder/'normalization.json',info);b.atomic_json(folder/'split_used.json',split)
        w,wc,wf,bias=b.common_weights(args.reservoir,args.seed,args.radius,device);raw=torch.tensor(np.random.default_rng(args.seed+1009).uniform(-.5,.5,(128,args.reservoir)).astype(np.float32),device=device)
        if dim!=128:
            rng_w=np.random.default_rng(args.seed);rng_w.uniform(-1,1,(args.reservoir,args.reservoir));rng_w.random((args.reservoir,args.reservoir));rng_w.uniform(-.5,.5,(20,args.reservoir))
            wf=torch.tensor(rng_w.uniform(-.5,.5,(dim,args.reservoir)).astype(np.float32)/math.sqrt(dim),device=device)
        model=b.CESN((w,torch.cat([wc[:2],raw,wc[18:20]],0),wf,bias),128,True,args.leak,b.OPTIONS['E20']);tick=time.perf_counter();model.fit(data['train'],args.ridge);b.sync(device);secs=time.perf_counter()-tick
        b.atomic_pt(folder/'cesn.pt',dict(W=model.W.cpu(),Wc=model.Wc.cpu(),Wf=model.Wf.cpu(),bias=model.bias.cpu(),readout=model.readout.cpu(),normalization=info,split=split,leak=args.leak));starts=torch.tensor(np.random.default_rng(args.seed).uniform(-.95,.95,(args.restarts,2)),device=device,dtype=torch.float32);starts[0]=0
        b.latent_geometry(data,cfg.RGB,folder,scale)
        for sn in (['validation','test'] if args.evaluate_test else ['validation']):
            d=data[sn];part=folder/sn;part.mkdir(exist_ok=True);o=model.oracle(d,scale);oracles.append(dict(experiment=name,split=sn,train_seconds=secs,**o,teacher_forced_nrmse=o['teacher_forced_rmse']/scale,autonomous_nrmse=o['autonomous_rmse']/scale)); print(name,sn,oracles[-1],flush=True);rows=[]
            for k in sorted(set(args.frames)):
                for i in range(0,len(d['keys']),args.trajectory_batch):
                    fp=part/f'prefix_{k:03d}_batch_{i:03d}.json';ids=list(range(i,min(i+args.trajectory_batch,len(d['keys']))))
                    if fp.exists():rr=json.loads(fp.read_text())
                    else:rr=b.invert(model,d,ids,k,starts,args,lo,hi,scale);b.atomic_json(fp,rr)
                    rows.extend(rr)
                ss=[r for r in rows if r['frames']==k];print(name,sn,k,'acc',100*np.mean([r['correct'] for r in ss]),'XY',np.mean([r['xy_error_mm'] for r in ss]),flush=True)
            b.csv_write(part/'per_trajectory.csv',rows)
            for r in b.summarize(rows):
                r['future_nrmse']=None if r['future_rmse'] is None else r['future_rmse']/scale;summaries.append(dict(experiment=name,split=sn,**r))
            land.extend(dict(experiment=name,split=sn,**r) for r in landscape(model,d,lo,hi,args.grid_frames,args.grid_mm,args.grid_batch))
            rgbs.extend(dict(experiment=name,split=sn,**r) for r in rgb_metrics(model,d,info,ae/'autoencoder.pt',dim,device))
        for filename,rr in [('comparison.csv',summaries),('oracle_forward.csv',oracles),('xy_landscape.csv',land),('rgb_sample_metrics.csv',rgbs)]:b.csv_write(out/filename,rr)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for sn in sorted({r['split'] for r in summaries}):
        for metric,label in [('accuracy_pct','Intent accuracy (%)'),('xy_mean_mm','XY error (mm)'),('future_nrmse','Future normalized latent RMSE'),('pain_recall_pct','PAIN recall (%)')]:
            fig,ax=plt.subplots(figsize=(3.5,2.5));subset='precontact' if metric=='pain_recall_pct' else 'all'
            for name,_,_ in b.EXPS:
                rr=[r for r in summaries if r['experiment']==name and r['split']==sn and r['subset']==subset and r[metric] is not None];ax.plot([r['frames'] for r in rr],[r[metric] for r in rr],marker='o',ms=3,label=name.split('_')[1])
            ax.set(xlabel='Observed frames',ylabel=label,xscale='log');ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/f'{sn}_{metric}.pdf');fig.savefig(out/f'{sn}_{metric}.png',dpi=300);plt.close(fig)
    print('DONE',out,flush=True)
if __name__=='__main__':main()

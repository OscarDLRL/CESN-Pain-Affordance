"""E31C: final low-leak sweep. No augmentation. Selection on grouped validation only."""
import argparse, itertools, json, math, time
from pathlib import Path
import numpy as np
import torch
from CLEAVER_OPTIMIZED import config as cfg
from CLEAVER_OPTIMIZED import dimension_cesn_core as b
from CLEAVER_OPTIMIZED.run_dimensions import normalize, landscape, rgb_metrics

@torch.no_grad()
def gram(model,d):
    y=d['yn']; c=model.context(torch.as_tensor(d['label'],device=y.device),d['pca'],d['xyn']); D=model.D
    G=torch.zeros((D,D),device=y.device,dtype=torch.float64); H=torch.zeros((D,y.shape[-1]),device=y.device,dtype=torch.float64)
    for i in range(0,len(y),8):
        f=model.features(c[i:i+8],y[i:i+8]).reshape(-1,D).double(); G+=f.T@f; H+=f.T@y[i:i+8].reshape(-1,y.shape[-1]).double()
    return G,H

def make_model(weights,n,radius,leak,gain,seed,device):
    w,wc,wf,bias=weights
    raw=torch.as_tensor(np.random.default_rng(seed+1009).uniform(-.5,.5,(128,n)).astype(np.float32),device=device)
    return b.CESN((w*radius,torch.cat([wc[:2],raw,wc[18:20]],0),wf,bias),128,True,leak,(1,1,1,gain,True,False,False))

def metrics(model,d):
    o=model.oracle(d,1.0)
    return dict(tf_nrmse=o['teacher_forced_rmse'],auto_nrmse=o['autonomous_rmse'],nonfinite=o['teacher_forced_nonfinite_count']+o['autonomous_nonfinite_count'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--reservoirs',nargs='+',type=int,default=[1500]);p.add_argument('--radii',nargs='+',type=float,default=[.3,.45,.6]);p.add_argument('--leaks',nargs='+',type=float,default=[.015,.025,.035,.05,.065]);p.add_argument('--ridges',nargs='+',type=float,default=[7.5,10.,15.,20.,30.]);p.add_argument('--gains',nargs='+',type=float,default=[.5,.75,1.]);p.add_argument('--shortlist',type=int,default=20);p.add_argument('--finalists',type=int,default=12);p.add_argument('--grid-mm',type=float,default=5);p.add_argument('--grid-batch',type=int,default=128);p.add_argument('--restarts',type=int,default=9);p.add_argument('--max-iter',type=int,default=40);p.add_argument('--trajectory-batch',type=int,default=3);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');p.add_argument('--run-name',default='E31C_low_leak_sweep');a=p.parse_args()
    assert min(a.reservoirs)>0 and min(a.radii)>0 and min(a.leaks)>0 and max(a.leaks)<=1 and min(a.ridges)>0 and min(a.gains)>=0 and min(a.shortlist,a.finalists,a.restarts,a.trajectory_batch,a.grid_batch)>0 and a.grid_mm>0
    device=torch.device(a.device);torch.manual_seed(a.seed);torch.set_float32_matmul_precision('high')
    ae=cfg.OUT/'autoencoder_lightweight128_matched';lat=ae/'cleaver_lightweight_latent128.h5';data,split=b.load(lat,ae/'split.json');data,split=b.grouped_validation(data,split);info,lo,hi=normalize(data,device);scale=info['dynamic_scale'];out=cfg.OUT/a.run_name;out.mkdir(parents=True,exist_ok=True)
    sig=dict(args=vars(a),split=split,latent=b.digest(lat),code=b.digest(Path(__file__)),core=b.digest(Path(b.__file__)),normalizer=b.digest(Path(__file__).with_name('run_dimensions.py'))); mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==sig,'Changed settings: choose a new --run-name'
    else:b.atomic_json(mp,sig)
    b.atomic_json(out/'normalization.json',info);b.atomic_json(out/'split_used.json',split)
    baseline=(1500,.95,.25,1.,.5); anchors=[baseline,(1500,.6,.1,1.,5.),(1500,.3,.1,1.,5.),(1500,.45,.05,.75,10.),(1500,.3,.05,1.,10.),(1500,.6,.05,1.,10.)]; combos=list(dict.fromkeys(list(itertools.product(a.reservoirs,a.radii,a.leaks,a.gains,a.ridges))+anchors)); groups={}
    for n,r,l,g,ridge in combos:groups.setdefault((n,r,l,g),[]).append(ridge)
    cache=out/'cache';cache.mkdir(exist_ok=True); rows=[]; weights={}
    def getweights(n):
        if n not in weights:weights[n]=b.common_weights(n,a.seed,1.,device)
        return weights[n]
    def model_for(row):
        return make_model(getweights(row['reservoir']),row['reservoir'],row['radius'],row['leak'],row['gain'],a.seed,device)
    for j,((n,r,l,g),ridges) in enumerate(groups.items()):
        path=cache/f'group_{j:03d}.json'
        if path.exists():rr=json.loads(path.read_text())
        else:
            model=make_model(getweights(n),n,r,l,g,a.seed,device);G,H=gram(model,data['train']);rr=[]
            for ridge in ridges:
                mat=G.clone();mat.diagonal().add_(ridge);mat[0,0]-=ridge;model.readout=torch.linalg.solve(mat,H).float();m=metrics(model,data['validation']);m={k:(v if math.isfinite(v) else None) for k,v in m.items()};rr.append(dict(id=f'C{len(rows)+len(rr):03d}',reservoir=n,radius=r,leak=l,gain=g,ridge=ridge,**m))
            b.atomic_json(path,rr)
        rows.extend(rr);b.csv_write(out/'E31C_forward_sweep.csv',rows); print(f'E31C group {j+1}/{len(groups)} | {len(rows)}/{len(combos)}',flush=True)
    valid=[x for x in rows if x['nonfinite']==0 and x['auto_nrmse'] is not None and x['tf_nrmse'] is not None];assert valid
    ref=next(x for x in rows if tuple(x[k] for k in ('reservoir','radius','leak','gain','ridge'))==baseline)
    # Union of best autonomous and best teacher-forced configs, plus reference.
    anchor_rows=[x for x in valid if tuple(x[k] for k in ('reservoir','radius','leak','gain','ridge')) in anchors]
    diverse=[]
    for leak in a.leaks:
        pool=[x for x in valid if x['leak']==leak]
        diverse+=sorted(pool,key=lambda x:x['auto_nrmse'])[:3]+sorted(pool,key=lambda x:x['tf_nrmse'])[:2]
    short={x['id']:x for x in sorted(valid,key=lambda x:x['auto_nrmse'])[:a.shortlist]+sorted(valid,key=lambda x:x['tf_nrmse'])[:a.shortlist]+diverse+anchor_rows}; gridrows=[]; rank=[]
    for row in short.values():
        model=model_for(row);model.fit(data['train'],row['ridge']);fp=cache/f"grid_{row['id']}.json"
        if fp.exists():rr=json.loads(fp.read_text())
        else:rr=landscape(model,data['validation'],lo,hi,[10,12,223],a.grid_mm,a.grid_batch);b.atomic_json(fp,rr)
        gridrows.extend(dict(id=row['id'],**x) for x in rr)
        rank.append(dict(**row,grid_xy10=float(np.mean([x['grid_xy_error_mm'] for x in rr if x['frames']==10])),grid_xy12=float(np.mean([x['grid_xy_error_mm'] for x in rr if x['frames']==12])),grid_xy223=float(np.mean([x['grid_xy_error_mm'] for x in rr if x['frames']==223])),true_win_pct=100*float(np.mean([x['true_better_than_grid'] for x in rr]))));print('E31C grid',rank[-1],flush=True)
        b.csv_write(out/'E31_grid_ranking.csv',rank);b.csv_write(out/'E31_grid_details.csv',gridrows)
    rank.sort(key=lambda x:((x['grid_xy12']+x['grid_xy223'])/2,x['auto_nrmse'] if x['auto_nrmse'] is not None else math.inf)); finalists={x['id']:x for x in rank[:a.finalists]+sorted(rank,key=lambda x:x['grid_xy12'])[:4]+sorted(rank,key=lambda x:x['grid_xy223'])[:4]+sorted(rank,key=lambda x:x['auto_nrmse'])[:4]+[x for x in rank if x['id'] in {q['id'] for q in anchor_rows}]}
    starts=torch.tensor(np.random.default_rng(a.seed).uniform(-.95,.95,(a.restarts,2)),device=device,dtype=torch.float32);starts[0]=0
    def evaluate(row,sn):
        model=model_for(row);model.fit(data['train'],row['ridge']);folder=out/row['id'];folder.mkdir(exist_ok=True);part=folder/sn;part.mkdir(exist_ok=True);d=data[sn];rr=[]
        b.atomic_pt(folder/'cesn.pt',dict(W=model.W.cpu(),Wc=model.Wc.cpu(),Wf=model.Wf.cpu(),bias=model.bias.cpu(),readout=model.readout.cpu(),normalization=info,split=split,leak=row['leak'],parameters=row))
        for k in [1,2,3,5,10,12,23,45,67,112,157,223]:
            for i in range(0,len(d['keys']),a.trajectory_batch):
                fp=part/f'prefix_{k:03d}_{i:03d}.json';ids=list(range(i,min(i+a.trajectory_batch,len(d['keys']))))
                if fp.exists():chunk=json.loads(fp.read_text())
                else:chunk=b.invert(model,d,ids,k,starts,a,lo,hi,scale);b.atomic_json(fp,chunk)
                rr.extend(chunk)
            sub=[x for x in rr if x['frames']==k];print('E31C',row['id'],sn,k,'accuracy',100*np.mean([x['correct'] for x in sub]),'XY',np.mean([x['xy_error_mm'] for x in sub]),flush=True)
        b.csv_write(part/'per_trajectory.csv',rr); summary=[dict(id=row['id'],split=sn,**x,future_nrmse=None if x['future_rmse'] is None else x['future_rmse']/scale) for x in b.summarize(rr)];b.csv_write(part/'summary.csv',summary);return summary
    summaries=[];scores=[]
    for row in finalists.values():
        ss=evaluate(row,'validation');summaries.extend(ss);allr={x['frames']:x for x in ss if x['subset']=='all'};pre={x['frames']:x for x in ss if x['subset']=='precontact'}
        early_frames=[k for k in (12,23,45,67,112) if k in pre and pre[k]['count']==allr[k]['count']]
        early_xy=float(np.mean([pre[k]['xy_mean_mm'] for k in early_frames])) if early_frames else None
        scores.append(dict(**row,accuracy5=allr[5]['accuracy_pct'],accuracy10=allr[10]['accuracy_pct'],accuracy12=allr[12]['accuracy_pct'],pain_recall12=pre.get(12,{}).get('pain_recall_pct'),precontact_count12=pre.get(12,{}).get('count',0),total_count12=allr[12]['count'],xy10=allr[10]['xy_mean_mm'],xy12=allr[12]['xy_mean_mm'],xy223=allr[223]['xy_mean_mm'],precontact_xy_mean=early_xy,precontact_frames=','.join(map(str,early_frames))))
        b.csv_write(out/'E31_comparison.csv',summaries);b.csv_write(out/'E31_finalists.csv',scores)
    def early_ok(x):
        return x['accuracy12']==100.0 and x['pain_recall12']==100.0 and x['precontact_count12']==x['total_count12'] and x['precontact_xy_mean'] is not None
    def rank_key(x):
        return (x['precontact_xy_mean'] if x['precontact_xy_mean'] is not None else math.inf,x['xy223'],x['auto_nrmse'])
    eligible=[x for x in scores if early_ok(x) and x['auto_nrmse']<.6 and x['xy223']<10.0]
    early=[x for x in scores if early_ok(x)]
    winner=min(eligible,key=rank_key) if eligible else None
    provisional=min(early,key=lambda x:(max(0,x['auto_nrmse']/.6-1)+max(0,x['xy223']/10-1),*rank_key(x))) if early else None
    result=dict(status='selected' if winner is not None else 'no_model_meets_all_constraints',selected=winner,provisional=provisional,eligible_count=len(eligible),early_qualified_count=len(early))
    b.atomic_json(out/'E31_selected.json',result)
    b.atomic_json(out/'E31C_tradeoffs.json',dict(**result,best_autonomous=min(scores,key=lambda x:x['auto_nrmse']),best_early_xy=min(scores,key=lambda x:x['xy12']),best_final_xy=min(scores,key=lambda x:x['xy223']),best_precontact_xy=min(early,key=rank_key) if early else None))
    print('SELECTED',result,flush=True)
    print('DONE',out,flush=True)
if __name__=='__main__':main()

"""E30: forward sweep, E31: spatial shortlist and prefix inversion. No augmentation."""
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
    p=argparse.ArgumentParser();p.add_argument('--reservoirs',nargs='+',type=int,default=[500,1000,1500]);p.add_argument('--radii',nargs='+',type=float,default=[.3,.6,.9,.95]);p.add_argument('--leaks',nargs='+',type=float,default=[.1,.25,.5,1.]);p.add_argument('--ridges',nargs='+',type=float,default=[.01,.1,.5,1.,5.]);p.add_argument('--gains',nargs='+',type=float,default=[.5,1.]);p.add_argument('--shortlist',type=int,default=6);p.add_argument('--finalists',type=int,default=3);p.add_argument('--grid-mm',type=float,default=5);p.add_argument('--grid-batch',type=int,default=128);p.add_argument('--restarts',type=int,default=9);p.add_argument('--max-iter',type=int,default=40);p.add_argument('--trajectory-batch',type=int,default=3);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');p.add_argument('--run-name',default='E30_E31_forward_sweep');p.add_argument('--evaluate-test',action='store_true');a=p.parse_args()
    assert min(a.reservoirs)>0 and min(a.radii)>0 and min(a.leaks)>0 and max(a.leaks)<=1 and min(a.ridges)>0 and min(a.gains)>=0 and min(a.shortlist,a.finalists,a.restarts,a.trajectory_batch,a.grid_batch)>0 and a.grid_mm>0
    device=torch.device(a.device);torch.manual_seed(a.seed);torch.set_float32_matmul_precision('high')
    ae=cfg.OUT/'autoencoder_lightweight128_matched';lat=ae/'cleaver_lightweight_latent128.h5';data,split=b.load(lat,ae/'split.json');data,split=b.grouped_validation(data,split);info,lo,hi=normalize(data,device);scale=info['dynamic_scale'];out=cfg.OUT/a.run_name;out.mkdir(parents=True,exist_ok=True)
    sig=dict(args={k:v for k,v in vars(a).items() if k!='evaluate_test'},split=split,latent=b.digest(lat),code=b.digest(Path(__file__)),core=b.digest(Path(b.__file__)),normalizer=b.digest(Path(__file__).with_name('run_dimensions.py'))); mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==sig,'Changed settings: choose a new --run-name'
    else:b.atomic_json(mp,sig)
    b.atomic_json(out/'normalization.json',info);b.atomic_json(out/'split_used.json',split)
    baseline=(1500,.95,.25,1.,.5); combos=list(dict.fromkeys(list(itertools.product(a.reservoirs,a.radii,a.leaks,a.gains,a.ridges))+[baseline])); groups={}
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
        rows.extend(rr);b.csv_write(out/'E30_forward_sweep.csv',rows); print(f'E30 group {j+1}/{len(groups)} | {len(rows)}/{len(combos)}',flush=True)
    valid=[x for x in rows if x['nonfinite']==0 and x['auto_nrmse'] is not None and x['tf_nrmse'] is not None];assert valid
    ref=next(x for x in rows if tuple(x[k] for k in ('reservoir','radius','leak','gain','ridge'))==baseline)
    # Union of best autonomous and best teacher-forced configs, plus reference.
    short={x['id']:x for x in sorted(valid,key=lambda x:x['auto_nrmse'])[:a.shortlist]+sorted(valid,key=lambda x:x['tf_nrmse'])[:a.shortlist]+[ref]}; gridrows=[]; rank=[]
    for row in short.values():
        model=model_for(row);model.fit(data['train'],row['ridge']);fp=cache/f"grid_{row['id']}.json"
        if fp.exists():rr=json.loads(fp.read_text())
        else:rr=landscape(model,data['validation'],lo,hi,[10,223],a.grid_mm,a.grid_batch);b.atomic_json(fp,rr)
        gridrows.extend(dict(id=row['id'],**x) for x in rr)
        rank.append(dict(**row,grid_xy10=float(np.mean([x['grid_xy_error_mm'] for x in rr if x['frames']==10])),grid_xy223=float(np.mean([x['grid_xy_error_mm'] for x in rr if x['frames']==223])),true_win_pct=100*float(np.mean([x['true_better_than_grid'] for x in rr]))));print('E31 grid',rank[-1],flush=True)
        b.csv_write(out/'E31_grid_ranking.csv',rank);b.csv_write(out/'E31_grid_details.csv',gridrows)
    rank.sort(key=lambda x:((x['grid_xy10']+x['grid_xy223'])/2,x['auto_nrmse'] if x['auto_nrmse'] is not None else math.inf)); finalists={x['id']:x for x in rank[:a.finalists]+[next(x for x in rank if x['id']==ref['id'])]}
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
            sub=[x for x in rr if x['frames']==k];print('E31',row['id'],sn,k,'accuracy',100*np.mean([x['correct'] for x in sub]),'XY',np.mean([x['xy_error_mm'] for x in sub]),flush=True)
        b.csv_write(part/'per_trajectory.csv',rr); summary=[dict(id=row['id'],split=sn,**x,future_nrmse=None if x['future_rmse'] is None else x['future_rmse']/scale) for x in b.summarize(rr)];b.csv_write(part/'summary.csv',summary);return summary
    summaries=[];scores=[]
    for row in finalists.values():
        ss=evaluate(row,'validation');summaries.extend(ss);allr={x['frames']:x for x in ss if x['subset']=='all'};pre={x['frames']:x for x in ss if x['subset']=='precontact'}
        scores.append(dict(**row,accuracy5=allr[5]['accuracy_pct'],accuracy10=allr[10]['accuracy_pct'],pain_recall5=pre.get(5,{}).get('pain_recall_pct'),xy10=allr[10]['xy_mean_mm'],xy223=allr[223]['xy_mean_mm']));b.csv_write(out/'E31_comparison.csv',summaries);b.csv_write(out/'E31_finalists.csv',scores)
    # Safety classification gate: no degradation vs reference at 5 and 10 frames or precontact pain recall.
    refscore=next(x for x in scores if x['id']==ref['id']);eligible=[x for x in scores if x['accuracy5']>=refscore['accuracy5'] and x['accuracy10']>=refscore['accuracy10'] and x['pain_recall5'] is not None and refscore['pain_recall5'] is not None and x['pain_recall5']>=refscore['pain_recall5']]
    winner=min(eligible or [refscore],key=lambda x:((x['xy10']+x['xy223'])/2,x['auto_nrmse']));b.atomic_json(out/'E31_selected.json',winner);print('SELECTED',winner,flush=True)
    if a.evaluate_test:
        for row in {x['id']:x for x in [winner,refscore]}.values():summaries.extend(evaluate(row,'test'))
        b.csv_write(out/'E31_comparison.csv',summaries)
    print('DONE',out,flush=True)
if __name__=='__main__':main()

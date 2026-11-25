"""Read-only diagnostics for round 2. No model fitting except train-only probes."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import torch,h5py
from scipy.spatial import ConvexHull, distance_matrix
from . import config as cfg
from . import run_four_cesn_v2 as v2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--grid-mm',type=float,default=2);p.add_argument('--batch',type=int,default=256);p.add_argument('--frames',nargs='+',type=int,default=[10,223]);p.add_argument('--models',nargs='+',default=['E06','E07','E08']);p.add_argument('--pairs-per-split',type=int,default=2);a=p.parse_args();assert a.grid_mm>0 and a.batch>0 and all(1<=n<=223 for n in a.frames)
 device=torch.device(a.device);torch.set_float32_matmul_precision('highest')
 ae=cfg.OUT/'autoencoder_lightweight128'; source=cfg.OUT/'four_cesn_round2_lightweight128';out=cfg.OUT/'diagnostics_lightweight128';out.mkdir(exist_ok=True)
 lat=ae/'cleaver_lightweight_latent128.h5'; data,split=v2.load(lat,ae/'split.json'); info,lo,hi=v2.normalize(data,device)
 manifest=json.loads((source/'manifest.json').read_text());assert manifest['latent_sha256']==v2.digest(lat) and manifest['split']==split,'Checkpoint data mismatch'
 # Geometry from original HDF5, never as input to CESN.
 with h5py.File(cfg.RGB,'r') as h:
  for name,d in data.items():
   robot=[];knife=[];objects=[];starts=[]
   for key,pair,xy in zip(d['keys'],d['pair'],d['xy']):
    g=h[key];assert int(g.attrs['pair_id'])==pair and np.allclose(g['target_world'][:2],xy)
    robot.append(np.asarray(g['tcp_xyz'][0])[:2]);knife.append(np.asarray(g['object_xyz'][0])[:2]);objects.append(int(g.attrs['object_id']));starts.append(int(g.attrs['start_id']))
   d.update(robot=np.stack(robot),knife=np.stack(knife),object_id=np.array(objects),start_id=np.array(starts))
 tr=data['train']; sm=tr['static'].astype('float64').mean(0);_,sv,pc=np.linalg.svd(tr['static']-sm,full_matrices=False)
 # One row per pair; avoid double-weighting identical initial images.
 unique={k:np.unique(d['pair'],return_index=True)[1] for k,d in data.items()}; ti=unique['train'];vi=unique['validation'];
 probe=[]; detail=[]
 for dim in [2,4,8,16,32,128]:
  x={k:(d['static']-sm)@(pc[:dim].T) if dim!=128 else d['static']-sm for k,d in data.items()};scale=max(float(np.sqrt(np.mean(x['train'][ti]**2))),1e-10);x={k:np.column_stack([np.ones(len(z)),z/scale]) for k,z in x.items()}
  for target in ['knife','robot','xy']:
   X=x['train'][ti];Y=tr[target][ti]; G=X.T@X;H=X.T@Y; scores=[]
   for ridge in [1e-6,1e-4,.01,.1,1,10,100]:
    reg=np.eye(len(G))*ridge;reg[0,0]=0;w=np.linalg.solve(G+reg,H); err=np.linalg.norm(x['validation'][vi]@w-data['validation'][target][vi],axis=1);scores.append((err.mean(),ridge,w))
   _,ridge,w=min(scores,key=lambda r:r[0]);
   for k,d in data.items():
    ii=unique[k];pred=x[k][ii]@w;err=1000*np.linalg.norm(pred-d[target][ii],axis=1);base=1000*np.linalg.norm(Y.mean(0)-d[target][ii],axis=1)
    probe.append(dict(representation='raw128' if dim==128 else f'PCA{dim}',target=target,split=k,ridge=ridge,mean_mm=err.mean(),median_mm=np.median(err),baseline_mean_mm=base.mean(),count=len(ii)))
    for j,i in enumerate(ii):detail.append(dict(representation=dim,target=target,split=k,pair_id=int(d['pair'][i]),true_x=d[target][i,0],true_y=d[target][i,1],pred_x=pred[j,0],pred_y=pred[j,1],error_mm=err[j]))
 v2.csv_write(out/'static_probes.csv',probe);v2.csv_write(out/'static_probe_predictions.csv',detail)
 for row in probe:
  if row['split']!='train':print('PROBE',row,flush=True)
 # D04: hull, nearest training position, IDs, pair coverage.
 centers=np.unique(tr['xy'].round(6),axis=0); hull=ConvexHull(centers);spatial=[];fig,ax=plt.subplots(1,2,figsize=(11,5))
 for k,d in data.items():
  ii=unique[k];xy=d['xy'][ii];inside=np.all(xy@hull.equations[:,:2].T+hull.equations[:,2]<=1e-6,axis=1);near=distance_matrix(xy,centers).min(1)*1000
  for j,i in enumerate(ii):spatial.append(dict(split=k,pair_id=int(d['pair'][i]),object_id=int(d['object_id'][i]),start_id=int(d['start_id'][i]),x_m=xy[j,0],y_m=xy[j,1],inside_train_hull=int(inside[j]),nearest_train_goal_mm=near[j]))
  ax[0].scatter(*xy.T,label=k,alpha=.65);ax[1].scatter(*d['robot'][ii].T,label=k,alpha=.65);print('SPATIAL',k,'inside',sum(inside),'/',len(ii),'nearest mean mm',near.mean(),flush=True)
 ring=centers[np.r_[hull.vertices,hull.vertices[0]]];ax[0].plot(*ring.T,'k--');
 for z,title in zip(ax,['Goal XY / train convex hull','Initial TCP XY']):z.set_title(title);z.set_aspect('equal');z.legend();z.set_xlabel('X (m)');z.set_ylabel('Y (m)')
 fig.tight_layout();fig.savefig(out/'spatial_coverage.png',dpi=160);plt.close(fig);v2.csv_write(out/'spatial_coverage.csv',spatial)
 # D03: true-label landscape, exact true goal loss, compare existing LBFGS candidate with same true label only when correct.
 low=lo.cpu().numpy();high=hi.cpu().numpy();axes=[np.linspace(low[j],high[j],int(np.ceil((high[j]-low[j])/(a.grid_mm/1000)))+1) for j in range(2)];gx,gy=np.meshgrid(*axes);grid=np.stack([gx.ravel(),gy.ravel()],1);gn=torch.tensor(2*(grid-low)/(high-low)-1,dtype=torch.float32,device=device);land=[]
 for name,dim,fb in v2.EXPS:
  if name.split('_')[0] not in a.models:continue
  ck=torch.load(source/name/'cesn.pt',map_location=device,weights_only=False);assert ck['pca']==dim and ck['feedback']==fb
  model=v2.CESN((ck['W'],torch.zeros(20,len(ck['bias']),device=device),ck['Wf'],ck['bias']),dim,fb,ck['leak']);model.Wc=ck['Wc'];model.readout=ck['readout']
  for k in ['validation','test']:
   d=data[k];pairs=np.unique(d['pair']); chosen=pairs[np.linspace(0,len(pairs)-1,min(a.pairs_per_split,len(pairs)),dtype=int)]
   with (source/name/k/'per_trajectory.csv').open() as f:old={(r['trajectory'],int(r['frames'])):r for r in csv.DictReader(f)}
   for i in np.flatnonzero(np.isin(d['pair'],chosen)):
    for n in a.frames:
     obs=d['yn'][i:i+1,:n];static=d['pca'][i:i+1];label=int(d['label'][i]);
     @torch.no_grad()
     def evaluate(x):
      B=len(x);c=model.context(torch.full((B,),label,device=device),static.expand(B,-1),x);y=obs.expand(B,-1,-1);return (model.predict(c,n,y if fb else None)-y).square().mean((1,2)).cpu().numpy()
     losses=np.concatenate([evaluate(x) for x in gn.split(a.batch)]);true=float(evaluate(d['xyn'][i:i+1])[0]);j=int(losses.argmin());best=float(losses[j]);dist=float(np.linalg.norm(grid[j]-d['xy'][i])*1000);r=old.get((d['keys'][i],n));lb=None;lb_error=None
     if r and int(r['pred_label'])==label:
      w=np.array([float(r['pred_x_m']),float(r['pred_y_m'])]);lb=float(evaluate(torch.tensor((2*(w-low)/(high-low)-1)[None],device=device,dtype=torch.float32))[0]);lb_error=float(r['xy_error_mm'])
     row=dict(model=name,split=k,trajectory=d['keys'][i],pair_id=int(d['pair'][i]),label=label,frames=n,true_loss=true,grid_min_loss=best,grid_xy_error_mm=dist,lbfgs_loss_correct_label=lb,lbfgs_xy_error_mm=lb_error,loss_range=float(losses.max()-best),true_minus_grid=true-best,grid_x=grid[j,0],grid_y=grid[j,1]);land.append(row)
     fig,ax=plt.subplots(figsize=(6,5));im=ax.pcolormesh(gx*1000,gy*1000,np.log10(np.maximum(losses.reshape(gx.shape),1e-12)),shading='auto');fig.colorbar(im,ax=ax,label='log10 normalized prefix MSE');ax.scatter(*(d['xy'][i]*1000),marker='*',s=130,c='white',edgecolors='black',label='True');ax.scatter(*(grid[j]*1000),marker='x',s=90,c='red',label='Grid min');ax.legend();ax.set(xlabel='X (mm)',ylabel='Y (mm)',title=f'{name} {k} {d["keys"][i]} K={n}');fig.tight_layout();fig.savefig(out/f'{name}_{k}_{d["keys"][i]}_K{n}.png',dpi=130);plt.close(fig);v2.csv_write(out/'xy_landscape.csv',land);print('LAND',name,k,d['keys'][i],n,'grid error mm',round(dist,2),'true loss',round(true,6),'min',round(best,6),flush=True)
 print('DONE',out,flush=True)
if __name__=='__main__':main()

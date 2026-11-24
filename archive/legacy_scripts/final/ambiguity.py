"""Spatial ambiguity, local Hessian and loss maps for the frozen CESN.

Task is deliberately known (oracle) to isolate continuous XY identifiability.
Local Hessian eigenvalues are loss curvature, NOT calibrated confidence.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from CESN.core import parametersCESN as p
from CESN.analysis.final.experiment_common import (
    OUTPUT,load_environment,selected_rows,K_from_fraction,invert_group,xy_of,
    save_csv,output_dir,save_metadata)


def surface_and_hessian(runner,row,K,point,bounds,grid_size):
    truth=np.asarray(row['joints'],np.float64); T=len(truth)
    task=torch.tensor(np.asarray(p.TASK_LABELS[str(row['kind']).lower()])[None],
                      dtype=runner.dtype,device=runner.device)
    target=torch.tensor(truth[:K][None],dtype=runner.dtype,device=runner.device)
    tt=torch.tensor([T],dtype=torch.long,device=runner.device)
    kk=torch.tensor([K],dtype=torch.long,device=runner.device)
    def scalar(z):
        return runner.prefix_loss(torch.cat([task,z],1),target,tt,kk)[0]
    x0=torch.tensor(np.asarray(point,dtype=float)[None],dtype=runner.dtype,
                    device=runner.device,requires_grad=True)
    H=torch.autograd.functional.hessian(scalar,x0).detach().cpu().numpy().reshape(2,2)
    H=(H+H.T)/2
    ev=np.linalg.eigvalsh(H)
    true_xy=xy_of(row)
    lo=np.minimum(np.asarray([bounds[0],bounds[2]]),true_xy)-.03
    hi=np.maximum(np.asarray([bounds[1],bounds[3]]),true_xy)+.03
    xs=np.linspace(lo[0],hi[0],grid_size);ys=np.linspace(lo[1],hi[1],grid_size)
    XX,YY=np.meshgrid(xs,ys);pts=np.column_stack([XX.ravel(),YY.ravel()]);chunks=[]
    with torch.no_grad():
        for start in range(0,len(pts),256):
            g=pts[start:start+256]; n=len(g)
            t=torch.tensor(np.repeat(task.detach().cpu().numpy(),n,0),dtype=runner.dtype,device=runner.device)
            ctx=torch.cat([t,torch.tensor(g,dtype=runner.dtype,device=runner.device)],dim=1)
            tgt=target.expand(n,-1,-1)
            Tt=torch.full((n,),T,dtype=torch.long,device=runner.device)
            Kt=torch.full((n,),K,dtype=torch.long,device=runner.device)
            chunks.append(runner.prefix_loss(ctx,tgt,Tt,Kt).cpu().numpy())
    return H,ev,XX,YY,np.concatenate(chunks).reshape(XX.shape)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',default='cuda');ap.add_argument('--dtype',default='float32',choices=('float32','float64'))
    ap.add_argument('--fractions',type=float,nargs='+',default=[.1,.3,.7])
    ap.add_argument('--max-cases',type=int,default=4)
    ap.add_argument('--restarts',type=int,default=16)
    ap.add_argument('--grid-size',type=int,default=70)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--out-dir',type=Path,default=OUTPUT/'ambiguity')
    ap.add_argument('--overwrite',action='store_true')
    a=ap.parse_args()
    if a.max_cases<1 or a.restarts<1 or a.grid_size<10:ap.error('invalid cases, restarts, grid')
    model,runner,test,train,bounds=load_environment(a.device,a.dtype)
    cases=selected_rows(test,a.max_cases)
    out=output_dir(a.out_dir,a.overwrite);(out/'figures').mkdir(exist_ok=True)
    rows=[]
    for fi,f in enumerate(a.fractions):
        for ci,row in enumerate(cases):
            K=K_from_fraction(len(row['joints']),f)
            result,stats=invert_group(runner,[row],[K],bounds,a.seed+fi*10007+ci,
                                      restarts=a.restarts,mode='known')
            est=result[0].estimated_xy;true=xy_of(row)
            H,ev,XX,YY,Z=surface_and_hessian(runner,row,K,est,bounds,a.grid_size)
            rows.append(dict(trajectory=str(row['id']),region=row.get('context_region','unknown'),
                             fraction=f,K=K,T=len(row['joints']),xy_error_mm=1000*np.linalg.norm(est-true),
                             h_xx=H[0,0],h_xy=H[0,1],h_yy=H[1,1],
                             eig_min=ev[0],eig_max=ev[1],
                             curvature_ratio=(ev[1]/ev[0] if ev[0]>0 else float('nan')),
                             optimizer='lbfgs',task_mode='known'))
            fig,ax=plt.subplots(figsize=(7,6));cs=ax.contourf(XX,YY,np.log10(np.maximum(Z,1e-15)),levels=32)
            fig.colorbar(cs,ax=ax,label='log10(prefix MSE)')
            ax.scatter(*true,s=180,marker='*',label='True XY')
            ax.scatter(*est,s=65,marker='D',label='L-BFGS XY')
            ax.set(xlabel='cube x [m]',ylabel='cube y [m]',title=f'{row["id"]}, K={K}/{len(row["joints"])}')
            ax.legend();fig.tight_layout()
            fig.savefig(out/'figures'/f'loss_{ci:02d}_K{K:03d}.png',dpi=160);plt.close(fig)
            print('ambiguity',row['id'],'fraction',f,'min eig',ev[0],flush=True)
    save_csv(out/'ambiguity_detail.csv',rows)
    save_metadata(out,experiment='ambiguity',task_mode='known',fractions=a.fractions,
                  n_cases=len(cases),restarts=a.restarts,seed=a.seed,
                  note='Eigenvalues are local curvature, not calibrated uncertainty; no test-independent holdout.')
    print('Saved:',out)

if __name__=='__main__':main()

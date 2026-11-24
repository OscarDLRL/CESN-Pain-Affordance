"""Compare selected unknown-task L-BFGS Method A to two ORIGINAL baselines.

Grid search and Nearest Demonstration do not use Adam. They are imported from
legacy_adam only to avoid duplicating their original evaluation definitions.
Outputs contain same trajectories/fractions for all three strategies.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from CESN.core import cesn
from CESN.analysis.final.experiment_common import (
    OUTPUT,load_environment,selected_rows,K_from_fraction,invert_group,xy_of,
    save_csv,summary,output_dir,save_metadata)
from CESN.analysis.legacy_adam.spatial_baselines import grid_search_case,nearest_demo_case


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full',action='store_true');ap.add_argument('--max-cases',type=int,default=None)
    ap.add_argument('--fractions',type=float,nargs='+',default=[.1,.3,.5,.7,.9])
    ap.add_argument('--grid-size',type=int,default=31)
    ap.add_argument('--restarts',type=int,default=4)
    ap.add_argument('--device',default='cuda');ap.add_argument('--dtype',choices=('float32','float64'),default='float32')
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--out-dir',type=Path,default=OUTPUT/'baselines')
    ap.add_argument('--overwrite',action='store_true')
    a=ap.parse_args()
    if a.grid_size<2 or a.restarts<1 or any(not 0<x<=1 for x in a.fractions):ap.error('invalid parameters')
    model,runner,test,train,bounds=load_environment(a.device,a.dtype)
    cases=selected_rows(test,a.max_cases if a.max_cases is not None else (None if a.full else 8))
    out=output_dir(a.out_dir,a.overwrite)
    # Original baseline grid spans training XY range, same as its legacy implementation.
    xs=np.linspace(bounds[0],bounds[1],a.grid_size);ys=np.linspace(bounds[2],bounds[3],a.grid_size)
    detail=[]
    for fi,f in enumerate(a.fractions):
        for i,row in enumerate(cases):
            T=len(row['joints']);K=K_from_fraction(T,f)
            inv,_=invert_group(runner,[row],[K],bounds,a.seed+fi*100003+i,a.restarts)
            baseline=[('A_discrete_lbfgs60',inv[0].predicted_task,inv[0].estimated_xy),
                      ('Grid Search',*grid_search_case(runner,row,f,xs,ys,runner.dtype)[:2]),
                      ('Nearest Demo',*nearest_demo_case(train,row,f)[:2])]
            for name,task,xy in baseline:
                xy=np.asarray(xy,float)
                pred=np.asarray(cesn.predict_trajectory(model,cesn.make_context(task,xy),T),float)
                truth=np.asarray(row['joints'],float)
                future=float(np.sqrt(np.mean((pred[K:]-truth[K:])**2))) if K<T else float('nan')
                detail.append(dict(trajectory=str(row['id']),region=str(row.get('context_region','unknown')),
                     fraction=f,T=T,K=K,method=name,true_task=str(row['kind']).lower(),pred_task=task,
                     task_correct=int(str(row['kind']).lower()==task),
                     xy_error_mm=1000*np.linalg.norm(xy-xy_of(row)),future_rmse=future))
            print('baselines fraction',f,'cases',i+1,'/',len(cases),flush=True)
    save_csv(out/'baselines_detail.csv',detail)
    ss=[]
    for method in ('A_discrete_lbfgs60','Grid Search','Nearest Demo'):
        for stat in summary([dict(r, predicted_task=r['pred_task']) for r in detail if r['method']==method]):
            ss.append(dict(method=method,**stat))
    save_csv(out/'baselines_summary.csv',ss)
    save_metadata(out,experiment='baselines',n_cases=len(cases),fractions=a.fractions,
                  grid_size=a.grid_size,seed=a.seed,restarts=a.restarts,
                  note='Baseline procedures reused from legacy; Method A per-case shared curvature across 8 candidates. Timing not paired.')
    print('Saved:',out)
if __name__=='__main__':main()

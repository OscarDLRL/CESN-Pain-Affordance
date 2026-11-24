"""Shared evaluation loop for unknown task, emergence (exact K), and noise.

Called by three small CLI entrypoints. Frozen CESN, Method A/L-BFGS 60 only.
"""
import argparse
from pathlib import Path
import numpy as np
from CESN.analysis.final.experiment_common import (
    ROOT, OUTPUT, load_environment, selected_rows, K_from_fraction, invert_group,
    score, save_csv, summary, output_dir, save_metadata,
)

FRACTIONS=[round(i/10,1) for i in range(1,11)]


def execute(kind):
    ap=argparse.ArgumentParser(description=f'CESN final {kind}: Method A + L-BFGS 60')
    ap.add_argument('--full',action='store_true')
    ap.add_argument('--max-cases',type=int,default=None)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--dtype',choices=('float32','float64'),default='float32')
    ap.add_argument('--batch-cases',type=int,default=64)
    ap.add_argument('--restarts',type=int,default=4)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--fractions',nargs='+',type=float,default=None)
    ap.add_argument('--k-values',nargs='+',type=int,default=None)
    ap.add_argument('--noise-levels',nargs='+',type=float,default=None)
    ap.add_argument('--out-dir',type=Path,default=OUTPUT/kind)
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    if args.batch_cases<1 or args.restarts<1: ap.error('batch-cases and restarts must be positive')
    if kind=='task_emergence':
        levels=args.k_values or ([1,2,3,5,10,17,30] if args.full else [1,2,3,5,10])
        if any(k<1 for k in levels):ap.error('K must be >= 1')
    else:
        levels=args.fractions or (FRACTIONS if args.full else [.3,.5,.7])
        if any(not 0<f<=1 for f in levels):ap.error('fraction must satisfy 0 < f <= 1')
    sigmas= (args.noise_levels or ([0,.001,.002,.005,.01,.02] if args.full else [0,.005])) if kind=='noise_robustness' else [0]
    if any(s<0 for s in sigmas):ap.error('noise sigma must be >= 0')
    model,runner,test,training,bounds=load_environment(args.device,args.dtype)
    cases=selected_rows(test,args.max_cases if args.max_cases is not None else (None if args.full else 8))
    if not cases:raise RuntimeError('No test cases')
    out=output_dir(args.out_dir,args.overwrite)
    details=[];batches=[]
    for li,level in enumerate(levels):
        for si,sigma in enumerate(sigmas):
            for start in range(0,len(cases),args.batch_cases):
                group=cases[start:start+args.batch_cases]
                Ks=[max(1,min(len(row['joints']),int(level))) if kind=='task_emergence'
                    else K_from_fraction(len(row['joints']),level) for row in group]
                # Identical noisy prefix for every XY restart and both task hypotheses.
                noise=None
                if sigma:
                    rng=np.random.default_rng(args.seed+li*1000003+si*10007+start)
                    noise=[np.asarray(row['joints'],float)[:K]+rng.normal(0,sigma,size=(K,9))
                           for row,K in zip(group,Ks)]
                init_seed=args.seed+li*100003+si*1009+start
                inversions,stats=invert_group(runner,group,Ks,bounds,init_seed,args.restarts,noise=noise)
                new=score(group,Ks,inversions,model,noise_sigma=sigma,
                          fraction=(float(level) if kind!='task_emergence' else None))
                if kind=='task_emergence':
                    for row in new:row['K_requested']=int(level)
                details+=new
                batches.append(dict(level=float(level),sigma=float(sigma),first_index=start,
                                    n_cases=len(group),n_candidates=len(group)*args.restarts*2,
                                    batch_elapsed_s=float(stats.elapsed_s),
                                    gradient_evaluations=int(stats.gradient_evaluations)))
                print(f'{kind}: level={level:g} sigma={sigma:g} cases={min(start+len(group),len(cases))}/{len(cases)}',flush=True)
    dims=('K_requested','region') if kind=='task_emergence' else (('fraction','noise_sigma','region') if kind=='noise_robustness' else ('fraction','region'))
    prefix=kind
    save_csv(out/f'{prefix}_detail.csv',details)
    save_csv(out/f'{prefix}_summary.csv',summary(details,dimensions=dims))
    save_csv(out/f'{prefix}_batches.csv',batches)
    save_metadata(out,experiment=kind,n_cases=len(cases),levels=levels,noise_sigmas=sigmas,
                  batch_cases=args.batch_cases,restarts=args.restarts,seed=args.seed,
                  device=args.device,dtype=args.dtype,source='TEST',
                  note=('K is EXACT frames, including K=1' if kind=='task_emergence' else
                        'Joint-space Gaussian noise uses raw mixed units (rad and m)' if kind=='noise_robustness'
                        else 'Unknown task, sigma=0'))
    print('Completed',kind,'rows=',len(details),'out=',out)

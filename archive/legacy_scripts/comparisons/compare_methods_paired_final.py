#!/usr/bin/env python3
"""Paired evaluation: discrete Method A / L-BFGS 60 vs softmax Method B / L-BFGS 120.

Uses the SAME frozen CESN, TEST trajectories, observation prefixes, sampled
noise and XY starting points. Each method gets 4 XY starts x 2 task/logit
hypotheses, with batched L-BFGS (shared curvature history). No retraining.

Quick (8 cases, 3 fractions, no noise):
    python3 -m CESN.analysis.comparisons.compare_methods_paired_final
Full (120 cases, 10 fractions, 6 noise levels):
    python3 -m CESN.analysis.comparisons.compare_methods_paired_final --full

Outputs per-trajectory and per-batch CSV, paired summaries, and a short
conclusion. Not an independent holdout: these methods/hyperparameters were
previously evaluated/tuned on TEST. Do not present as an unbiased final test.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from CESN.core import parametersCESN as p, cesn
from CESN.core.data import load_split
from CESN.rollout.gpu_batch_inversion import GPUBatchedCESN
from CESN.analysis.comparisons.benchmark_optimizers_full import (
    MODEL, cube_xy, select_balanced, run_method, write_csv,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / 'CESN/outputs/reports/paired_method_selection'
METHODS = (('A_discrete_lbfgs60', 'discrete', 60),
           ('B_softmax_lbfgs120', 'softmax', 120))


def aggregate(rows):
    xy = np.asarray([r['xy_error_mm'] for r in rows], float)
    fm = np.asarray([r['future_rmse'] for r in rows], float)
    return dict(n=len(rows), task_accuracy=float(np.mean([r['task_correct'] for r in rows])),
                xy_mean_mm=float(np.mean(xy)), xy_median_mm=float(np.median(xy)),
                future_rmse_mean=float(np.mean(fm[np.isfinite(fm)]))
                if np.isfinite(fm).any() else float('nan'))


def build_pairs(details):
    keyed = {}
    for r in details:
        k = (r['trajectory'], r['fraction'], r['noise_sigma'])
        keyed.setdefault(k, {})[r['method_label']] = r
    pairs = []
    for (trajectory, fraction, sigma), item in sorted(keyed.items()):
        if len(item) != 2:
            raise RuntimeError(f'Missing method for paired condition: {(trajectory,fraction,sigma)}')
        a = item[METHODS[0][0]]
        b = item[METHODS[1][0]]
        pairs.append(dict(trajectory=trajectory, region=a['region'], fraction=fraction,
                          noise_sigma=sigma, T=a['T'], K=a['K'],
                          a_task_correct=a['task_correct'], b_task_correct=b['task_correct'],
                          a_xy_error_mm=a['xy_error_mm'], b_xy_error_mm=b['xy_error_mm'],
                          a_future_rmse=a['future_rmse'], b_future_rmse=b['future_rmse'],
                          delta_xy_mm=b['xy_error_mm']-a['xy_error_mm'],
                          delta_task_correct=a['task_correct']-b['task_correct'],
                          delta_future_rmse=b['future_rmse']-a['future_rmse']
                          if np.isfinite(a['future_rmse']) and np.isfinite(b['future_rmse'])
                          else float('nan'),
                          xy_winner='A' if a['xy_error_mm'] < b['xy_error_mm']
                          else ('B' if b['xy_error_mm'] < a['xy_error_mm'] else 'tie')))
    return pairs


def grouped_summary(pairs):
    groups = defaultdict(list)
    for r in pairs:
        region = r['region']
        for key in [('GLOBAL','ALL','ALL'), (region,'ALL','ALL'),
                    ('GLOBAL',r['fraction'],'ALL'), ('GLOBAL','ALL',r['noise_sigma']),
                    (region,r['fraction'],r['noise_sigma'])]:
            groups[key].append(r)
    out = []
    for (region, fraction, sigma), rr in sorted(groups.items(),key=lambda z: str(z[0])):
        d = np.asarray([r['delta_xy_mm'] for r in rr],float)
        fd = np.asarray([r['delta_future_rmse'] for r in rr],float)
        out.append(dict(region=region, fraction=fraction, noise_sigma=sigma, n=len(rr),
                        a_task_accuracy=float(np.mean([r['a_task_correct'] for r in rr])),
                        b_task_accuracy=float(np.mean([r['b_task_correct'] for r in rr])),
                        a_xy_mean_mm=float(np.mean([r['a_xy_error_mm'] for r in rr])),
                        b_xy_mean_mm=float(np.mean([r['b_xy_error_mm'] for r in rr])),
                        b_minus_a_xy_mean_mm=float(d.mean()),
                        b_minus_a_xy_median_mm=float(np.median(d)),
                        a_future_rmse=float(np.nanmean([r['a_future_rmse'] for r in rr]))
                        if any(np.isfinite(r['a_future_rmse']) for r in rr) else float('nan'),
                        b_future_rmse=float(np.nanmean([r['b_future_rmse'] for r in rr]))
                        if any(np.isfinite(r['b_future_rmse']) for r in rr) else float('nan'),
                        b_minus_a_future_rmse=float(np.nanmean(fd)) if np.isfinite(fd).any() else float('nan'),
                        fraction_xy_a_wins=float(np.mean([r['xy_winner']=='A' for r in rr])),
                        fraction_xy_b_wins=float(np.mean([r['xy_winner']=='B' for r in rr]))))
    return out


def bootstrap_ci_by_trajectory(pairs, seed=42, n_boot=2000):
    """Cluster-bootstrap whole trajectories to retain within-trajectory dependence."""
    by_id = defaultdict(list)
    for r in pairs:
        by_id[r['trajectory']].append(r)
    ids = list(by_id)
    if len(ids) < 2:
        return float('nan'), float('nan')
    per_trajectory = np.asarray([np.mean([r['delta_xy_mm'] for r in by_id[k]]) for k in ids])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(ids), (n_boot, len(ids)))
    averages = per_trajectory[sampled].mean(axis=1)
    return tuple(np.percentile(averages,[2.5,97.5]).astype(float))


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--full',action='store_true',help='120 cases, 10 fractions, 6 noise conditions')
    ap.add_argument('--max-cases',type=int,default=None)
    ap.add_argument('--fractions',nargs='+',type=float,default=None)
    ap.add_argument('--noise-levels',nargs='+',type=float,default=None)
    ap.add_argument('--restarts',type=int,default=4)
    ap.add_argument('--batch-cases',type=int,default=8)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--lbfgs-lr',type=float,default=0.8)
    ap.add_argument('--logit-bias',type=float,default=1.0)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--dtype',choices=('float32','float64'),default='float32')
    ap.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()
    fractions = args.fractions if args.fractions is not None else ([i/10 for i in range(1,11)] if args.full else [.3,.5,.7])
    sigmas = args.noise_levels if args.noise_levels is not None else ([0,.001,.002,.005,.01,.02] if args.full else [0.])
    if args.restarts < 1 or args.batch_cases < 1 or (args.max_cases is not None and args.max_cases < 1):
        ap.error('restarts, batch-cases and max-cases must be positive')
    if not fractions or any(not (0 < v <= 1) for v in fractions) or not sigmas or any(v < 0 for v in sigmas):
        ap.error('fraction must be in (0,1], noise >= 0')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        ap.error('CUDA not available (or use --device cpu)')
    if args.out_dir.exists() and any((args.out_dir/name).exists() for name in ['detail.csv','paired.csv','summary.csv','batches.csv']):
        ap.error(f'{args.out_dir} already contains reports; choose a NEW --out-dir')

    model = cesn.load_model(MODEL)
    dtype = torch.float32 if args.dtype=='float32' else torch.float64
    runner = GPUBatchedCESN(model,p.LEAK_RATE,device=args.device,dtype=dtype)
    test=load_split(p.TEST_SPLIT,p.DATA_PATH,joint_dim=p.JOINT_DIM)
    n=args.max_cases if args.max_cases is not None else (None if args.full else 8)
    test=select_balanced(test,n)
    if not test:
        raise RuntimeError('Empty test split')
    train=[]
    for split in p.TRAIN_SPLITS:
        train.extend(load_split(split,p.DATA_PATH,joint_dim=p.JOINT_DIM))
    xy_train=np.stack([cube_xy(row) for row in train])
    mn,mx=xy_train.min(axis=0),xy_train.max(axis=0)
    args.out_dir.mkdir(parents=True,exist_ok=True)

    print('PAIRED A vs B - frozen CESN, LBFGS 60 vs 120',flush=True)
    print(f'cases={len(test)}, fractions={fractions}, sigma={sigmas}, restarts={args.restarts}',flush=True)
    print('All candidates use the same XY starts and noisy prefixes; only task representations differ.',flush=True)
    details=[]; batches=[]
    # Collect completed batches; final CSVs are written after the run finishes.
    for fi,frac in enumerate(fractions):
        for ni,sigma in enumerate(sigmas):
            for start in range(0,len(test),args.batch_cases):
                rows=test[start:start+args.batch_cases]
                rng=np.random.default_rng(args.seed+fi*100003+ni*1009+start)
                starts=np.empty((len(rows),args.restarts,2),np.float64)
                starts[:,:,0]=rng.uniform(mn[0],mx[0],size=starts.shape[:2])
                starts[:,:,1]=rng.uniform(mn[1],mx[1],size=starts.shape[:2])
                noise=[]
                for row in rows:
                    q=np.asarray(row['joints'],np.float64)
                    noise.append(q+rng.normal(0.,sigma,size=q.shape))
                for label,mode,iters in METHODS:
                    settings=SimpleNamespace(logit_bias=args.logit_bias,adam_lr=.02,adam_iters=150,
                                             patience=20,lbfgs_lr=args.lbfgs_lr,
                                             lbfgs_iters=iters,active_sigma=float(sigma))
                    results=run_method(runner,rows,frac,noise,starts.copy(),mode,'lbfgs',settings,model)
                    for r in results:
                        r['method_label']=label
                        r['lbfgs_max_iter']=iters
                    details.extend(results)
                    head=results[0]
                    batches.append(dict(method=label,fraction=float(frac),noise_sigma=float(sigma),
                                        n_cases=len(rows),n_candidates=head['batch_candidates'],
                                        first_trajectory=head['trajectory'],elapsed_s=head['batch_elapsed_s'],
                                        gradient_evaluations=head['batch_gradient_evals'],
                                        optimizer_iterations=head['batch_iterations']))
                print(f'fraction={frac:.2f}, sigma={sigma:.4f}, cases={start+len(rows)}/{len(test)}',flush=True)
    pairs=build_pairs(details)
    summaries=grouped_summary(pairs)
    write_csv(args.out_dir/'detail.csv',details)
    write_csv(args.out_dir/'paired.csv',pairs)
    write_csv(args.out_dir/'summary.csv',summaries)
    write_csv(args.out_dir/'batches.csv',batches)
    all_global=next(r for r in summaries if r['region']=='GLOBAL' and r['fraction']=='ALL' and r['noise_sigma']=='ALL')
    times={label: sum(x['elapsed_s'] for x in batches if x['method']==label) for label,_,_ in METHODS}
    ci=bootstrap_ci_by_trajectory(pairs)
    a_acc,b_acc=all_global['a_task_accuracy'],all_global['b_task_accuracy']
    a_xy,b_xy=all_global['a_xy_mean_mm'],all_global['b_xy_mean_mm']
    a_fut,b_fut=all_global['a_future_rmse'],all_global['b_future_rmse']
    ta,tb=times[METHODS[0][0]],times[METHODS[1][0]]
    # Transparent, explicit decision rule: A if at least as good on all four
    # key metrics (accuracy, xy, future RMSE, total execution time).
    a_dominates = a_acc>=b_acc and a_xy<=b_xy and a_fut<=b_fut and ta<=tb
    b_dominates = b_acc>=a_acc and b_xy<=a_xy and b_fut<=a_fut and tb<=ta
    verdict='A_discrete_lbfgs60' if a_dominates else ('B_softmax_lbfgs120' if b_dominates else 'No single method dominates every metric')
    conclusion=f'''# Paired CESN inversion method comparison\n\nThis evaluates the existing frozen CESN; no retraining. Both methods see identical cases, prefixes, noise and XY starts. A uses discrete Pick/Push task hypotheses with batched L-BFGS max_iter=60; B optimizes softmax logits plus XY with batched L-BFGS max_iter=120.\n\nCases: {len(test)}; observation fractions: {fractions}; noise sigmas: {sigmas}.\n\n| Metric | Method A | Method B |\n|---|---:|---:|\n| Task accuracy | {a_acc:.4%} | {b_acc:.4%} |\n| XY mean error (mm) | {a_xy:.3f} | {b_xy:.3f} |\n| Future RMSE (excluding K=T) | {a_fut:.6f} | {b_fut:.6f} |\n| Total batched GPU optimization time (s) | {ta:.2f} | {tb:.2f} |\n\nB-minus-A XY mean difference: {b_xy-a_xy:+.3f} mm. Cluster bootstrap 95% CI, resampling trajectories: [{ci[0]:+.3f}, {ci[1]:+.3f}] mm. This interval quantifies variation across TEST trajectories, **not** a new independent validation set.\n\n**Decision using four-metric Pareto dominance:** {verdict}. A smaller error or time is better, and higher task accuracy is better. If there is no dominance, examine the detailed tables before picking a tradeoff.\n\n**Scientific limitation:** Method B's 120-iteration setting was selected using prior TEST results; this is a paired confirmation on the same test set, not an unbiased independent holdout. A/B also have different optimizer iteration budgets by design. To claim generalization of the selected winner, tune on validation and test once on newly held-out trajectories.\n\nNote: K=T has no unobserved future; future RMSE is excluded there. Six sigma values include the unperturbed sigma=0 condition. Batch GPU times sum one timing per batch (not per trajectory).\n'''
    (args.out_dir/'conclusion.md').write_text(conclusion,encoding='utf-8')
    print('\n'+conclusion)
    print('Saved:',args.out_dir)

if __name__=='__main__':
    main()

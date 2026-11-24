#!/usr/bin/env python3
"""Known-task CESN XY inversion with selected batched L-BFGS optimizer.

Run from repository root:
    python3 -m CESN.analysis.final.known_task --quick --device cuda
    python3 -m CESN.analysis.final.known_task --full --batch-cases 64 --device cuda

Known task is provided deliberately: this is the continuous-parameter experiment,
NOT Method A's unknown-task classification experiment. No retraining. The optimizer
sees only an observed trajectory prefix, its length T, and the known task label.
Ground-truth XY and future positions are used only AFTER optimization for scoring.

Batched L-BFGS uses one shared curvature history; keep batch size, candidate
order, seed and dtype fixed for comparisons. Outputs will not overwrite an
existing non-empty directory unless --overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from CESN.core import cesn, parametersCESN as p
from CESN.core.data import load_split
from CESN.rollout.gpu_batch_inversion import GPUBatchedCESN
from CESN.inversion import infer_context

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = REPO / 'CESN/outputs/models/N_900/run_01/model.npz'
DEFAULT_OUT = REPO / 'CESN/outputs/final_lbfgs/known_task'
FRACTIONS = [round(i / 10, 1) for i in range(1, 11)]


def xy_of(row):
    cube = np.asarray(row['cube_xyz'], dtype=np.float64)
    return (cube[0] if cube.ndim == 2 else cube)[:2].copy()


def write_csv(path, records):
    if not records:
        raise ValueError(f'No records for {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(records):
    groups = {}
    for rec in records:
        for region in ('GLOBAL', rec['region']):
            groups.setdefault((rec['fraction'], region), []).append(rec)
    summary = []
    for (fraction, region), cases in sorted(groups.items()):
        xy = np.array([c['xy_error_mm'] for c in cases], np.float64)
        future = np.array([c['future_rmse'] for c in cases], np.float64)
        summary.append({
            'fraction': fraction, 'region': region, 'n': len(cases),
            'xy_error_mean_mm': float(xy.mean()),
            'xy_error_median_mm': float(np.median(xy)),
            'xy_error_q25_mm': float(np.quantile(xy, .25)),
            'xy_error_q75_mm': float(np.quantile(xy, .75)),
            'prefix_loss_mean': float(np.mean([c['prefix_loss'] for c in cases])),
            'future_rmse_mean': (float(future[np.isfinite(future)].mean())
                                 if np.isfinite(future).any() else float('nan')),
        })
    return summary


def plot_summary(rows, filename):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: skipping plot')
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for region, label in [('interpolation', 'Interpolation'),
                          ('extrapolation', 'Extrapolation')]:
        rr = sorted((r for r in rows if r['region'] == region), key=lambda r: r['fraction'])
        if not rr:
            continue
        x = 100 * np.array([r['fraction'] for r in rr])
        med = np.array([r['xy_error_median_mm'] for r in rr])
        lo = np.array([r['xy_error_q25_mm'] for r in rr])
        hi = np.array([r['xy_error_q75_mm'] for r in rr])
        line, = ax.plot(x, med, marker='o', label=f'{label}: median XY error')
        ax.fill_between(x, lo, hi, alpha=.18, color=line.get_color())
    ax.set(xlabel='Observed trajectory (%)', ylabel='XY error (mm)',
           title='Known-task XY inversion — frozen CESN, batched L-BFGS (60 iterations)')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true', help='8 balanced cases, fractions 0.3/0.5/0.7')
    mode.add_argument('--full', action='store_true', help='All test cases, ten fractions (no noise)')
    ap.add_argument('--max-cases', type=int, default=None,
                    help='Optional number of balanced test cases; overrides quick/full count')
    ap.add_argument('--fractions', type=float, nargs='+', default=None)
    ap.add_argument('--restarts', type=int, default=4)
    ap.add_argument('--batch-cases', type=int, default=64)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--lbfgs-iters', type=int, default=60)
    ap.add_argument('--lbfgs-lr', type=float, default=.8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    ap.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--overwrite', action='store_true', help='Explicitly replace output files')
    args = ap.parse_args()
    if args.restarts < 1 or args.batch_cases < 1 or args.lbfgs_iters < 1:
        ap.error('restarts, batch-cases, lbfgs-iters must be positive')
    if args.max_cases is not None and args.max_cases < 1:
        ap.error('max-cases must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        ap.error('CUDA unavailable: pass --device cpu')
    fractions = (args.fractions if args.fractions is not None
                 else (FRACTIONS if args.full else [.3, .5, .7]))
    if not fractions or any(not 0 < f <= 1 for f in fractions):
        ap.error('fraction values must satisfy 0 < f <= 1')
    out = args.out_dir.resolve()
    expected_paths = [out / x for x in ('known_task_detail.csv', 'known_task_summary.csv',
                                      'known_task_batches.csv', 'config.json',
                                      'known_task_xy_error.png')]
    if not args.overwrite and any(path.exists() for path in expected_paths):
        ap.error(f'Output files already exist in {out}; choose --out-dir or --overwrite')
    if not args.model.is_file():
        ap.error(f'Model not found: {args.model}')
    if not Path(p.DATA_PATH).is_file():
        ap.error(f'Dataset not found: {p.DATA_PATH}')

    # Import the balanced-selection helper from comparisons. It is deterministic
    # and intentionally matches the selected benchmark's case ordering.
    from CESN.analysis.comparisons.benchmark_optimizers_full import select_balanced

    rows = load_split(p.TEST_SPLIT, p.DATA_PATH, joint_dim=p.JOINT_DIM)
    n_cases = args.max_cases if args.max_cases is not None else (None if args.full else 8)
    rows = select_balanced(rows, n_cases)
    if not rows:
        raise RuntimeError('No test rows selected')
    training_rows = []
    for split in p.TRAIN_SPLITS:
        training_rows.extend(load_split(split, p.DATA_PATH, joint_dim=p.JOINT_DIM))
    if not training_rows:
        raise RuntimeError('No training cases to derive XY restart bounds')
    bounds = np.stack([xy_of(row) for row in training_rows])
    xy_min, xy_max = bounds.min(axis=0), bounds.max(axis=0)

    model = cesn.load_model(args.model)
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    runner = GPUBatchedCESN(model, p.LEAK_RATE, device=args.device, dtype=dtype)
    details, batches = [], []
    print(f'Known task | cases={len(rows)}, fractions={fractions}, starts={args.restarts}, '
          f'batch_cases={args.batch_cases}, device={args.device}, dtype={args.dtype}', flush=True)
    for fi, fraction in enumerate(fractions):
        for start in range(0, len(rows), args.batch_cases):
            group = rows[start:start + args.batch_cases]
            # Match benchmark's deterministic initializations: fi and start;
            # noise index is zero because known-task report is noise-free.
            rng = np.random.default_rng(args.seed + fi * 100003 + start)
            starts = np.empty((len(group), args.restarts, 2), np.float64)
            starts[:, :, 0] = rng.uniform(xy_min[0], xy_max[0], size=starts.shape[:2])
            starts[:, :, 1] = rng.uniform(xy_min[1], xy_max[1], size=starts.shape[:2])
            full = [np.asarray(row['joints'], np.float64) for row in group]
            Ts = np.array([len(y) for y in full], np.int64)
            Ks = np.maximum(2, np.minimum(Ts, np.rint(fraction * Ts).astype(np.int64)))
            prefixes = [full[i][:int(Ks[i])].copy() for i in range(len(group))]
            tasks = [str(row['kind']).lower() for row in group]
            results = infer_context(
                runner, prefixes, Ts, starts,
                method='known', known_tasks=tasks, optimizer='lbfgs',
                lbfgs_max_iter=args.lbfgs_iters, lbfgs_lr=args.lbfgs_lr,
            )
            stats = results[0].batch_stats
            batches.append({
                'fraction': fraction, 'first_trajectory': str(group[0]['id']),
                'n_cases': len(group), 'n_candidates': len(group) * args.restarts,
                'elapsed_s': float(stats.elapsed_s),
                'gradient_evaluations': int(stats.gradient_evaluations),
                'iterations': int(stats.iterations),
            })
            for row, truth, K, result in zip(group, full, Ks, results):
                xy_true = xy_of(row)
                xy_pred = result.estimated_xy
                task_encoding = np.asarray(p.TASK_LABELS[str(row['kind']).lower()], np.float64)
                context = np.r_[task_encoding, xy_pred]
                pred = np.asarray(cesn.predict_trajectory(model, context, len(truth)), np.float64)
                future = (float(np.sqrt(np.mean((pred[int(K):] - truth[int(K):]) ** 2)))
                          if int(K) < len(truth) else float('nan'))
                details.append({
                    'trajectory': str(row['id']),
                    'region': str(row.get('context_region', 'unknown')),
                    'task': str(row['kind']).lower(),
                    'optimizer': 'lbfgs', 'task_mode': 'known',
                    'fraction': float(fraction), 'T': len(truth), 'K': int(K),
                    'true_x': float(xy_true[0]), 'true_y': float(xy_true[1]),
                    'estimated_x': float(xy_pred[0]), 'estimated_y': float(xy_pred[1]),
                    'xy_error_mm': float(1000 * np.linalg.norm(xy_pred - xy_true)),
                    'prefix_loss': float(result.prefix_loss), 'future_rmse': future,
                    'selected_restart': int(result.selected_restart),
                })
            print(f'fraction={fraction:.2f}: cases={start + len(group)}/{len(rows)}', flush=True)

    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(details)
    write_csv(out / 'known_task_detail.csv', details)
    write_csv(out / 'known_task_summary.csv', summary)
    write_csv(out / 'known_task_batches.csv', batches)
    config = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_sha(),
        'experiment': 'known_task_xy', 'method': 'known', 'optimizer': 'lbfgs',
        'lbfgs_max_iter': args.lbfgs_iters, 'lbfgs_lr': args.lbfgs_lr,
        'batch_curvature_history_shared': True,
        'batch_cases': args.batch_cases, 'restarts': args.restarts,
        'seed': args.seed, 'fractions': fractions, 'n_cases': len(rows),
        'device': args.device, 'dtype': args.dtype,
        'model': str(args.model.resolve()), 'model_sha256': sha256(args.model),
        'dataset': str(Path(p.DATA_PATH).resolve()), 'dataset_sha256': sha256(p.DATA_PATH),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'notes': 'Test trajectories evaluated; not an independent holdout. '
                 'Known task is supplied; no unknown-task classification. '
                 'No noise, no CESN retraining. K=T excluded from future RMSE.',
    }
    (out / 'config.json').write_text(json.dumps(config, indent=2, default=str) + '\n', encoding='utf-8')
    plot_summary(summary, out / 'known_task_xy_error.png')
    print('\nResults saved:', out)
    print('XY error (median mm) by fraction and region:')
    for r in summary:
        if r['region'] != 'GLOBAL':
            print(f"  fraction={r['fraction']:.2f} region={r['region']:<14} "
                  f"n={r['n']:3d} median={r['xy_error_median_mm']:.3f}")
    print('Timing is recorded ONCE per batch in known_task_batches.csv.')


if __name__ == '__main__':
    main()

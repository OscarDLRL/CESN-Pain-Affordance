#!/usr/bin/env python3
"""Method B (softmax) optimizer sensitivity study for the frozen spatial CESN.

Run from repository root after installing alongside optimizer_backend.py and
benchmark_optimizers_full.py in CESN/analysis:

    python3 -m CESN.analysis.comparisons.benchmark_softmax_optimizers
    python3 -m CESN.analysis.comparisons.benchmark_softmax_optimizers --full
    python3 -m CESN.analysis.comparisons.benchmark_softmax_optimizers --full --noise-levels 0 0.005 0.02

Compares several Adam schedules to L-BFGS on IDENTICAL test trajectories,
initial XY/logits, observed prefixes and noise draws. No model retraining.

Outputs in CESN/outputs/reports/softmax_optimizer_study/:
  detail.csv   per-trajectory results (selected best of 2*restarts)
  batches.csv  true batch timings and gradient/closure evaluations
  summary.csv  grouped accuracy / XY / future RMSE
  config.csv   optimizer settings and budgets

NOTE: Adam's early stopping monitors the batch mean (same as older scripts).
L-BFGS batched shares one curvature history across all candidate restarts.
Gradient evaluations mean *full-batch backward calls*, not individual cases.
Even matched gradient counts are not strictly equal compute; compare time too.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from CESN.core import parametersCESN as p, cesn
from CESN.core.data import load_split
from CESN.rollout.gpu_batch_inversion import GPUBatchedCESN
from CESN.analysis.comparisons.benchmark_optimizers_full import (
    MODEL, cube_xy, prepare, select_balanced, summarize as summarize_base,
    write_csv, run_method,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / 'CESN/outputs/reports/softmax_optimizer_study'


@dataclass(frozen=True)
class Configuration:
    name: str
    optimizer: str
    lr: float
    max_iters: int
    patience: int | None


CONFIGS = (
    Configuration('adam_baseline', 'adam', 0.02, 150, 20),
    Configuration('adam_60', 'adam', 0.02, 60, None),
    Configuration('adam_300', 'adam', 0.02, 300, None),
    Configuration('adam_600', 'adam', 0.02, 600, None),
    Configuration('adam_600_lr005', 'adam', 0.005, 600, None),
    Configuration('adam_600_lr05', 'adam', 0.05, 600, None),
    Configuration('lbfgs_60', 'lbfgs', 0.8, 60, None),
    Configuration('lbfgs_120', 'lbfgs', 0.8, 120, None),
)


def append_csv(path: Path, records):
    """Write each batch as it finishes, so an interrupted long run keeps data."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(records)


def summarize_by_configuration(records):
    # Retain noise and fraction, split out interpolation/extrapolation.
    groups = {}
    for r in records:
        for region in ('GLOBAL', r['region']):
            key = (r['configuration'], r['fraction'], r['noise_sigma'], region)
            groups.setdefault(key, []).append(r)
    result = []
    for (cfg, frac, sigma, region), rr in sorted(groups.items()):
        xy = np.array([r['xy_error_mm'] for r in rr], dtype=float)
        future = np.array([r['future_rmse'] for r in rr], dtype=float)
        result.append(dict(
            configuration=cfg, fraction=frac, noise_sigma=sigma,
            region=region, n=len(rr),
            task_accuracy=float(np.mean([r['task_correct'] for r in rr])),
            xy_error_mean_mm=float(xy.mean()),
            xy_error_median_mm=float(np.median(xy)),
            prefix_loss_mean=float(np.mean([r['prefix_loss'] for r in rr])),
            future_rmse_mean=(float(np.nanmean(future))
                              if np.isfinite(future).any() else float('nan')),
        ))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--full', action='store_true',
                    help='Use all TEST cases and 10 fractions; default noise-free')
    ap.add_argument('--max-cases', type=int, default=None)
    ap.add_argument('--fractions', type=float, nargs='+', default=None)
    ap.add_argument('--noise-levels', type=float, nargs='+', default=[0.0])
    ap.add_argument('--configs', nargs='+', choices=[c.name for c in CONFIGS],
                    default=None)
    ap.add_argument('--restarts', type=int, default=4,
                    help='Each XY start is repeated with Pick/Push biased logits')
    ap.add_argument('--batch-cases', type=int, default=8)
    ap.add_argument('--logit-bias', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', choices=['float32','float64'], default='float32')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.restarts <= 0 or args.batch_cases <= 0:
        ap.error('restarts and batch-cases must be positive')
    if args.max_cases is not None and args.max_cases <= 0:
        ap.error('max-cases must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        ap.error('CUDA unavailable; use --device cpu if desired')
    fractions = (args.fractions if args.fractions is not None else
                 ([i / 10 for i in range(1, 11)] if args.full else [.3, .5, .7]))
    if not fractions or not all(0 < f <= 1 for f in fractions):
        ap.error('Fractions must be in (0, 1]')
    if not args.noise_levels or not all(s >= 0 for s in args.noise_levels):
        ap.error('Noise levels must be >= 0')
    configs = [c for c in CONFIGS if args.configs is None or c.name in args.configs]

    # Avoid silently overwriting prior results. Explicitly choose a fresh dir.
    for n in ('detail.csv', 'summary.csv', 'batches.csv', 'config.csv'):
        if (args.out_dir / n).exists():
            ap.error(f'{args.out_dir / n} already exists; use a NEW --out-dir')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / 'config.csv', [dict(
        configuration=c.name, optimizer=c.optimizer,
        learning_rate=c.lr, max_iterations=c.max_iters,
        early_stop_patience=('disabled' if c.patience is None else c.patience),
        restarts_xy=args.restarts, total_softmax_candidates_per_case=2 * args.restarts,
    ) for c in configs])

    model = cesn.load_model(MODEL)
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    runner = GPUBatchedCESN(model, leak_rate=p.LEAK_RATE,
                            device=args.device, dtype=dtype)
    test = load_split(p.TEST_SPLIT, p.DATA_PATH, joint_dim=p.JOINT_DIM)
    n = args.max_cases if args.max_cases is not None else (None if args.full else 8)
    test = select_balanced(test, n)
    if not test:
        raise RuntimeError('No test cases were selected')
    training_rows = []
    for split in p.TRAIN_SPLITS:
        training_rows += load_split(split, p.DATA_PATH, joint_dim=p.JOINT_DIM)
    train_xy = np.stack([cube_xy(r) for r in training_rows])
    xy_min, xy_max = train_xy.min(axis=0), train_xy.max(axis=0)

    # These match the previous full benchmark seed logic. All configurations
    # see exactly the same starts / observation noise in each condition.
    details, batches = [], []
    print(f'Method B softmax: test cases={len(test)}, fractions={fractions}, '
          f'noise={args.noise_levels}, configs={[c.name for c in configs]}', flush=True)
    for fi, frac in enumerate(fractions):
        for ni, sigma in enumerate(args.noise_levels):
            for start in range(0, len(test), args.batch_cases):
                rows = test[start:start + args.batch_cases]
                rng = np.random.default_rng(args.seed + fi * 100003 + ni * 1009 + start)
                starts = np.empty((len(rows), args.restarts, 2), np.float64)
                starts[:, :, 0] = rng.uniform(xy_min[0], xy_max[0], size=starts.shape[:2])
                starts[:, :, 1] = rng.uniform(xy_min[1], xy_max[1], size=starts.shape[:2])
                noisy = []
                for row in rows:
                    y = np.asarray(row['joints'], np.float64)
                    noisy.append(y + rng.normal(0., sigma, size=y.shape))

                for c in configs:
                    settings = SimpleNamespace(
                        logit_bias=args.logit_bias,
                        adam_lr=c.lr if c.optimizer == 'adam' else 0.02,
                        adam_iters=c.max_iters if c.optimizer == 'adam' else 150,
                        patience=c.patience,
                        lbfgs_lr=c.lr if c.optimizer == 'lbfgs' else 0.8,
                        lbfgs_iters=c.max_iters if c.optimizer == 'lbfgs' else 60,
                        active_sigma=float(sigma),
                    )
                    result = run_method(runner, rows, frac, noisy, starts.copy(),
                                        'softmax', c.optimizer, settings, model)
                    for r in result:
                        r['configuration'] = c.name
                    details.extend(result)
                    head = result[0]
                    batch = dict(
                        configuration=c.name, optimizer=c.optimizer,
                        fraction=float(frac), noise_sigma=float(sigma),
                        first_trajectory=head['trajectory'], n_cases=len(rows),
                        n_candidates=head['batch_candidates'],
                        elapsed_s=head['batch_elapsed_s'],
                        gradient_evaluations=head['batch_gradient_evals'],
                        iterations=head['batch_iterations'],
                    )
                    batches.append(batch)
                    append_csv(args.out_dir / 'detail.csv', result)
                    append_csv(args.out_dir / 'batches.csv', [batch])
                print(f'fraction={frac:.2f}, sigma={sigma:.4f}, '
                      f'cases={start+len(rows)}/{len(test)}', flush=True)

    write_csv(args.out_dir / 'summary.csv', summarize_by_configuration(details))
    print('\nSaved:')
    for name in ('config.csv', 'detail.csv', 'batches.csv', 'summary.csv'):
        print(' ', args.out_dir / name)
    print('IMPORTANT: configurations with 600 Adam iterations may be slow. '
          'A 100% task score alone does not prove early intention recognition.', flush=True)


if __name__ == '__main__':
    main()

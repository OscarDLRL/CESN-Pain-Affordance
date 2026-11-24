#!/usr/bin/env python3
"""Turn an EXISTING full Adam/L-BFGS benchmark into tables and figures.

No training and no optimization. Reads detail.csv and batches.csv from the
original, matched benchmark. Times are summed ONCE per batch from batches.csv.
Outputs numerical CSV and PNG only. Does not write conclusions or README files.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / 'CESN/outputs/reports/optimizer_benchmark_full'
DEFAULT_OUTPUT = ROOT / 'CESN/outputs/final_lbfgs/final_report'
MODES = ('known', 'discrete', 'softmax')
OPTS = ('adam', 'lbfgs')


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    if not records:
        raise ValueError(f'No rows for {path.name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print('Wrote:', path)


def floating(v):
    return float(v)


def groups(rows, keys):
    d = defaultdict(list)
    for r in rows:
        d[tuple(r[k] for k in keys)].append(r)
    return d


def summarize(detail_rows, batch_rows, *, sigma):
    dd = [r for r in detail_rows if float(r['noise_sigma']) == sigma
          and r['task_mode'] in MODES and r['optimizer'] in OPTS]
    bb = [r for r in batch_rows if float(r['noise_sigma']) == sigma
          and r['task_mode'] in MODES and r['optimizer'] in OPTS]
    dgroup = groups(dd, ('task_mode', 'optimizer', 'fraction'))
    bgroup = groups(bb, ('task_mode', 'optimizer', 'fraction'))
    by_fraction = []
    summary = []
    for mode in MODES:
        for opt in OPTS:
            subset = [r for r in dd if r['task_mode'] == mode and r['optimizer'] == opt]
            batches = [r for r in bb if r['task_mode'] == mode and r['optimizer'] == opt]
            if not subset or not batches:
                continue
            # Each (trajectory, fraction, noise) is a unique observation.
            keys = [(r['trajectory'], r['fraction'], r['noise_sigma']) for r in subset]
            if len(keys) != len(set(keys)):
                raise ValueError(f'Duplicate evaluated cases for {mode}/{opt}')
            frac_values = sorted({float(r['fraction']) for r in subset})
            summary.append(dict(task_mode=mode, optimizer=opt, sigma=sigma,
                n_trajectories=len(set(r['trajectory'] for r in subset)),
                n_evaluations=len(subset), n_fractions=len(frac_values),
                task_accuracy=float(np.mean([floating(r['task_correct']) for r in subset])),
                xy_error_mean_mm=float(np.mean([floating(r['xy_error_mm']) for r in subset])),
                xy_error_median_mm=float(np.median([floating(r['xy_error_mm']) for r in subset])),
                future_rmse_mean=float(np.mean([floating(r['future_rmse']) for r in subset])),
                elapsed_s_sum_batches=sum(floating(r['elapsed_s']) for r in batches),
                gradient_evaluations_sum_batches=sum(int(r['gradient_evaluations']) for r in batches),
                iterations_sum_batches=sum(int(r['iterations']) for r in batches),
                n_batches=len(batches)))
            for frac in frac_values:
                ds = [r for r in subset if floating(r['fraction']) == frac]
                bs = [r for r in batches if floating(r['fraction']) == frac]
                by_fraction.append(dict(task_mode=mode, optimizer=opt, sigma=sigma,
                    fraction=frac, n=len(ds),
                    task_accuracy=np.mean([floating(r['task_correct']) for r in ds]),
                    xy_error_mean_mm=np.mean([floating(r['xy_error_mm']) for r in ds]),
                    xy_error_median_mm=np.median([floating(r['xy_error_mm']) for r in ds]),
                    future_rmse_mean=np.mean([floating(r['future_rmse']) for r in ds]),
                    elapsed_s_sum_batches=sum(floating(r['elapsed_s']) for r in bs),
                    gradient_evaluations_sum_batches=sum(int(r['gradient_evaluations']) for r in bs),
                    iterations_sum_batches=sum(int(r['iterations']) for r in bs)))
    # Require identical evaluated trajectory/fraction/noise sets for fair comparisons.
    for mode in MODES:
        keys_by_opt = {
            opt: {(r['trajectory'], r['fraction'], r['noise_sigma'])
                  for r in dd if r['task_mode'] == mode and r['optimizer'] == opt}
            for opt in OPTS
        }
        if any(keys_by_opt.values()) and keys_by_opt['adam'] != keys_by_opt['lbfgs']:
            raise ValueError(f'Non-matching paired cases for {mode}: '
                             f'Adam={len(keys_by_opt["adam"])} LBFGS={len(keys_by_opt["lbfgs"])}')
    pairs = []
    for mode in MODES:
        a = next((r for r in summary if r['task_mode'] == mode and r['optimizer'] == 'adam'), None)
        b = next((r for r in summary if r['task_mode'] == mode and r['optimizer'] == 'lbfgs'), None)
        if not a or not b:
            continue
        for key in ('n_trajectories', 'n_evaluations', 'n_fractions'):
            if a[key] != b[key]:
                raise ValueError(f'Unequal paired benchmark counts: {mode} {key}')
        pairs.append(dict(task_mode=mode, sigma=sigma,
            n_evaluations=a['n_evaluations'],
            adam_time_s=a['elapsed_s_sum_batches'], lbfgs_time_s=b['elapsed_s_sum_batches'],
            speedup_adam_over_lbfgs=a['elapsed_s_sum_batches']/b['elapsed_s_sum_batches'],
            adam_gradient_evaluations=a['gradient_evaluations_sum_batches'],
            lbfgs_gradient_evaluations=b['gradient_evaluations_sum_batches'],
            adam_xy_mean_mm=a['xy_error_mean_mm'], lbfgs_xy_mean_mm=b['xy_error_mean_mm'],
            adam_task_accuracy=a['task_accuracy'], lbfgs_task_accuracy=b['task_accuracy'],
            adam_future_rmse=a['future_rmse_mean'], lbfgs_future_rmse=b['future_rmse_mean']))
    return dd, bb, summary, by_fraction, pairs


def paired_detail(detail, mode='discrete'):
    by = groups([r for r in detail if r['task_mode'] == mode],
                ('trajectory', 'fraction', 'noise_sigma'))
    result = []
    for key, records in sorted(by.items()):
        recs = {r['optimizer']: r for r in records}
        if len(records) != 2 or set(recs) != set(OPTS):
            raise ValueError(f'Benchmark not fully paired for {key}')
        a, b = recs['adam'], recs['lbfgs']
        if a['true_task'] != b['true_task'] or a['region'] != b['region']:
            raise ValueError(f'Mismatched case for {key}')
        result.append(dict(trajectory=key[0], fraction=key[1], noise_sigma=key[2],
            region=a['region'], true_task=a['true_task'],
            adam_predicted_task=a['predicted_task'], lbfgs_predicted_task=b['predicted_task'],
            adam_xy_error_mm=a['xy_error_mm'], lbfgs_xy_error_mm=b['xy_error_mm'],
            adam_future_rmse=a['future_rmse'], lbfgs_future_rmse=b['future_rmse'],
            adam_task_correct=a['task_correct'], lbfgs_task_correct=b['task_correct']))
    return result


def plot(by_frac, out):
    data = [r for r in by_frac if r['task_mode'] == 'discrete']
    fig, ax = plt.subplots(figsize=(7.7, 4.7))
    for opt in OPTS:
        rows = sorted((r for r in data if r['optimizer'] == opt), key=lambda r: r['fraction'])
        ax.plot([r['fraction']*100 for r in rows], [r['elapsed_s_sum_batches'] for r in rows],
                marker='o', label=opt.upper() if opt == 'lbfgs' else 'Adam')
    ax.set(xlabel='Observed trajectory (%)', ylabel='Total batch time (s)',
           title='Adam vs L-BFGS — discrete task hypotheses (Method A)')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out/'12_adam_vs_lbfgs_time.png', dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    metrics = [('xy_error_mean_mm', 'Mean XY error (mm)'),
               ('task_accuracy', 'Task accuracy'),
               ('future_rmse_mean', 'Future RMSE')]
    for ax, (key, ylabel) in zip(axes, metrics):
        for opt in OPTS:
            rows = sorted((r for r in data if r['optimizer'] == opt), key=lambda r:r['fraction'])
            ax.plot([r['fraction']*100 for r in rows], [r[key] for r in rows],
                    marker='o', label=opt.upper() if opt == 'lbfgs' else 'Adam')
        ax.set(xlabel='Observed trajectory (%)', ylabel=ylabel)
        ax.grid(alpha=.25)
    axes[0].legend(); fig.suptitle('Method A — Adam vs L-BFGS (noise = 0)')
    fig.tight_layout(); fig.savefig(out/'13_adam_vs_lbfgs_quality.png', dpi=200)
    plt.close(fig)
    for p in sorted(out.glob('1[23]_adam_vs_lbfgs_*.png')):
        print('Wrote:', p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--noise-sigma', type=float, default=0.0)
    args = ap.parse_args()
    detail_path = args.source/'detail.csv'
    batch_path = args.source/'batches.csv'
    if not detail_path.is_file() or not batch_path.is_file():
        ap.error(f'Missing benchmark input: {detail_path} or {batch_path}')
    details, batches = read_csv(detail_path), read_csv(batch_path)
    required_detail = {'trajectory','task_mode','optimizer','fraction','noise_sigma',
                       'region','true_task','predicted_task','task_correct','xy_error_mm','future_rmse'}
    required_batch = {'task_mode','optimizer','fraction','noise_sigma',
                      'elapsed_s','gradient_evaluations','iterations','n_cases'}
    if not details or required_detail - set(details[0]):
        ap.error(f'Invalid detail.csv schema, missing {required_detail-set(details[0] if details else {})}')
    if not batches or required_batch - set(batches[0]):
        ap.error(f'Invalid batches.csv schema, missing {required_batch-set(batches[0] if batches else {})}')
    filtered, filtered_batches, summary, fractions, pairs = summarize(details, batches, sigma=args.noise_sigma)
    # Check number of cases represented by batch-level timing; no row-level timing sum.
    for mode in MODES:
        for opt in OPTS:
            n_detail = sum(r['task_mode'] == mode and r['optimizer'] == opt for r in filtered)
            n_batch = sum(int(r['n_cases']) for r in filtered_batches
                          if r['task_mode'] == mode and r['optimizer'] == opt)
            if n_detail != n_batch:
                ap.error(f'Incomplete/mismatched benchmark for {mode}/{opt}: '
                         f'{n_detail} detail rows vs {n_batch} cases in timed batches')
    if not any(r['task_mode']=='discrete' for r in pairs):
        ap.error('No paired Adam and L-BFGS discrete results at chosen noise level')
    tables = args.output/'tables'; figures = args.output/'figures'
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    write_csv(tables/'12_adam_vs_lbfgs_summary.csv', summary)
    write_csv(tables/'13_adam_vs_lbfgs_by_fraction.csv', fractions)
    write_csv(tables/'14_adam_vs_lbfgs_paired_cases.csv', paired_detail(filtered))
    write_csv(tables/'15_adam_vs_lbfgs_comparison.csv', pairs)
    plot(fractions, figures)
    for r in pairs:
        print(f"{r['task_mode']:8s} | Adam {r['adam_time_s']:.2f} s | "
              f"LBFGS {r['lbfgs_time_s']:.2f} s | "
              f"speedup {r['speedup_adam_over_lbfgs']:.2f}x | "
              f"XY {r['adam_xy_mean_mm']:.2f} vs {r['lbfgs_xy_mean_mm']:.2f} mm")
    print('NOTE: Times are per-batch sums, not per-row duplicates.')
    print('NOTE: This benchmark is exploratory; check settings before treating it as the selected final protocol.')

if __name__ == '__main__':
    main()

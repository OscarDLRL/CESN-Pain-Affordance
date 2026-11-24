#!/usr/bin/env python3
"""Compare complete CESN inversion strategies, not optimizers alone.

Uses results of benchmark_optimizers_full.py. No CESN retraining.
Compare on exactly matched trajectory / observation fraction / noise.

Main decision: compare A_discrete+L-BFGS against B_softmax+L-BFGS;
Adam variants are reported as controls. Known-task is an oracle reference,
not a valid contender when the task is unknown.

Run (from repository root):
    python3 -m CESN.analysis.comparisons.select_best_inversion_method
    python3 -m CESN.analysis.comparisons.select_best_inversion_method --detail path/to/detail.csv --batches path/to/batches.csv

Output: CESN/outputs/reports/method_selection/ (CSV and conclusion.md).
The method winner is conditional on these benchmark settings; tuning Adam
on the same TEST set would not be a fresh independent evaluation.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / 'CESN/outputs/reports/optimizer_benchmark_full'
OUT = ROOT / 'CESN/outputs/reports/method_selection'
METHODS = {
    ('discrete', 'adam'): 'A_discrete_adam',
    ('discrete', 'lbfgs'): 'A_discrete_lbfgs',
    ('softmax', 'adam'): 'B_softmax_adam',
    ('softmax', 'lbfgs'): 'B_softmax_lbfgs',
    ('known', 'adam'): 'known_oracle_adam',
    ('known', 'lbfgs'): 'known_oracle_lbfgs',
}
KEY = ('trajectory', 'fraction', 'noise_sigma')


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean_of(rr, field):
    a = np.array([float(x[field]) for x in rr], float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float('nan')


def summarize_details(detail):
    grouped = defaultdict(list)
    for r in detail:
        method = METHODS.get((r['task_mode'], r['optimizer']))
        if method is None:
            continue
        grouped[(method, 'ALL')].append(r)
        grouped[(method, r['region'])].append(r)
        grouped[(method, 'sigma=' + r['noise_sigma'])].append(r)
        grouped[(method, 'fraction=' + r['fraction'])].append(r)
    out = []
    for (method, condition), rr in sorted(grouped.items()):
        out.append(dict(method=method, condition=condition, n=len(rr),
                        task_accuracy=mean_of(rr,'task_correct'),
                        xy_mean_mm=mean_of(rr,'xy_error_mm'),
                        xy_median_mm=float(np.median([float(r['xy_error_mm']) for r in rr])),
                        future_rmse=mean_of(rr,'future_rmse'),
                        prefix_loss=mean_of(rr,'prefix_loss')))
    return out


def time_summary(batches):
    groups = defaultdict(list)
    for row in batches:
        m = METHODS.get((row['task_mode'], row['optimizer']))
        if m is not None:
            groups[m].append(row)
    out = []
    for method, rr in sorted(groups.items()):
        # Batch runtimes summed ONCE; never sum repeated per-case timing in detail.csv.
        t = np.array([float(r['elapsed_s']) for r in rr])
        n = np.array([int(r['n_cases']) for r in rr])
        out.append(dict(method=method, batches=len(rr), cases_processed=int(n.sum()),
                        total_time_s=float(t.sum()), avg_batch_time_s=float(t.mean()),
                        time_per_case_s=float(t.sum()/n.sum()),
                        mean_full_batch_backward_calls=mean_of(rr,'gradient_evaluations')))
    return out


def paired_comparison(detail, left, right):
    by_method = defaultdict(dict)
    for r in detail:
        m = METHODS.get((r['task_mode'], r['optimizer']))
        if m is None:
            continue
        key = tuple(r[k] for k in KEY)
        if key in by_method[m]:
            raise ValueError(f'Duplicate result for {m}, {key}')
        by_method[m][key] = r
    la, rb = by_method[left], by_method[right]
    if set(la) != set(rb):
        raise ValueError(f'Unmatched cases: {left} vs {right}, left_only={len(set(la)-set(rb))}, right_only={len(set(rb)-set(la))}')
    out = []
    for k in sorted(la):
        a, b = la[k], rb[k]
        if a['true_task'] != b['true_task'] or a['region'] != b['region']:
            raise ValueError('Mismatched labels or regions')
        out.append(dict(trajectory=k[0], fraction=k[1], noise_sigma=k[2],
                        region=a['region'],
                        A_correct=int(a['task_correct']), B_correct=int(b['task_correct']),
                        A_xy_mm=float(a['xy_error_mm']), B_xy_mm=float(b['xy_error_mm']),
                        B_minus_A_xy_mm=float(b['xy_error_mm'])-float(a['xy_error_mm']),
                        A_future_rmse=float(a['future_rmse']),
                        B_future_rmse=float(b['future_rmse'])))
    return out


def paired_stats(rr, seed=2026, n_boot=2000):
    diff = np.array([r['B_minus_A_xy_mm'] for r in rr], float)
    ok_a = np.array([r['A_correct'] for r in rr], float)
    ok_b = np.array([r['B_correct'] for r in rr], float)
    fa = np.array([r['A_future_rmse'] for r in rr], float)
    fb = np.array([r['B_future_rmse'] for r in rr], float)
    valid_f = np.isfinite(fa) & np.isfinite(fb)
    # Group bootstrap by trajectory, because fractions/noise for the same
    # trajectory are correlated (not 7200 independent robot trajectories).
    trajectories = sorted(set(r['trajectory'] for r in rr))
    idx_by_id = {tid: np.array([i for i,r in enumerate(rr) if r['trajectory']==tid])
                 for tid in trajectories}
    if not trajectories:
        raise ValueError('No paired cases')
    rng = np.random.default_rng(seed)
    boot_xy = []
    boot_task = []
    for _ in range(n_boot):
        ids = rng.integers(0, len(trajectories), size=len(trajectories))
        indices = np.concatenate([idx_by_id[trajectories[j]] for j in ids])
        boot_xy.append(float(diff[indices].mean()))
        boot_task.append(float((ok_a[indices]-ok_b[indices]).mean()))
    return dict(n_conditions=len(rr), n_unique_trajectories=len(trajectories),
                A_accuracy=float(ok_a.mean()), B_accuracy=float(ok_b.mean()),
                A_xy_mean_mm=float(np.mean([r['A_xy_mm'] for r in rr])),
                B_xy_mean_mm=float(np.mean([r['B_xy_mm'] for r in rr])),
                B_minus_A_xy_mean_mm=float(diff.mean()),
                B_minus_A_xy_cluster_bootstrap_95lo_mm=float(np.quantile(boot_xy,.025)),
                B_minus_A_xy_cluster_bootstrap_95hi_mm=float(np.quantile(boot_xy,.975)),
                A_minus_B_accuracy=float((ok_a-ok_b).mean()),
                A_minus_B_accuracy_cluster_bootstrap_95lo=float(np.quantile(boot_task,.025)),
                A_minus_B_accuracy_cluster_bootstrap_95hi=float(np.quantile(boot_task,.975)),
                A_better_xy_fraction=float(np.mean(diff>1e-6)),
                B_better_xy_fraction=float(np.mean(diff< -1e-6)),
                A_future_rmse=float(fa[valid_f].mean()) if valid_f.any() else float('nan'),
                B_future_rmse=float(fb[valid_f].mean()) if valid_f.any() else float('nan'))


def fmt(x, places=3):
    return f'{x:.{places}f}' if np.isfinite(x) else 'N/A'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--detail', type=Path, default=INPUT/'detail.csv')
    ap.add_argument('--batches', type=Path, default=INPUT/'batches.csv')
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--bootstrap', type=int, default=2000)
    args = ap.parse_args()
    detail = read(args.detail)
    batches = read(args.batches)
    if not detail or not batches:
        raise ValueError('detail.csv or batches.csv empty')
    summary = summarize_details(detail)
    times = time_summary(batches)
    paired = paired_comparison(detail, 'A_discrete_lbfgs', 'B_softmax_lbfgs')
    total = paired_stats(paired, n_boot=args.bootstrap)
    by_region = [dict(condition=region, **paired_stats([r for r in paired if r['region']==region], n_boot=args.bootstrap))
                 for region in ('interpolation', 'extrapolation')]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.out_dir/'method_metrics.csv', summary)
    save_csv(args.out_dir/'method_timings.csv', times)
    save_csv(args.out_dir/'paired_A_vs_B_lbfgs.csv', paired)
    save_csv(args.out_dir/'paired_summary.csv', [dict(condition='ALL', **total)] + by_region)
    globally = {r['method']:r for r in summary if r['condition']=='ALL'}
    timing = {r['method']:r for r in times}
    a = globally['A_discrete_lbfgs']; b = globally['B_softmax_lbfgs']
    ta=timing['A_discrete_lbfgs']; tb=timing['B_softmax_lbfgs']
    # A takes priority when task unknown: task accuracy, then XY, future, compute.
    # This is a descriptive choice (NOT an optimized external test-set score).
    winner = 'A_discrete_lbfgs' if (
        a['task_accuracy'] >= b['task_accuracy']-1e-12 and
        a['xy_mean_mm'] <= b['xy_mean_mm']+1e-9 and
        a['future_rmse'] <= b['future_rmse']+1e-9 and
        ta['time_per_case_s'] <= tb['time_per_case_s']+1e-9
    ) else 'mixed_tradeoff_review_required'
    lines = [
        '# CESN strategy selection — unknown task', '',
        '**Scope:** fixed trained model; known-task is an oracle reference, not an eligible real-world strategy.',
        '**Methods compared:** A = two discrete Pick/Push hypotheses with optimized XY; B = softmax logits optimized jointly with XY.',
        '**Paired conditions:** same trajectory, observation fraction, noise level and benchmark settings.',
        '', '## Primary comparison: both methods use L-BFGS', '',
        '| Metric | Method A | Method B |', '|---|---:|---:|',
        f"| Task accuracy | {100*a['task_accuracy']:.2f}% | {100*b['task_accuracy']:.2f}% |",
        f"| Mean XY error | {fmt(a['xy_mean_mm'])} mm | {fmt(b['xy_mean_mm'])} mm |",
        f"| Median XY error | {fmt(a['xy_median_mm'])} mm | {fmt(b['xy_median_mm'])} mm |",
        f"| Future RMSE | {fmt(a['future_rmse'], 5)} | {fmt(b['future_rmse'], 5)} |",
        f"| Total batch GPU time | {fmt(ta['total_time_s'], 1)} s | {fmt(tb['total_time_s'], 1)} s |",
        f"| Time per case-condition | {fmt(ta['time_per_case_s'], 4)} s | {fmt(tb['time_per_case_s'], 4)} s |",
        '', '## Paired differences', '',
        f"- {total['n_conditions']} paired conditions from {total['n_unique_trajectories']} distinct trajectories.",
        f"- Mean XY advantage of A (B minus A): {fmt(total['B_minus_A_xy_mean_mm'])} mm.",
        f"- Cluster bootstrap 95% interval: [{fmt(total['B_minus_A_xy_cluster_bootstrap_95lo_mm'])}, {fmt(total['B_minus_A_xy_cluster_bootstrap_95hi_mm'])}] mm (descriptive; trajectories resampled).",
        f"- A has lower XY error in {100*total['A_better_xy_fraction']:.1f}% of matched conditions; B in {100*total['B_better_xy_fraction']:.1f}%.",
        '', f'## Working decision: `{winner}`', '',
        'If A wins each major metric here, prefer **Method A + L-BFGS** for unknown-task inversion: '
        'it preserves the discrete task code, has lower spatial error and future error, and is cheaper.',
        'Method B + L-BFGS can still be useful if smooth task probabilities are needed.',
        '', '## Important experimental caveats', '',
        '- Results are on the current TEST set; selecting/tuning by this set makes it a development set for model selection.',
        '- To make a definitive generalization claim, freeze the method and its hyperparameters, then validate on fresh untouched trajectories.',
        '- Adam may be under-tuned; comparison against a tuned Adam requires a separate development subset.',
        '- Sigma in the joint-space loss mixes radians and meters; consider per-joint scaling in a separate, clearly labeled experiment.',
        '- Method A and B have different parameterizations; lowest loss does not automatically imply best XY.',
        '- The batch L-BFGS curvature history is shared across restarts, and gradient counts are batch evaluations.',
        '- K=T has no unobserved future, so future RMSE excludes those conditions.',
        '', '## All four unknown-task strategies', '',
        '| Strategy | Accuracy | Mean XY mm | Future RMSE | Total GPU seconds |',
        '|---|---:|---:|---:|---:|'
    ]
    for m in ['A_discrete_adam','A_discrete_lbfgs','B_softmax_adam','B_softmax_lbfgs']:
        s=globally[m]; t=timing[m]
        lines.append(f"| {m} | {100*s['task_accuracy']:.2f}% | {s['xy_mean_mm']:.3f} | {s['future_rmse']:.5f} | {t['total_time_s']:.1f} |")
    (args.out_dir/'conclusion.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Unknown-task strategies (ALL conditions):')
    for m in ['A_discrete_adam','A_discrete_lbfgs','B_softmax_adam','B_softmax_lbfgs']:
        s=globally[m]; t=timing[m]
        print(f" {m:20s} acc={s['task_accuracy']:.3%} XY={s['xy_mean_mm']:.2f}mm future={s['future_rmse']:.5f} GPU={t['total_time_s']:.1f}s")
    print('Working decision:', winner)
    print('Saved:', args.out_dir)


if __name__ == '__main__':
    main()

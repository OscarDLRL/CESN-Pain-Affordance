#!/usr/bin/env python3
"""Generate an evidence-traceable L-BFGS spatial inversion report (no training/GPU).

Uses the actual frozen CESN known-task experiment and the paired A/B L-BFGS
comparison. Never silently imports Adam experiment data. Produces only figures
whose supporting results exist. Legacy qualitative/hessian/baselines plots
require porting their inversion to the selected engine; marked pending.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / 'CESN/outputs/final_lbfgs/final_report'
DEFAULT_KNOWN = ROOT / 'CESN/outputs/final_lbfgs/known_task/known_task_detail.csv'
PAIRED_SOURCES = (
    ROOT / 'CESN/outputs/reports/paired_method_selection/detail.csv',
)
METHOD_A = 'A_discrete_lbfgs60'
METHOD_B = 'B_softmax_lbfgs120'
FRACTION_TOL = 1e-8


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f'CSV is empty: {path}')
    return rows


def number(row, key):
    return float(row[key])


def save_csv(path, records):
    if not records:
        return
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def unique_fractions(rows):
    return sorted(set(float(r['fraction']) for r in rows))


def grouped_stats(rows, key):
    out = []
    for fraction in unique_fractions(rows):
        for region in ('GLOBAL', 'interpolation', 'extrapolation'):
            subset = [r for r in rows if abs(number(r, 'fraction') - fraction) < FRACTION_TOL
                      and (region == 'GLOBAL' or r['region'] == region)]
            if not subset:
                continue
            values = np.array([number(r, key) for r in subset], dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            out.append(dict(fraction=fraction, region=region, n=len(values),
                            mean=float(np.mean(values)), median=float(np.median(values)),
                            q25=float(np.quantile(values, .25)),
                            q75=float(np.quantile(values, .75))))
    return out


def get_curve(stats, region):
    rows = sorted((r for r in stats if r['region'] == region), key=lambda r: r['fraction'])
    return (np.array([100*r['fraction'] for r in rows]),
            np.array([r['median'] for r in rows]),
            np.array([r['q25'] for r in rows]),
            np.array([r['q75'] for r in rows]))


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches='tight')
    plt.close(fig)


def plot_known(detail, fig_dir, table_dir, generated):
    required = {'region','fraction','xy_error_mm','future_rmse'}
    if not required.issubset(detail[0]):
        raise ValueError(f'Known-task detail missing columns: {required - set(detail[0])}')
    # Known-task data must come from the selected engine, not original Adam results.
    if 'optimizer' in detail[0] and any(str(r['optimizer']).lower() != 'lbfgs' for r in detail):
        raise ValueError('Known-task input contains non-LBFGS records')
    if 'task_mode' in detail[0] and any(str(r['task_mode']).lower() != 'known' for r in detail):
        raise ValueError('Known-task input contains wrong task mode')
    xy = grouped_stats(detail, 'xy_error_mm')
    save_csv(table_dir/'04_context_error_median_iqr.csv', [
        {'fraction': r['fraction'], 'region': r['region'], 'n': r['n'],
         'mean_xy_error_mm': r['mean'], 'median_xy_error_mm': r['median'],
         'q25_xy_error_mm': r['q25'], 'q75_xy_error_mm': r['q75']} for r in xy])
    generated.append('tables/04_context_error_median_iqr.csv')

    fig, ax = plt.subplots(figsize=(8.6,5.2))
    for region, label in [('interpolation','Interpolation'),('extrapolation','Extrapolation')]:
        xx, med, q25, q75 = get_curve(xy, region)
        if len(xx):
            ln, = ax.plot(xx, med, marker='o', label=f'{label}: median')
            ax.fill_between(xx, q25, q75, color=ln.get_color(), alpha=.18)
    ax.set(xlabel='Observed trajectory (%)', ylabel='XY error (mm)',
           title='Known task: XY inversion with L-BFGS (60 iterations)')
    ax.set_yscale('log'); ax.grid(alpha=.3, which='both'); ax.legend()
    save_figure(fig, fig_dir/'01_known_task_xy_error.png')
    generated.append('figures/01_known_task_xy_error.png')

    future = grouped_stats(detail, 'future_rmse')
    future_global = sorted((r for r in future if r['region'] == 'GLOBAL'), key=lambda r:r['fraction'])
    save_csv(table_dir/'05_known_task_future_rmse.csv', future_global)
    generated.append('tables/05_known_task_future_rmse.csv')
    if future_global:
        fig, ax = plt.subplots(figsize=(7.5,5))
        ax.plot([100*r['fraction'] for r in future_global],
                [r['mean'] for r in future_global], marker='o')
        ax.set(xlabel='Observed trajectory (%)', ylabel='Future joint RMSE',
               title='Known task: unobserved trajectory RMSE (L-BFGS)')
        ax.grid(alpha=.3)
        save_figure(fig, fig_dir/'02_future_rmse_vs_observation.png')
        generated.append('figures/02_future_rmse_vs_observation.png')

    counts = [{'region': region, 'n_unique_trajectories':len(set(r['trajectory'] for r in detail
             if r['region']==region))} for region in ('interpolation','extrapolation')]
    save_csv(table_dir/'00_test_region_counts.csv', counts)
    generated.append('tables/00_test_region_counts.csv')


def plot_paired(detail, fig_dir, table_dir, generated):
    required = {'method_label','fraction','noise_sigma','region','xy_error_mm','task_correct'}
    if not required.issubset(detail[0]):
        raise ValueError(f'Paired detail missing columns: {required - set(detail[0])}')
    methods = {r['method_label'] for r in detail}
    if not {METHOD_A, METHOD_B}.issubset(methods):
        raise ValueError(f'Paired data does not contain both expected L-BFGS method labels: {methods}')
    if 'optimizer' in detail[0] and any(r['optimizer'].lower() != 'lbfgs' for r in detail):
        raise ValueError('Paired input contains non-LBFGS optimizer rows')
    # Pure noise-free conditions, directly comparable A and B.
    pure = [r for r in detail if abs(number(r, 'noise_sigma')) < 1e-12]
    comparisons = []
    for fraction in unique_fractions(pure):
        for label in (METHOD_A, METHOD_B):
            subset = [r for r in pure if r['method_label']==label
                      and abs(number(r,'fraction')-fraction) < FRACTION_TOL]
            if not subset:
                continue
            comparisons.append({'fraction':fraction, 'method':label, 'n':len(subset),
                'task_accuracy':float(np.mean([number(r,'task_correct') for r in subset])),
                'xy_mean_mm':float(np.mean([number(r,'xy_error_mm') for r in subset])),
                'xy_median_mm':float(np.median([number(r,'xy_error_mm') for r in subset]))})
    save_csv(table_dir/'03_unknown_task_method_comparison.csv', comparisons)
    generated.append('tables/03_unknown_task_method_comparison.csv')
    for attr, filename, ylabel, title in (
        ('xy_mean_mm','03_methodA_vs_methodB_xy.png','Mean XY error (mm)',
         'Unknown task: XY recovery, sigma=0'),
        ('task_accuracy','04_methodA_vs_methodB_task_accuracy.png','Task accuracy',
         'Unknown task: classification, sigma=0'),
    ):
        fig, ax = plt.subplots(figsize=(8,5.2))
        for label, name in ((METHOD_A,'Method A: discrete, L-BFGS 60'),
                            (METHOD_B,'Method B: softmax, L-BFGS 120')):
            ss = sorted((r for r in comparisons if r['method']==label), key=lambda r:r['fraction'])
            ax.plot([100*r['fraction'] for r in ss], [r[attr] for r in ss],marker='o',label=name)
        ax.set(xlabel='Observed trajectory (%)', ylabel=ylabel, title=title)
        if attr=='task_accuracy': ax.set_ylim(0,1.04)
        ax.grid(alpha=.3);ax.legend()
        save_figure(fig,fig_dir/filename)
        generated.append('figures/'+filename)

    # Noise robustness for the SELECTED unknown-task method A; no Adam.
    noise = []
    method_a = [r for r in detail if r['method_label']==METHOD_A]
    for sigma in sorted(set(number(r,'noise_sigma') for r in method_a)):
        for fraction in unique_fractions(method_a):
            rr=[r for r in method_a if abs(number(r,'noise_sigma')-sigma)<1e-12
                and abs(number(r,'fraction')-fraction)<FRACTION_TOL]
            if rr:
                noise.append({'noise_sigma':sigma,'fraction':fraction,'n':len(rr),
                              'xy_mean_mm':float(np.mean([number(r,'xy_error_mm') for r in rr]))})
    save_csv(table_dir/'06_selected_method_noise.csv',noise)
    generated.append('tables/06_selected_method_noise.csv')
    if len(set(r['noise_sigma'] for r in noise)) > 1:
        fig, ax = plt.subplots(figsize=(8.5,5.3))
        for sigma in sorted(set(r['noise_sigma'] for r in noise)):
            ss=sorted((r for r in noise if r['noise_sigma']==sigma),key=lambda r:r['fraction'])
            ax.plot([100*r['fraction'] for r in ss], [r['xy_mean_mm'] for r in ss],
                    marker='o',label=f'sigma={sigma:g}')
        ax.set(xlabel='Observed trajectory (%)',ylabel='Mean XY error (mm)',
               title='Noise robustness: selected Method A + L-BFGS 60')
        ax.legend();ax.grid(alpha=.3)
        save_figure(fig,fig_dir/'06_noise_robustness.png')
        generated.append('figures/06_noise_robustness.png')


def plot_spatial_split(fig_dir, generated, sources, missing):
    """Figure 00: exact 20 x 20 cm nominal train/validation support."""
    try:
        from CESN.core import parametersCESN as p
        from CESN.core.data import load_split
        from matplotlib.patches import Rectangle

        # Configured dataset-generator domain: NOT the bounding box of sampled points.
        X0, X1, Y0, Y1 = -0.15, 0.05, -0.10, 0.10
        loaded = [(str(s), load_split(s, p.DATA_PATH, joint_dim=p.JOINT_DIM))
                  for s in p.TRAIN_SPLITS]
        test = load_split(p.TEST_SPLIT, p.DATA_PATH, joint_dim=p.JOINT_DIM)

        def xy(row):
            c = np.asarray(row['cube_xyz'], dtype=float)
            return (c[0] if c.ndim == 2 else c)[:2]

        train, val = [], []
        for name, rows in loaded:
            (val if 'val' in name.lower() else train).extend(rows)
        interp = [r for r in test if r.get('context_region') == 'interpolation']
        extrap = [r for r in test if r.get('context_region') == 'extrapolation']
        groups = [
            ('Train', train, '#2878B5', 'o', 27),
            ('Validation', val, '#F28E2B', 's', 38),
            ('Test interpolation', interp, '#309E40', '^', 43),
            ('Test extrapolation', extrap, '#D62728', 'x', 49),
        ]
        if not train:
            raise ValueError('Training split is empty')

        fig, ax = plt.subplots(figsize=(9.2, 8.7))
        handles = []
        for label, rows, color, marker, size in groups:
            if not rows:
                continue
            pts = np.stack([xy(r) for r in rows])
            h = ax.scatter(pts[:, 0], pts[:, 1], label=f'{label} (n={len(rows)})',
                           marker=marker, s=size, color=color, alpha=0.92,
                           linewidths=1.6 if marker == 'x' else 0.6,
                           zorder=3)
            handles.append(h)

        # This rectangle depicts the exact generator-domain boundaries.
        box = Rectangle((X0, Y0), X1-X0, Y1-Y0, fill=False,
                        edgecolor='#222222', linewidth=1.8, linestyle='--',
                        zorder=4, label='Training support boundary')
        ax.add_patch(box)
        handles.append(box)

        ax.set(xlim=(-0.215, 0.115), ylim=(-0.165, 0.165),
               xlabel='Cube X position (m)', ylabel='Cube Y position (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, color='#DDE0E4', linewidth=.7, alpha=.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
        ax.set_title('Spatial context protocol', fontsize=16, pad=14)

        # Legend outside plot: no overlap with observations or dimension labels.
        fig.legend(handles=handles, labels=[h.get_label() for h in handles],
                   loc='lower center', bbox_to_anchor=(.5, .03),
                   ncol=3, fontsize=9.7, frameon=False, columnspacing=1.6)
        fig.subplots_adjust(left=.10, right=.96, top=.93, bottom=.19)

        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / '00_spatial_context_split.png', dpi=200,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        generated.append('figures/00_spatial_context_split.png')
        sources['spatial_split'] = {
            'dataset': str(p.DATA_PATH),
            'train_split_names': [name for name, _ in loaded],
            'test_split_name': str(p.TEST_SPLIT),
            'train_n': len(train), 'validation_n': len(val),
            'test_interpolation_n': len(interp),
            'test_extrapolation_n': len(extrap),
            'rectangle_definition': 'configured inner XY domain from dataset generator',
            'rectangle_xy_bounds_m': [[X0, Y0], [X1, Y1]],
            'nominal_support_size_cm': [20, 20],
        }
    except (FileNotFoundError, ImportError, KeyError, ValueError) as exc:
        missing.append(f'Figure 00 spatial context: {exc}')


def plot_remaining(fig_dir,table_dir,generated,sources,missing):
    """Consume only final L-BFGS results; never call legacy Adam in reporting."""
    base=ROOT/'CESN/outputs/final_lbfgs'
    emergence=base/'task_emergence/task_emergence_summary.csv'
    if emergence.is_file():
        rr=csv_rows(emergence)
        fig,ax=plt.subplots(figsize=(8,5))
        for region in ('interpolation','extrapolation','GLOBAL'):
            ss=sorted([r for r in rr if r['region']==region],key=lambda r:int(r['K_requested']))
            if ss:ax.plot([int(r['K_requested']) for r in ss],
                          [float(r['task_accuracy']) for r in ss],marker='o',label=region)
        ax.set(xlabel='Exact observed frames K',ylabel='Task accuracy',
               title='Task emergence: Method A + L-BFGS 60',ylim=(0,1.05))
        ax.legend();ax.grid(alpha=.3);save_figure(fig,fig_dir/'05_task_emergence.png')
        generated.append('figures/05_task_emergence.png');sources['task_emergence']=str(emergence)
    else:missing.append(f'Exact-K task-emergence not yet generated: {emergence}')

    baseline=base/'baselines/baselines_summary.csv'
    if baseline.is_file():
        rr=csv_rows(baseline)
        fig,ax=plt.subplots(figsize=(8.5,5))
        for method in sorted(set(r['method'] for r in rr)):
            ss=sorted((r for r in rr if r['method']==method and r['region']=='GLOBAL'),
                      key=lambda r:float(r['fraction']))
            if ss:ax.plot([100*float(r['fraction']) for r in ss],
                          [float(r['xy_median_mm']) for r in ss],marker='o',label=method)
        ax.set(xlabel='Observed trajectory (%)',ylabel='Median XY error (mm)',
               title='Selected L-BFGS Method A vs. non-optimizer baselines')
        ax.legend();ax.grid(alpha=.3);save_figure(fig,fig_dir/'07_baselines_vs_inversion.png')
        generated.append('figures/07_baselines_vs_inversion.png');sources['baselines']=str(baseline)
    else:missing.append(f'Baselines not yet generated: {baseline}')

    ambiguity=base/'ambiguity/ambiguity_detail.csv'
    if ambiguity.is_file():
        rr=csv_rows(ambiguity)
        fig,ax=plt.subplots(figsize=(8,5))
        for region in sorted(set(r['region'] for r in rr)):
            ss=[r for r in rr if r['region']==region]
            ax.scatter([float(r['xy_error_mm']) for r in ss],
                       [float(r['eig_min']) for r in ss],label=region,alpha=.8)
        ax.axhline(0,color='gray',lw=1)
        ax.set(xlabel='Recovered XY error (mm)',ylabel='Smallest Hessian eigenvalue',
               title='Known-task local Hessian curvature at L-BFGS solutions')
        ax.legend();ax.grid(alpha=.3);save_figure(fig,fig_dir/'08_hessian_confidence.png')
        generated.append('figures/08_hessian_confidence.png');sources['ambiguity']=str(ambiguity)
    else:missing.append(f'Ambiguity/Hessian not yet generated: {ambiguity}')

    completion_dir=base/'movement_completion/figures'
    fk_dir=base/'movement_completion_with_fk/figures'
    for family in ('09_joint_completion_','10_task_space_'):
        search_dirs=([completion_dir,fk_dir] if family.startswith('10_') else [completion_dir])
        files=[]
        for source_dir in search_dirs:
            if source_dir.is_dir():files.extend(source_dir.glob(family+'*.png'))
        files=sorted(set(files))
        for src in files:
            dest=fig_dir/src.name;shutil.copy2(src,dest)
            generated.append('figures/'+src.name)
        if files:sources[family]=[str(p) for p in files]
        else:missing.append(f'No {family} figures yet in {completion_dir}')

    # Loss landscape already generated by its own optimizer trace script; do not
    # regenerate its optimization here or copy a stale Adam version.
    traces=sorted(fig_dir.glob('11_loss_landscape_lbfgs_runs_K*.png'))
    if traces:
        sources['loss_landscape']=[str(p) for p in traces]
        generated.extend('figures/'+p.name for p in traces)
    else:missing.append('Missing Figure 11 L-BFGS output: run final.loss_landscape --verify')


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--known-detail',type=Path,default=DEFAULT_KNOWN)
    ap.add_argument('--paired-detail',type=Path,default=None)
    ap.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--strict',action='store_true',help='Require both known and paired data')
    return ap.parse_args()


def main():
    args=parse_args()
    out=args.out_dir.resolve()
    figure_dir=out/'figures';table_dir=out/'tables'
    figure_dir.mkdir(parents=True,exist_ok=True);table_dir.mkdir(parents=True,exist_ok=True)
    paired=args.paired_detail or next((p for p in PAIRED_SOURCES if p.is_file()),None)
    sources={};generated=[];missing=[]
    if args.known_detail.is_file():
        details=csv_rows(args.known_detail)
        plot_known(details,figure_dir,table_dir,generated)
        sources['known_task']={'path':str(args.known_detail.resolve()),'rows':len(details)}
    else:
        missing.append(f'Known-task CSV missing: {args.known_detail}')
    if paired is not None and paired.is_file():
        details=csv_rows(paired)
        plot_paired(details,figure_dir,table_dir,generated)
        sources['paired_methods']={'path':str(paired.resolve()),'rows':len(details)}
    else:
        missing.append('Paired A/B CSV missing: expected '
                       'or CESN/outputs/reports/paired_method_selection/detail.csv')

    plot_spatial_split(figure_dir,generated,sources,missing)
    plot_remaining(figure_dir,table_dir,generated,sources,missing)

    pending={}
    if not (figure_dir/'05_task_emergence.png').is_file(): pending['05_task_emergence.png']='Run final.task_emergence'
    if not (figure_dir/'07_baselines_vs_inversion.png').is_file(): pending['07_baselines_vs_inversion.png']='Run final.baselines'
    if not (figure_dir/'08_hessian_confidence.png').is_file(): pending['08_hessian_confidence.png']='Run final.ambiguity'
    if not list(figure_dir.glob('09_joint_completion_*.png')): pending['09_joint_completion_*.png']='Run final.movement_completion'
    if not list(figure_dir.glob('10_task_space_*.png')): pending['10_task_space_*.png']='Run final.movement_completion --task-space'
    if not list(figure_dir.glob('11_loss_landscape_lbfgs_runs_K*.png')): pending['11_loss_landscape_lbfgs.png']='Run final.loss_landscape'
    try:
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError,OSError): sha='unknown'
    manifest={'generated_utc':datetime.now(timezone.utc).isoformat(),
              'git_commit':sha, 'selected_method':METHOD_A,
              'model_retraining':False,'source_files':sources,
              'generated':generated,'missing_inputs':missing,'pending_figures':pending,
              'limitations':['Paired method hyperparameters previously tuned using TEST.',
                             'B has different iteration budget (120 vs A 60).',
                             'K=T excluded from future RMSE. No independent holdout.']}
    (out/'report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('\nL-BFGS REPORT |',out)
    for item in generated:print('  generated:',item)
    for item in missing:print('  missing:',item)
    print('  pending:',len(pending),'figure groups (see report_manifest.json)')
    if args.strict and missing:raise SystemExit('ERROR: --strict requires known-task and paired data')
    if not generated:raise SystemExit('ERROR: no supported CSV sources; report not generated')

if __name__=='__main__':main()

"""Shared experiment utilities for the frozen CESN and selected Method A + L-BFGS.

Only the observed q[:K] is supplied to the optimizer. Ground truth and future
frames are used for evaluation afterward. Batch curvature is SHARED across all
candidates: keep batch_cases, restarts, ordering and seeds fixed to reproduce.
"""
from __future__ import annotations
import csv
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from CESN.core import cesn, parametersCESN as p
from CESN.core.data import load_split
from CESN.rollout.gpu_batch_inversion import GPUBatchedCESN
from CESN.inversion.infer_context import infer_context, sample_xy_starts

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT/'CESN/outputs/models/N_900/run_01/model.npz'
OUTPUT = ROOT/'CESN/outputs/final_lbfgs'


def xy_of(row):
    a = np.asarray(row['cube_xyz'], dtype=np.float64)
    return (a[0] if a.ndim == 2 else a)[:2].copy()


def load_environment(device='cuda', dtype='float32', model_path=MODEL):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f'Frozen model missing: {model_path}')
    if not Path(p.DATA_PATH).is_file():
        raise FileNotFoundError(f'Dataset missing: {p.DATA_PATH}')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu')
    model = cesn.load_model(model_path)
    runner = GPUBatchedCESN(model, p.LEAK_RATE, device=device,
                            dtype=torch.float32 if dtype == 'float32' else torch.float64)
    test = load_split(p.TEST_SPLIT, p.DATA_PATH, joint_dim=p.JOINT_DIM)
    training = []
    for split in p.TRAIN_SPLITS:
        training.extend(load_split(split, p.DATA_PATH, joint_dim=p.JOINT_DIM))
    if not training:
        raise RuntimeError('No TRAIN cases to define initialization range')
    xy = np.stack([xy_of(r) for r in training]); low, high = xy.min(0), xy.max(0)
    bounds = (float(low[0]), float(high[0]), float(low[1]), float(high[1]))
    return model, runner, test, training, bounds


def selected_rows(rows, max_cases=None):
    if max_cases is None:
        return list(rows)
    from CESN.analysis.comparisons.benchmark_optimizers_full import select_balanced
    return select_balanced(rows, max_cases)


def K_from_fraction(T, fraction):
    return max(2, min(int(T), int(round(float(fraction) * int(T)))))


def save_csv(path, records):
    if not records:
        raise ValueError(f'No records for {path}')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in records for k in r.keys()))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)


def output_dir(path, overwrite=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f'Output not empty: {path}; use --overwrite or choose a fresh --out-dir')
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_sha256(path):
    path=Path(path)
    if not path.is_file():return None
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            sha.update(block)
    return sha.hexdigest()


def metadata(**kwargs):
    try:
        git = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True,
                                      stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): git = 'unavailable'
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), git_commit=git,
                frozen_model_path=str(MODEL), model_sha256=file_sha256(MODEL),
                dataset_path=str(p.DATA_PATH), dataset_sha256=file_sha256(p.DATA_PATH),
                torch_version=torch.__version__, cuda_version=torch.version.cuda,
                model_frozen=True, optimizer='lbfgs', max_iter=60, lr=0.8,
                optimization='shared_curvature_batched',
                limitations='TEST used for earlier tuning; no independent holdout. '
                            'Batched L-BFGS depends on candidate grouping and order.', **kwargs)


def save_metadata(out, **kwargs):
    (Path(out)/'config.json').write_text(json.dumps(metadata(**kwargs), indent=2, default=str)+'\n',
                                        encoding='utf-8')


def invert_group(runner, group, Ks, bounds, seed, restarts=4, mode='discrete',
                 noise=None, max_iter=60, lr=.8):
    """Returns (InversionResult list, batch stats); noise is list [K,9] already sampled."""
    Ts = [len(r['joints']) for r in group]
    if len(Ks) != len(Ts): raise ValueError('K length mismatch')
    prefixes = [np.asarray(r['joints'], np.float64)[:int(k)].copy()
                for r,k in zip(group, Ks)] if noise is None else noise
    starts = sample_xy_starts(len(group), restarts, bounds, seed)
    tasks = [str(r['kind']).lower() for r in group] if mode == 'known' else None
    result = infer_context(runner, prefixes, Ts, starts, method=mode,
                           known_tasks=tasks, optimizer='lbfgs',
                           lbfgs_max_iter=max_iter, lbfgs_lr=lr)
    return result, result[0].batch_stats


def score(group, Ks, inversions, model, fraction=None, noise_sigma=0):
    output=[]
    for row,K,r in zip(group,Ks,inversions):
        truth=np.asarray(row['joints'],np.float64); T=len(truth)
        ctx=cesn.make_context(r.predicted_task, r.estimated_xy)
        pred=np.asarray(cesn.predict_trajectory(model,ctx,T),np.float64)
        future=float(np.sqrt(np.mean((pred[int(K):]-truth[int(K):])**2))) if int(K)<T else float('nan')
        true_xy=xy_of(row)
        output.append(dict(trajectory=str(row['id']), region=str(row.get('context_region','unknown')),
               true_task=str(row['kind']).lower(), predicted_task=r.predicted_task,
               task_correct=int(str(row['kind']).lower()==r.predicted_task),
               fraction=(float(fraction) if fraction is not None else float(K)/T),
               noise_sigma=float(noise_sigma), T=T, K=int(K),
               true_x=float(true_xy[0]),true_y=float(true_xy[1]),
               estimated_x=float(r.estimated_xy[0]),estimated_y=float(r.estimated_xy[1]),
               xy_error_mm=float(1000*np.linalg.norm(r.estimated_xy-true_xy)),
               prefix_loss=float(r.prefix_loss),future_rmse=future,
               selected_restart=int(r.selected_restart),optimizer='lbfgs',method='A_discrete' ))
    return output


def summary(detail, *, dimensions=('fraction','region')):
    groups={}
    for r in detail:
        for region in ('GLOBAL',r['region']):
            key=tuple(region if d=='region' else r[d] for d in dimensions)
            groups.setdefault(key,[]).append(r)
    records=[]
    for key,rows in sorted(groups.items()):
        xy=np.array([r['xy_error_mm'] for r in rows],float)
        fut=np.array([r['future_rmse'] for r in rows],float)
        rr=dict(zip(dimensions,key));rr.update(n=len(rows),task_accuracy=float(np.mean([r['task_correct'] for r in rows])),
              xy_mean_mm=float(xy.mean()),xy_median_mm=float(np.median(xy)),
              xy_q25_mm=float(np.quantile(xy,.25)),xy_q75_mm=float(np.quantile(xy,.75)),
              future_rmse_mean=float(np.mean(fut[np.isfinite(fut)])) if np.isfinite(fut).any() else float('nan'))
        records.append(rr)
    return records

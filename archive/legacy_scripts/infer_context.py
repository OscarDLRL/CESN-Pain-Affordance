#!/usr/bin/env python3
"""Selected CESN context inversion: discrete task hypotheses + batched L-BFGS.

NO RETRAINING. The caller supplies a frozen GPUBatchedCESN runner.
Only observed joint prefixes are accepted; nothing here reads future frames or
true task/XY. Exactly follows the candidate ordering and summed batched loss
in CESN.analysis.comparisons.benchmark_optimizers_full.prepare/run_method.

By default Method A optimizes four common XY starts separately under Pick and
Push: [start0/Pick, start0/Push, start1/Pick, start1/Push, ...].
L-BFGS uses ONE curvature history for all candidates in this batch.
Consequently outputs can depend on batching, candidate order and dtype.

The selected (on prior TEST experiments, not a new independent holdout)
configuration is Method A with L-BFGS max_iter=60, lr=0.8, history_size=20,
strong_wolfe; stopping tolerances are supplied by optimizer_backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from CESN.core import parametersCESN as p
from CESN.inversion.optimizer_backend import OptimizationStats, optimize

TASKS = ("pick", "push")


@dataclass
class InversionResult:
    predicted_task: str
    estimated_xy: np.ndarray
    prefix_loss: float
    selected_restart: int
    selected_hypothesis: str
    T: int
    K: int
    # Timing is for the entire batch, NOT this individual trajectory.
    batch_stats: OptimizationStats


def _as_prefixes(observed_prefixes: Sequence[np.ndarray]) -> list[np.ndarray]:
    prefixes = []
    for i, obs in enumerate(observed_prefixes):
        arr = np.asarray(obs, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != int(p.JOINT_DIM):
            raise ValueError(f"Observation {i} must have shape [K,{p.JOINT_DIM}], got {arr.shape}")
        if not 1 <= len(arr) or not np.isfinite(arr).all():
            raise ValueError(f"Observation {i} is empty or nonfinite")
        prefixes.append(arr)
    if not prefixes:
        raise ValueError("At least one observation prefix is required")
    return prefixes


def sample_xy_starts(n_cases: int, restarts: int, bounds, seed: int) -> np.ndarray:
    """Return [cases,restarts,2] initial XY; bounds derived from TRAIN only.

    `bounds` must be (xmin,xmax,ymin,ymax). This samples initial points only;
    optimization remains unbounded for spatial extrapolation as in the study.
    """
    if restarts < 1 or n_cases < 1:
        raise ValueError("n_cases and restarts must be positive")
    xmin, xmax, ymin, ymax = map(float, bounds)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid initialization bounds")
    rng = np.random.default_rng(seed)
    starts = np.empty((n_cases, restarts, 2), dtype=np.float64)
    starts[:, :, 0] = rng.uniform(xmin, xmax, size=(n_cases, restarts))
    starts[:, :, 1] = rng.uniform(ymin, ymax, size=(n_cases, restarts))
    return starts


def infer_context(
    runner,
    observed_prefixes: Sequence[np.ndarray],
    T_each: Sequence[int],
    xy_starts: np.ndarray,
    *,
    method: str = "discrete",
    optimizer: str = "lbfgs",
    known_tasks: Sequence[str] | None = None,
    lbfgs_max_iter: int = 60,
    lbfgs_lr: float = 0.8,
    adam_iters: int = 150,
    adam_lr: float = 0.02,
    adam_patience: int | None = 20,
) -> list[InversionResult]:
    """Invert multiple prefixes in ONE batched optimizer call.

    Parameters
    ----------
    observed_prefixes: list of arrays [K_i,9]. Only visible frames are passed.
    T_each: full episode length per case (metadata for phase t/(T-1)).
    xy_starts: [C,R,2] starts; may be prepared from TRAIN XY bounds.
    method: 'discrete' (selected Method A), or 'known' (task provided).
    optimizer: 'lbfgs' selected, or 'adam' for reference only.
    known_tasks: required for method='known', must NOT be used for discrete.
    """
    prefixes = _as_prefixes(observed_prefixes)
    C = len(prefixes)
    Ts = np.asarray(T_each, dtype=np.int64)
    if Ts.shape != (C,):
        raise ValueError(f"T_each must be [{C}]")
    Ks = np.asarray([len(x) for x in prefixes], dtype=np.int64)
    if np.any(Ts < Ks) or np.any(Ts < 1):
        raise ValueError("Full episode T must be >= observed K >= 1")
    starts = np.asarray(xy_starts, dtype=np.float64)
    if starts.ndim != 3 or starts.shape[0] != C or starts.shape[2] != 2 or starts.shape[1] < 1:
        raise ValueError(f"xy_starts must be [{C},R,2]")
    if not np.isfinite(starts).all():
        raise ValueError("Nonfinite XY starts")
    if method not in ("discrete", "known"):
        raise ValueError("Supported methods: 'discrete', 'known'; softmax stays in comparisons")
    if optimizer not in ("lbfgs", "adam"):
        raise ValueError("Supported optimizers: 'lbfgs' and 'adam'")
    if method == "discrete" and known_tasks is not None:
        raise ValueError("Do not supply ground-truth task to discrete unknown-task inversion")
    if method == "known":
        if known_tasks is None or len(known_tasks) != C:
            raise ValueError("known_tasks must specify one task per case for method='known'")
        labels = [str(t).lower() for t in known_tasks]
        if any(t not in TASKS for t in labels):
            raise ValueError("known_tasks must be 'pick' or 'push'")
    R = starts.shape[1]
    if method == "discrete":
        # EXACT order used by previous benchmark: repeat each XY start twice
        # and alternate Pick, Push for each pair.
        xy_candidates = np.repeat(starts, 2, axis=1)
        hypotheses = np.array([[TASKS[j % 2] for j in range(2 * R)] for _ in range(C)])
    else:
        xy_candidates = starts
        hypotheses = np.array([[labels[i]] * R for i in range(C)])
    H = xy_candidates.shape[1]
    B = C * H
    xy = torch.tensor(xy_candidates.reshape(B, 2), device=runner.device,
                      dtype=runner.dtype, requires_grad=True)
    task_encoding = np.stack([np.asarray(p.TASK_LABELS[str(t)], np.float64)
                              for t in hypotheses.ravel()])
    task = torch.tensor(task_encoding, dtype=runner.dtype, device=runner.device)
    target_np = np.zeros((B, int(Ks.max()), int(p.JOINT_DIM)), dtype=np.float64)
    for i, prefix in enumerate(prefixes):
        target_np[i * H:(i + 1) * H, :len(prefix)] = prefix
    target = torch.tensor(target_np, dtype=runner.dtype, device=runner.device)
    t_gpu = torch.tensor(np.repeat(Ts, H), dtype=torch.long, device=runner.device)
    k_gpu = torch.tensor(np.repeat(Ks, H), dtype=torch.long, device=runner.device)

    def losses_per_candidate():
        context = torch.cat([task, xy], dim=1)
        return runner.prefix_loss(context, target, t_gpu, k_gpu)

    stats = optimize(
        [xy], losses_per_candidate, method=optimizer,
        lbfgs_iters=lbfgs_max_iter, lbfgs_lr=lbfgs_lr,
        adam_iters=adam_iters, adam_lr=adam_lr, adam_patience=adam_patience,
    )
    with torch.no_grad():
        final_losses = losses_per_candidate().detach().cpu().numpy().reshape(C, H)
        final_xy = xy.detach().cpu().numpy().reshape(C, H, 2)
    results = []
    for i in range(C):
        j = int(np.argmin(final_losses[i]))
        label = str(hypotheses[i, j])
        results.append(InversionResult(
            predicted_task=label,
            estimated_xy=final_xy[i, j].astype(np.float64),
            prefix_loss=float(final_losses[i, j]),
            selected_restart=j // 2 if method == "discrete" else j,
            selected_hypothesis=label,
            T=int(Ts[i]), K=int(Ks[i]), batch_stats=stats,
        ))
    return results


def infer_one(runner, y_obs: np.ndarray, T: int, xy_starts: np.ndarray, **kwargs) -> InversionResult:
    """Convenience wrapper; y_obs is ONLY the observed K-frame prefix.

    `xy_starts`: [R,2]. For benchmark parity use infer_context with the SAME
    grouping/batch as the benchmark; a single case has different L-BFGS history.
    """
    starts = np.asarray(xy_starts, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("xy_starts must be [R,2]")
    return infer_context(runner, [y_obs], [T], starts[None, :, :], **kwargs)[0]

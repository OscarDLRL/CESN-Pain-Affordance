#!/usr/bin/env python3
"""Shared optimizer implementations for frozen-CESN context inversion.

The loss function must return a *vector* of independent prefix MSE values,
with one entry per candidate initialization. All parameters are leaf tensors.
Neither optimizer ever receives the true context or the unobserved trajectory.

LBFGS batched uses ONE curvature history for the SUM of losses; it is not
mathematically equivalent to independent LBFGS per restart. Use independent
mode to examine that distinction.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import torch


@dataclass
class OptimizationStats:
    optimizer: str
    elapsed_s: float
    gradient_evaluations: int
    iterations: int
    initial_loss_mean: float
    final_loss_mean: float


def _sync(params: Sequence[torch.Tensor]) -> None:
    if params and params[0].device.type == "cuda":
        torch.cuda.synchronize(params[0].device)


def optimize(
    params: Sequence[torch.Tensor],
    per_candidate_loss: Callable[[], torch.Tensor],
    *,
    method: str,
    adam_lr: float = 0.02,
    adam_iters: int = 150,
    adam_patience: int | None = 20,
    adam_tol: float = 1e-9,
    lbfgs_lr: float = 0.8,
    lbfgs_iters: int = 60,
    lbfgs_history: int = 20,
) -> OptimizationStats:
    """Optimize all candidates jointly; return synchronized time and counts.

    For Adam, early stopping is on batch-mean loss, matching prior scripts.
    For L-BFGS, `gradient_evaluations` counts closure calls (line-search calls
    included). One batch evaluation can process many independent candidates.
    """
    params = list(params)
    if not params or not all(x.is_leaf and x.requires_grad for x in params):
        raise ValueError("Expected one or more trainable leaf tensors")
    method = method.lower()
    if method not in ("adam", "lbfgs"):
        raise ValueError(f"Unknown optimizer: {method}")

    with torch.no_grad():
        initial = float(per_candidate_loss().mean().item())
    grad_evals = 0
    iterations = 0
    _sync(params)
    t0 = perf_counter()

    if method == "adam":
        opt = torch.optim.Adam(params, lr=adam_lr)
        best, stale = float("inf"), 0
        for _ in range(adam_iters):
            opt.zero_grad(set_to_none=True)
            candidate_losses = per_candidate_loss()
            if not torch.isfinite(candidate_losses).all().item():
                raise FloatingPointError("Nonfinite Adam loss")
            current = float(candidate_losses.mean().detach().item())
            candidate_losses.sum().backward()
            grad_evals += 1
            opt.step()
            iterations += 1
            if best - current > adam_tol:
                best, stale = current, 0
            else:
                stale += 1
            if adam_patience is not None and stale >= adam_patience:
                break
    else:
        opt = torch.optim.LBFGS(
            params,
            lr=lbfgs_lr,
            max_iter=lbfgs_iters,
            history_size=lbfgs_history,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            nonlocal grad_evals
            opt.zero_grad(set_to_none=True)
            candidate_losses = per_candidate_loss()
            if not torch.isfinite(candidate_losses).all().item():
                raise FloatingPointError("Nonfinite L-BFGS loss")
            loss = candidate_losses.sum()
            loss.backward()
            grad_evals += 1
            return loss

        opt.step(closure)
        state = opt.state.get(params[0], {})
        iterations = int(state.get("n_iter", 0))

    _sync(params)
    elapsed = perf_counter() - t0
    with torch.no_grad():
        final = float(per_candidate_loss().mean().item())
    return OptimizationStats(method, elapsed, grad_evals, iterations, initial, final)

"""Reusable GPU-batched CESN forward/loss engine.

This module is intentionally generic.

It knows how to:
- run the frozen CESN on GPU
- handle cases x restarts
- use each case's own T for phase t/(T-1)
- compute prefix reconstruction loss for different K values

It does NOT know what the unknown context parameter means.
That logic belongs in gpu_delta_inversion.py.

Expected CESN context:
    [task_onehot, cube_xyz, delta]
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class InversionCase:
    case_id: str
    task_onehot: np.ndarray
    cube_xyz: np.ndarray
    y_obs: np.ndarray
    T: int
    K: int
    true_delta: float | None = None


class GPUBatchedCESN:
    def __init__(
        self,
        np_model,
        leak_rate,
        device="cuda",
        dtype=torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.alpha = float(leak_rate)

        self.W = torch.as_tensor(
            np_model["W"],
            dtype=dtype,
            device=self.device,
        )
        self.Win = torch.as_tensor(
            np_model["Win"],
            dtype=dtype,
            device=self.device,
        )
        self.Wout = torch.as_tensor(
            np_model["Wout"],
            dtype=dtype,
            device=self.device,
        )

        self.context_dim = int(np_model["context_dim"])
        self.joint_dim = int(np_model["joint_dim"])
        self.n_reservoir = int(self.W.shape[0])
        self.use_time_index = bool(np_model["use_time_index"])

        expected_input_dim = self.context_dim + (1 if self.use_time_index else 0)
        if self.Win.shape[1] != expected_input_dim + 1:
            raise ValueError(
                "Win dimension mismatch: "
                f"Win={tuple(self.Win.shape)}, "
                f"expected second dim={expected_input_dim + 1}"
            )

        expected_feature_dim = (
            1
            + self.n_reservoir
            + self.context_dim
            + (1 if self.use_time_index else 0)
        )
        if self.Wout.shape[0] != expected_feature_dim:
            raise ValueError(
                "Wout dimension mismatch: "
                f"Wout={tuple(self.Wout.shape)}, "
                f"expected first dim={expected_feature_dim}"
            )

        if self.Wout.shape[1] != self.joint_dim:
            raise ValueError(
                "Wout joint dimension mismatch: "
                f"Wout={tuple(self.Wout.shape)}, "
                f"joint_dim={self.joint_dim}"
            )

    def prefix_loss(
        self,
        context,
        target,
        T_each,
        K_each,
    ):
        """Return one mean-squared prefix loss per batch element."""
        if context.ndim != 2:
            raise ValueError(
                f"context must be [B,C], got {tuple(context.shape)}"
            )

        if context.shape[1] != self.context_dim:
            raise ValueError(
                "Context dimension mismatch: "
                f"got {context.shape[1]}, expected {self.context_dim}"
            )

        B = context.shape[0]
        maxK = target.shape[1]

        x = torch.zeros(
            B,
            self.n_reservoir,
            dtype=self.dtype,
            device=self.device,
        )

        sq_sum = torch.zeros(
            B,
            dtype=self.dtype,
            device=self.device,
        )

        count = (
            K_each.to(self.dtype) * float(self.joint_dim)
        ).clamp_min(1.0)

        ones = torch.ones(
            B,
            1,
            dtype=self.dtype,
            device=self.device,
        )

        denom = (
            T_each.to(self.dtype) - 1.0
        ).clamp_min(1.0)

        for t in range(maxK):
            input_parts = [context]
            phase = None

            if self.use_time_index:
                phase = (
                    torch.full(
                        (B, 1),
                        float(t),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    / denom[:, None]
                )
                input_parts.append(phase)

            u = torch.cat(input_parts, dim=1)

            z = (
                x @ self.W.T
                + self.Win[:, 0][None, :]
                + u @ self.Win[:, 1:].T
            )

            xn = torch.tanh(z)
            x = (
                (1.0 - self.alpha) * x
                + self.alpha * xn
            )

            feat_parts = [
                ones,
                x,
                context,
            ]

            if self.use_time_index:
                feat_parts.append(phase)

            feat = torch.cat(
                feat_parts,
                dim=1,
            )

            y = feat @ self.Wout

            active = (
                t < K_each
            ).to(self.dtype)

            err2 = (
                (y - target[:, t, :]) ** 2
            ).sum(dim=1)

            sq_sum = (
                sq_sum
                + active * err2
            )

        return sq_sum / count
from __future__ import annotations

import numpy as np
import torch

from CESN.core import parametersCESN as p


torch.set_default_dtype(
    torch.float64
)


class TorchCESN(
    torch.nn.Module
):

    def __init__(
        self,
        model,
    ):
        super().__init__()

        self.context_dim = int(
            model["context_dim"]
        )

        self.joint_dim = int(
            model["joint_dim"]
        )

        self.use_time_index = bool(
            model["use_time_index"]
        )

        self.register_buffer(
            "W",
            torch.as_tensor(
                np.asarray(
                    model["W"]
                ),
                dtype=torch.float64,
            ),
        )

        self.register_buffer(
            "Win",
            torch.as_tensor(
                np.asarray(
                    model["Win"]
                ),
                dtype=torch.float64,
            ),
        )

        self.register_buffer(
            "Wout",
            torch.as_tensor(
                np.asarray(
                    model["Wout"]
                ),
                dtype=torch.float64,
            ),
        )

        # --------------------------------------------------------
        # Sanity checks
        # --------------------------------------------------------

        expected_input_dim = (
            self.context_dim
            + (
                1
                if self.use_time_index
                else 0
            )
        )

        if (
            self.Win.shape[1]
            != expected_input_dim + 1
        ):
            raise ValueError(
                "Win dimension mismatch: "
                f"Win={tuple(self.Win.shape)}, "
                f"expected second dim="
                f"{expected_input_dim + 1}"
            )

        expected_feature_dim = (
            1
            + self.W.shape[0]
            + self.context_dim
            + (
                1
                if self.use_time_index
                else 0
            )
        )

        if (
            self.Wout.shape[0]
            != expected_feature_dim
        ):
            raise ValueError(
                "Wout dimension mismatch: "
                f"Wout={tuple(self.Wout.shape)}, "
                f"expected first dim="
                f"{expected_feature_dim}"
            )

        if (
            self.Wout.shape[1]
            != self.joint_dim
        ):
            raise ValueError(
                "Wout joint dimension mismatch: "
                f"Wout={tuple(self.Wout.shape)}, "
                f"joint_dim={self.joint_dim}"
            )

    def forward(
        self,
        context,
        T,
    ):

        # --------------------------------------------------------
        # Batch dimension
        # --------------------------------------------------------

        if context.ndim == 1:
            context = (
                context.unsqueeze(0)
            )

        if context.ndim != 2:
            raise ValueError(
                "context must have shape "
                "[context_dim] or "
                "[batch, context_dim]"
            )

        if (
            context.shape[1]
            != self.context_dim
        ):
            raise ValueError(
                "Context dimension mismatch: "
                f"got {context.shape[1]}, "
                f"expected {self.context_dim}"
            )

        B = context.shape[0]

        # --------------------------------------------------------
        # Initial reservoir state
        # --------------------------------------------------------

        x = torch.zeros(
            B,
            self.W.shape[0],
            dtype=context.dtype,
            device=context.device,
        )

        ys = []

        # --------------------------------------------------------
        # Unroll CESN
        # --------------------------------------------------------

        for t in range(
            int(T)
        ):

            input_parts = [
                context
            ]

            ti = None

            if self.use_time_index:

                ti = torch.full(
                    (
                        B,
                        1,
                    ),
                    float(t)
                    / max(
                        int(T) - 1,
                        1,
                    ),
                    dtype=context.dtype,
                    device=context.device,
                )

                input_parts.append(
                    ti
                )

            u = torch.cat(
                input_parts,
                dim=1,
            )

            # ----------------------------------------------------
            # Reservoir update
            # ----------------------------------------------------

            z = (
                x @ self.W.T
                + self.Win[
                    :,
                    0
                ].unsqueeze(0)
                + u
                @ self.Win[
                    :,
                    1:
                ].T
            )

            xn = torch.tanh(
                z
            )

            x = (
                (
                    1.0
                    - p.LEAK_RATE
                )
                * x
                + p.LEAK_RATE
                * xn
            )

            # ----------------------------------------------------
            # Readout feature
            # ----------------------------------------------------

            feat_parts = [
                torch.ones(
                    B,
                    1,
                    dtype=context.dtype,
                    device=context.device,
                ),

                x,

                context,
            ]

            if self.use_time_index:
                feat_parts.append(
                    ti
                )

            feat = torch.cat(
                feat_parts,
                dim=1,
            )

            y = (
                feat
                @ self.Wout
            )

            ys.append(
                y
            )

        return torch.stack(
            ys,
            dim=1,
        )
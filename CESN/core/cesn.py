from __future__ import annotations

import math
import torch


class ContextualESNTorch:
    """
    PyTorch CUDA implementation of forward RGB-only CESN.

    Input:
        [pain_onehot(2), z_static(128), normalized_time(1)]

    Output:
        z_dynamic(t)

    No joints, qpos, xyz, image feedback, or delta-z target.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        reservoir_size=3000,
        spectral_radius=0.95,
        leak_rate=0.25,
        input_scale=0.50,
        bias_scale=0.10,
        ridge=1e-2,
        seed=42,
        device=None,
        dtype=torch.float32,
        density=0.05,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.reservoir_size = int(reservoir_size)

        self.spectral_radius = float(spectral_radius)
        self.leak_rate = float(leak_rate)
        self.input_scale = float(input_scale)
        self.bias_scale = float(bias_scale)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.density = float(density)

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.dtype = dtype

        g = torch.Generator(
            device=self.device
        )
        g.manual_seed(
            self.seed
        )

        self.Win = (
            torch.empty(
                (
                    self.reservoir_size,
                    self.input_dim,
                ),
                device=self.device,
                dtype=self.dtype,
            )
            .uniform_(
                -1.0,
                1.0,
                generator=g,
            )
            * self.input_scale
        )

        self.bias = (
            torch.empty(
                (
                    self.reservoir_size,
                ),
                device=self.device,
                dtype=self.dtype,
            )
            .uniform_(
                -1.0,
                1.0,
                generator=g,
            )
            * self.bias_scale
        )

        W = torch.randn(
            (
                self.reservoir_size,
                self.reservoir_size,
            ),
            device=self.device,
            dtype=self.dtype,
            generator=g,
        )

        mask = (
            torch.rand(
                W.shape,
                device=self.device,
                generator=g,
            )
            < self.density
        )

        W = W * mask

        # Power iteration for spectral radius estimate.
        v = torch.randn(
            (
                self.reservoir_size,
            ),
            device=self.device,
            dtype=self.dtype,
            generator=g,
        )

        v = v / (
            torch.linalg.vector_norm(v)
            + 1e-8
        )

        for _ in range(80):
            v = W @ v

            v = v / (
                torch.linalg.vector_norm(v)
                + 1e-8
            )

        radius_est = torch.linalg.vector_norm(
            W @ v
        )

        if float(
            radius_est.item()
        ) < 1e-8:
            raise RuntimeError(
                "Reservoir spectral radius estimate is zero."
            )

        W = (
            W
            * (
                self.spectral_radius
                / radius_est
            )
        )

        self.W = W.contiguous()
        self.Wout = None

    @property
    def feature_dim(self):
        return (
            1
            + self.input_dim
            + self.reservoir_size
        )

    def reset_state(self):
        return torch.zeros(
            (
                self.reservoir_size,
            ),
            device=self.device,
            dtype=self.dtype,
        )

    @torch.no_grad()
    def collect_sequence(
        self,
        U,
    ):
        """
        U: [T, input_dim]

        Returns:
            X: [T, 1 + input_dim + reservoir_size]
        """

        U = torch.as_tensor(
            U,
            device=self.device,
            dtype=self.dtype,
        )

        T = int(
            U.shape[0]
        )

        X = torch.empty(
            (
                T,
                self.feature_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        state = self.reset_state()

        for t in range(T):
            u_t = U[t]

            pre = (
                self.Win @ u_t
                + self.W @ state
                + self.bias
            )

            candidate = torch.tanh(
                pre
            )

            state = (
                (
                    1.0
                    - self.leak_rate
                )
                * state
                +
                self.leak_rate
                * candidate
            )

            X[
                t,
                0
            ] = 1.0

            X[
                t,
                1:
                1 + self.input_dim
            ] = u_t

            X[
                t,
                1 + self.input_dim:
            ] = state

        return X

    @torch.no_grad()
    def fit(
        self,
        sequences,
        solve_dtype=torch.float64,
    ):
        """
        sequences:
            iterable of (U, Y)

        U:
            [T, input_dim]

        Y:
            [T, output_dim]

        Accumulates X^T X and X^T Y directly on GPU,
        then performs ridge solve on GPU.
        """

        XTX = torch.zeros(
            (
                self.feature_dim,
                self.feature_dim,
            ),
            device=self.device,
            dtype=solve_dtype,
        )

        XTY = torch.zeros(
            (
                self.feature_dim,
                self.output_dim,
            ),
            device=self.device,
            dtype=solve_dtype,
        )

        sample_count = 0

        for U, Y in sequences:
            X = self.collect_sequence(
                U
            ).to(
                dtype=solve_dtype
            )

            Y = torch.as_tensor(
                Y,
                device=self.device,
                dtype=solve_dtype,
            )

            if X.shape[0] != Y.shape[0]:
                raise ValueError(
                    "Input/output sequence length mismatch."
                )

            XTX.addmm_(
                X.T,
                X,
            )

            XTY.addmm_(
                X.T,
                Y,
            )

            sample_count += int(
                X.shape[0]
            )

        reg = torch.eye(
            self.feature_dim,
            device=self.device,
            dtype=solve_dtype,
        )

        reg *= self.ridge
        reg[0, 0] = 0.0

        A = XTX + reg

        self.Wout = torch.linalg.solve(
            A,
            XTY,
        ).to(
            dtype=self.dtype
        )

        return sample_count

    @torch.no_grad()
    def predict(
        self,
        U,
    ):
        if self.Wout is None:
            raise RuntimeError(
                "CESN has not been fitted."
            )

        X = self.collect_sequence(
            U
        )

        return X @ self.Wout

    def state_dict(self):
        return {
            "input_dim":
                self.input_dim,

            "output_dim":
                self.output_dim,

            "reservoir_size":
                self.reservoir_size,

            "spectral_radius":
                self.spectral_radius,

            "leak_rate":
                self.leak_rate,

            "input_scale":
                self.input_scale,

            "bias_scale":
                self.bias_scale,

            "ridge":
                self.ridge,

            "seed":
                self.seed,

            "density":
                self.density,

            "Win":
                self.Win.detach().cpu(),

            "W":
                self.W.detach().cpu(),

            "bias":
                self.bias.detach().cpu(),

            "Wout":
                (
                    None
                    if self.Wout is None
                    else self.Wout.detach().cpu()
                ),
        }

    @classmethod
    def from_state_dict(
        cls,
        state,
        device=None,
    ):
        model = cls(
            input_dim=
                int(
                    state[
                        "input_dim"
                    ]
                ),

            output_dim=
                int(
                    state[
                        "output_dim"
                    ]
                ),

            reservoir_size=
                int(
                    state[
                        "reservoir_size"
                    ]
                ),

            spectral_radius=
                float(
                    state[
                        "spectral_radius"
                    ]
                ),

            leak_rate=
                float(
                    state[
                        "leak_rate"
                    ]
                ),

            input_scale=
                float(
                    state[
                        "input_scale"
                    ]
                ),

            bias_scale=
                float(
                    state[
                        "bias_scale"
                    ]
                ),

            ridge=
                float(
                    state[
                        "ridge"
                    ]
                ),

            seed=
                int(
                    state[
                        "seed"
                    ]
                ),

            density=
                float(
                    state.get(
                        "density",
                        0.05,
                    )
                ),

            device=
                device,
        )

        model.Win = state[
            "Win"
        ].to(
            model.device
        )

        model.W = state[
            "W"
        ].to(
            model.device
        )

        model.bias = state[
            "bias"
        ].to(
            model.device
        )

        if state[
            "Wout"
        ] is not None:
            model.Wout = state[
                "Wout"
            ].to(
                model.device
            )

        return model

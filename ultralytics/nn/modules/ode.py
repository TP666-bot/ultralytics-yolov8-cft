# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""ODE-inspired robust feature evolution blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv


class ODERobustBlock(nn.Module):
    """Lightweight ODE-inspired residual feature evolution block.

    This block approximates continuous feature evolution with fixed-step Euler or RK2 updates:
        dx / dt = f_theta(x)

    Args:
        c1: Input channels.
        c2: Output channels. Must equal c1 for this residual ODE block.
        steps: Number of solver steps.
        step_size: Integration step size.
        solver: Fixed-step solver, either "euler" or "rk2".
        expansion: Hidden channel expansion ratio.
        gamma: Residual output scale.
    """

    def __init__(self, c1, c2=None, steps=2, step_size=0.5, solver="euler", expansion=0.5, gamma=1.0):
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 != c2:
            raise ValueError(f"ODERobustBlock requires c1 == c2, got c1={c1}, c2={c2}")
        if int(steps) < 1:
            raise ValueError(f"ODERobustBlock steps must be >= 1, got {steps}")
        solver = str(solver).lower()
        if solver not in {"euler", "rk2"}:
            raise ValueError(f"ODERobustBlock solver must be 'euler' or 'rk2', got {solver!r}")

        hidden = max(16, int(c1 * float(expansion)))
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.solver = solver
        self.gamma = float(gamma)
        self.ode_func = nn.Sequential(
            Conv(c1, hidden, 1, 1),
            Conv(hidden, hidden, 3, 1),
            Conv(hidden, c1, 1, 1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fixed-step ODE-inspired feature evolution."""
        z = x
        h = self.step_size
        for _ in range(self.steps):
            if self.solver == "rk2":
                k1 = self.ode_func(z)
                k2 = self.ode_func(z + h * k1)
                z = z + 0.5 * h * (k1 + k2)
            else:
                z = z + h * self.ode_func(z)
        return x + self.gamma * (z - x)

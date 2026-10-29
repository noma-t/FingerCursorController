"""One Euro Filter (Casiez et al. 2012), vectorised over landmark arrays."""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def __call__(self, x: np.ndarray, t_ms: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None or t_ms <= self._t:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t_ms
            return x.copy()
        dt = (t_ms - self._t) / 1000.0
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True) if x.ndim > 1 else np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        tau = 1.0 / (2.0 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1.0 - a) * self._x
        self._x, self._dx, self._t = x_hat, dx_hat, t_ms
        return x_hat.copy()

#!/usr/bin/env python3
#

import numpy as np


class Barrier:
    def __init__(self):
        # self.to_clamp = True
        self.clamp_lim = 1e18
        self.throw_error_on_OOD = False

    def __call__(self, x, der=False):
        if self.throw_error_on_OOD:
            assert self.is_safe(
                x
            ), f"OutOfDomainError for {x=} with norm {np.linalg.norm(x)}"

        if not der:
            if not self.is_safe(x):
                return self.clamp_lim * np.ones((1, 1))
            return np.clip(self.barrier(x), -self.clamp_lim, self.clamp_lim)
        else:
            if not self.is_safe(x):
                return self.clamp_lim * np.ones((self.state_dim, 1))
            return np.clip(self.grad_barrier(x), -self.clamp_lim, self.clamp_lim)


class NormBarrier(Barrier):
    def __init__(self, state_dim, upper_lim):
        super(NormBarrier, self).__init__()
        self.upper_lim = upper_lim
        self.state_dim = state_dim

    def is_safe(self, x):
        return self.upper_lim**2 - np.matmul(x.T, x) > 0

    def barrier(self, x):
        """Expect a 2D vector - (n*1) as x."""
        xtx = np.matmul(x.T, x)
        return np.power((self.upper_lim**2) / (self.upper_lim**2 - xtx) - 1, 2)

    def grad_barrier(self, x):
        """Expect a 2D vector - (n*1) as x."""
        xtx = np.matmul(x.T, x)
        ans = (2 / (self.upper_lim**2 - xtx)) * x
        return ans

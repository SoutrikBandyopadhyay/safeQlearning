#!/usr/bin/env python3
"""State space representation of plants."""
import numpy as np


class System(object):
    """Base class for systems."""

    def __init__(self):
        pass

    def __call__(self, x, u):
        return self.plant(x, u)

    def rg(self, x):
        raise NotImplementedError("Not Implemented R_g function")

    def dimensions(self):
        raise NotImplementedError("Not Implemented dimensions function")

    def drift_dynamics(self, x):
        raise NotImplementedError("Not Implemented drift_dynamics function")

    def g(self, x):
        raise NotImplementedError("Not Implemented g(x) function")


class MIMO(System):
    def __init__(self, A, B, R):
        super(MIMO, self).__init__()
        self.A = A
        self.B = B
        self.Rinv = np.linalg.inv(R)
        self.rg_precomputed = np.einsum("ij,jk,lk->il", self.B, self.Rinv, self.B)
        self.rinvg_precomputed = np.matmul(self.Rinv, self.B.T)

    def rg(self, x):
        return self.rg_precomputed

    def rinvg(self, x):
        return self.rinvg_precomputed

    def plant(self, x, u):
        return np.matmul(self.A, x) + np.matmul(self.B, u)

    def dimensions(self):
        return self.A.shape[0], self.B.shape[1]

    def drift_dynamics(self, x):
        return np.matmul(self.A, x)

    def g(self, x):
        return self.B

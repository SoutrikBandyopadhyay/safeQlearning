#!/usr/bin/env python3
#
import numpy as np


class QuadraticCost:
    def __init__(self, Q, R):
        self.Qmatrix = Q
        self.R = R

    def __call__(self, x, u):
        return 0.5 * np.einsum("ji,jk,kl->il", x, self.Qmatrix, x) + np.einsum(
            "ji,jk,kl->il", u, self.R, u
        )

    def Q(self, x):
        return 0.5 * np.einsum("ji,jk,kl->il", x, self.Qmatrix, x)

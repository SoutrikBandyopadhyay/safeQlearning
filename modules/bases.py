#!/usr/bin/env python3
import numpy as np
from numba import jit


class Basis:
    def __init__(self):
        pass

    def __call__(self, x, der=False):
        if not der:
            return self.basis(x)
        else:
            return self.grad_basis(x)

    def __len__(self):
        return self.length()


class QuadraticBasis(Basis):
    def __init__(self, state_dim):
        self.n = state_dim
        self.l = self.n * (self.n + 1) // 2

    def length(self):
        return self.l

    @jit(forceobj=True)
    def basis(self, x):
        """This is a function that implements the quadratic basis function in the
        article

        :x: A numpy 2D array which is assumed to be a column vector
        :returns: A numpy 2D column vector with quadratic basis stuff
        """
        ans = []
        A = np.matmul(x, x.T)
        for i in range(self.n):
            for j in range(i, self.n):
                ans.append(A[i, j])
        return np.array(ans).reshape((-1, 1))

    @jit(forceobj=True)
    def grad_basis(self, x):
        ans = np.zeros((self.l, self.n))
        index = 0

        for i in range(self.n):
            for j in range(i, self.n):
                if i == j:
                    ans[index, i] = 2 * x[i, 0]
                else:
                    ans[index, i] = x[j, 0]
                    ans[index, j] = x[i, 0]

                index += 1
        return ans

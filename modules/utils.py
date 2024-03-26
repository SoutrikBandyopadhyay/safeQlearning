##############################################################################
# This is a program that is the solution of the repeated problem             #
# of decomposing a state/numpy array into its constituent components         #
# Requires the capabilities of ordered dictionaries that was introduced in   #
# Python 3.6+                                                                #
##############################################################################


import numpy as np


class Decomposer(object):
    """
    This is a class that helps decompose a numpy 1d array into constituent 2d
    components
    """

    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.decomposition = {}
        self.composition = {}

    def __len__(self):
        ans = 0
        for i in self.dimensions.keys():
            dim = self.dimensions[i]
            ans += dim[0] * dim[1]
        return ans

    def computeDecomposition(self, state):
        self.clear()
        ptr = 0
        for i in self.dimensions.keys():
            dim = self.dimensions[i]
            elem = state[ptr : ptr + (dim[0] * dim[1])]
            self.decomposition[i] = elem.reshape((dim[0], dim[1]))
            ptr += dim[0] * dim[1]

    def get(self, item):
        assert (
            not len(self.decomposition) == 0
        ), "No state provided. Please use the computeDecomposition(state)"

        try:
            return self.decomposition[item]
        except KeyError:
            assert (
                False
            ), f"The item '{item}' is not in the predefined list of variables"

    def clear(self):
        self.decomposition = {}
        self.composition = {}

    def initCompositor(self):
        self.clear()
        for i in self.dimensions.keys():
            dim = self.dimensions[i]
            self.composition[i] = np.zeros(dim)

    def add(self, item, value):
        assert (
            not len(self.composition) == 0
        ), "Compositor not initialized. Call the method initCompositor"

        try:

            dim = self.dimensions[item]
            assert (
                value.shape[0] == dim[0] and value.shape[1] == dim[1]
            ), f"The value in {item} given must match the predefined dimensions {dim} but the dimensions given are {value.shape}"

            self.composition[item] = value

        except KeyError:
            assert (
                False
            ), f"The item '{item}' is not in the predefined list of variables"

    def combine(self):
        ans = np.array([])
        for i in self.dimensions:
            ans = np.concatenate([ans, self.composition[i].reshape(-1)])
        self.clear()
        return ans

    def decompose2D(self, sol):
        self.clear()
        ptr = 0
        for i in self.dimensions.keys():
            dim = self.dimensions[i]
            elem = sol[:, ptr : ptr + (dim[0] * dim[1])]
            self.decomposition[i] = elem
            ptr += dim[0] * dim[1]

"""
paths.py
========

Module containing utilities for finding paths through the design matrix.
"""

import abc

import networkx as nx


class IDesignPath(abc.ABC):

    def __init__(self, design_matrix):
        self.design_matrix = design_matrix

    @abc.abstractmethod
    def graph(self):
        pass

    @abc.abstractmethod
    def path(self):
        pass


class SquareRadiator(IDesignPath):

    def __init__(self, design_matrix):

        super().__init__(design_matrix)

        if len(design_matrix.columns) != 2:
            raise ValueError(f"more than two columns in design_matrix: {design_matrix.columns=}")

        self.nside = int(np.sqrt(len(self.design_matrix)))
        if not (len(self.design_matrix) % self.nside == 0):
            raise ValueError(f"{design_matrix=} cannot be square")

        k = np.arange(len(self.design_matrix), dtype=int)
        self.i = k // self.nside
        self.j = (k % self.nside)*(-1)**(self.i % 2) - self.nside**(self.i % 2) - 1  ### ???????

    def graph(self):
        pass

    def path(self):
        return self.i, self.j


import torch
from torch.nn.parameter import Parameter
import numpy as np

from sketch import AbstractSketch


class GaussianJLSketch(AbstractSketch):
    def __init__(self, in_dim, out_dim):
        super(GaussianJLSketch, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = self._gen_proj_matrix()

    def gen_sketch_funcs(self):
        return Parameter(torch.randn(self.out_dim, self.in_dim) / np.sqrt(self.out_dim), requires_grad=False)

    def sketch_mat(self, x):
        return self.p @ x

    def unsketch_mat(self, x):
        return self.p.T @ x


class CountSketchNaive(AbstractSketch):
    def __init__(self, in_dim, out_dim):
        super(CountSketchNaive, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h, self.s = self._gen_sketch_funcs()

    def gen_sketch_funcs(self):
        hash_matrix = np.zeros((self.out_dim, self.in_dim))
        rand_signs = (torch.randint(2, (self.in_dim, 1), dtype=torch.float32) * 2 - 1)
        hashed_indices = np.random.choice(self.out_dim, self.in_dim, replace=True)
        j = 0
        for v in hashed_indices:
            hash_matrix[v, j] = 1
            j += 1
        return Parameter(torch.from_numpy(hash_matrix).float(), requires_grad=False), \
               Parameter(rand_signs, requires_grad=False)

    def sketch_mat(self, x):
        return self.h @ (x * self.s)

    def unsketch_mat(self, x):
        return (self.h.T @ x) * self.s

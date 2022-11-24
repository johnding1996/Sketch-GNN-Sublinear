import numpy as np
import numba as nb
import torch


@nb.jit(nopython=True)
def _second_degree_tensor_sketch_vector(u, v, in_dim, out_dim, h1, h2, s1, s2):
    for i1 in range(in_dim):
        for i2 in range(in_dim):
            j = (h1[i1] + h2[i2]) % out_dim
            v[j] += s1[i1] * s2[i2] * u[i1] * u[i2]


@nb.jit(nopython=True)
def _third_degree_tensor_sketch_vector(u, v, in_dim, out_dim, h1, h2, h3, s1, s2, s3):
    for i1 in range(in_dim):
        for i2 in range(in_dim):
            for i3 in range(in_dim):
                j = (h1[i1] + h2[i2] + h3[i3]) % out_dim
                v[j] += s1[i1] * s2[i2] * s3[i3] * u[i1] * u[i2] * u[i3]


@nb.jit(nopython=True)
def _second_degree_tensor_sketch(x, in_dim, out_dim, h1, h2, s1, s2):
    # Note that we sketch the first dimension in the implementation
    z = np.zeros((out_dim, x.shape[1]))
    for i in range(x.shape[1]):
        _third_degree_tensor_sketch_vector(x[:, i], z[:, i], in_dim, out_dim, h1, h2, s1, s2)
    return z


@nb.jit(nopython=True)
def _third_degree_tensor_sketch(x, in_dim, out_dim, h1, h2, h3, s1, s2, s3):
    z = np.zeros((out_dim, x.shape[1]))
    for i in range(x.shape[1]):
        _third_degree_tensor_sketch_vector(x[:, i], z[:, i], in_dim, out_dim, h1, h2, h3, s1, s2, s3)
    return z


def count_sketch_to_numpy(count_sketch):
    return count_sketch.h.data.numpy(), count_sketch.s.data.numpy()


def second_degree_tensor_sketch(x, cs1, cs2):
    x = x.data.numpy()
    h1, s1 = count_sketch_to_numpy(cs1)
    h2, s2 = count_sketch_to_numpy(cs2)
    in_dim = cs1.in_dim
    out_dim = cs1.out_dim
    return torch.from_numpy(_second_degree_tensor_sketch(x, in_dim, out_dim, h1, h2, s1, s2)).float()


def third_degree_tensor_sketch(x, cs1, cs2, cs3):
    x = x.data.numpy()
    h1, s1 = count_sketch_to_numpy(cs1)
    h2, s2 = count_sketch_to_numpy(cs2)
    h3, s3 = count_sketch_to_numpy(cs3)
    in_dim = cs1.in_dim
    out_dim = cs1.out_dim
    return torch.from_numpy(_third_degree_tensor_sketch(x, in_dim, out_dim, h1, h2, h3, s1, s2, s3)).float()

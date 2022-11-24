import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch_sparse.tensor import SparseTensor
from torch_sparse.matmul import matmul as sparse_sparse_matmul
import numpy as np
from functools import reduce
from enum import Enum
from dataclasses import dataclass, astuple

from sketch_sparse import hash_dense_matmul, transpose_hash_dense_matmul
from sketch_non_fft import second_degree_tensor_sketch, third_degree_tensor_sketch


class AbstractSketch(nn.Module):
    def __init__(self):
        super(AbstractSketch, self).__init__()
        self.cache_dict = {}

    def gen_sketch_funcs(self):
        raise NotImplementedError

    def clear_cache(self):
        self.cache_dict = {}
        return self

    def sketch_mat(self, x):
        raise NotImplementedError

    def unsketch_mat(self, x):
        raise NotImplementedError

    def forward(self, x):
        if id(x) in self.cache_dict:
            return self.cache_dict[id(x)]
        self.cache_dict[id(x)] = self.sketch_mat(x)
        return self.cache_dict[id(x)]


class Backends(Enum):
    # CountSketch methods
    PYTORCH = 1
    PYTORCH_SPARSE = 2
    PYG_SPARSE = 3
    PYG_SCATTER = 4
    CYTHON_HASH = 5
    CUPY_HASH = 6
    # FFT methods
    PYTORCH_FFT = 11
    CUPY_FFT = 12
    CYTHON_SFFT = 13
    CUPY_SFFT = 14
    # TensorSketch methods
    PYTORCH_TENSORDOT = 21
    NUMBA_CPU = 22
    NUMBA_CUDA = 23


class OutputTypes(Enum):
    PYTORCH_TENSOR = 1
    PYTORCH_SPARSE = 2
    PYG_SPARSE = 3
    PYG_SCATTER = 4


@dataclass
class SketchConfig:
    dense_CPU: Backends
    dense_GPU: Backends
    sparse_CPU: Backends
    sparse_GPU: Backends
    output_type: OutputTypes


DEFAULT_COUNT_SKETCH_CONFIG = SketchConfig(
    dense_CPU=Backends.PYTORCH_SPARSE,
    dense_GPU=Backends.PYG_SPARSE,
    sparse_CPU=Backends.PYG_SPARSE,
    sparse_GPU=Backends.PYTORCH_SPARSE,
    output_type=OutputTypes.PYTORCH_TENSOR
)


class CountSketch(AbstractSketch):
    def __init__(self, in_dim, out_dim, config=DEFAULT_COUNT_SKETCH_CONFIG):
        super(CountSketch, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        assert isinstance(config, SketchConfig)
        self.config = config
        self.s = None
        self.h_pytorch = None
        self.h_pytorch_sparse, self.ht_pytorch_sparse = None, None
        self.h_pyg_sparse, self.ht_pyg_sparse = None, None
        self.h_pyg_scatter = None
        self.gen_sketch_funcs()

    def gen_sketch_funcs(self):
        rand_signs = (torch.randint(2, (self.in_dim, 1), dtype=torch.float32) * 2 - 1)
        self.s = rand_signs
        hashed_indices = np.random.choice(self.out_dim, self.in_dim, replace=True)
        hash_matrix = SparseTensor(row=torch.from_numpy(hashed_indices).type(torch.LongTensor),
                                   col=torch.arange(self.in_dim, dtype=torch.int64),
                                   sparse_sizes=(self.out_dim, self.in_dim),
                                   is_sorted=False)
        if Backends.PYTORCH in astuple(self.config):
            self.h_pytorch = hash_matrix.to_dense()
        if Backends.PYTORCH_SPARSE in astuple(self.config):
            self.h_pytorch_sparse = hash_matrix.to_torch_sparse_coo_tensor()
            self.ht_pytorch_sparse = self.h_pytorch_sparse.t()
        if Backends.PYG_SPARSE in astuple(self.config):
            self.h_pyg_sparse = hash_matrix
            self.ht_pyg_sparse = self.h_pyg_sparse.t()
        if Backends.PYG_SCATTER in astuple(self.config):
            self.h_pyg_scatter = torch.from_numpy(hashed_indices).type(torch.LongTensor)

    def _apply(self, fn):
        self.s = fn(self.s)
        if self.h_pytorch is not None:
            self.h_pytorch = fn(self.h_pytorch)
        if self.h_pytorch_sparse is not None:
            self.h_pytorch_sparse = fn(self.h_pytorch_sparse)
            self.ht_pytorch_sparse = fn(self.ht_pytorch_sparse)
        if self.h_pyg_sparse is not None:
            self.h_pyg_sparse = fn(self.h_pyg_sparse)
            self.ht_pyg_sparse = fn(self.ht_pyg_sparse)
        if self.h_pyg_scatter is not None:
            self.h_pyg_scatter = fn(self.h_pyg_scatter)
        super(CountSketch, self)._apply(fn)
        return self

    def sketch_mat(self, x):
        if isinstance(x, torch.Tensor):
            if x.device == torch.device('cpu'):
                backend = self.config.dense_CPU
            else:
                backend = self.config.dense_GPU

            if backend == Backends.PYTORCH:
                return self.h_pytorch @ (x * self.s)
            elif backend == Backends.PYTORCH_SPARSE:
                return self.h_pytorch_sparse @ (x * self.s)
            elif backend == Backends.PYG_SPARSE:
                return self.h_pyg_sparse @ (x * self.s)
            elif backend == Backends.PYG_SCATTER:
                return hash_dense_matmul(self.h_pyg_scatter, self.out_dim, x * self.s)
            else:
                raise NotImplementedError

        elif isinstance(x, SparseTensor):
            if x.device() == torch.device('cpu'):
                backend = self.config.sparse_CPU
            else:
                backend = self.config.sparse_GPU

            if backend == Backends.PYTORCH:
                raise NotImplementedError
            elif backend == Backends.PYTORCH_SPARSE:
                z = torch.sparse.mm(self.h_pytorch_sparse, (x * self.s).to_torch_sparse_coo_tensor()).coalesce()
                if self.config.output_type == OutputTypes.PYTORCH_TENSOR:
                    return z.to_dense()
                elif self.config.output_type == OutputTypes.PYTORCH_SPARSE:
                    return z
                elif self.config.output_type == OutputTypes.PYG_SPARSE:
                    return SparseTensor.from_torch_sparse_coo_tensor(z)
                else:
                    raise NotImplementedError
            elif backend == Backends.PYG_SPARSE:
                z = sparse_sparse_matmul(self.h_pyg_sparse, x * self.s).coalesce()
                if self.config.output_type == OutputTypes.PYTORCH_TENSOR:
                    return z.to_dense()
                elif self.config.output_type == OutputTypes.PYTORCH_SPARSE:
                    return z.to_torch_sparse_coo_tensor()
                elif self.config.output_type == OutputTypes.PYG_SPARSE:
                    return z
                else:
                    raise NotImplementedError
            elif backend == Backends.PYG_SCATTER:
                raise NotImplementedError
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def unsketch_mat(self, x):
        if isinstance(x, torch.Tensor):
            if x.device == torch.device('cpu'):
                backend = self.config.dense_CPU
            else:
                backend = self.config.dense_GPU

            if backend == Backends.PYTORCH:
                return (self.h_pytorch.T @ x) * self.s
            elif backend == Backends.PYTORCH_SPARSE:
                return (self.ht_pytorch_sparse @ x) * self.s
            elif backend == Backends.PYG_SPARSE:
                return (self.ht_pyg_sparse @ x) * self.s
            elif backend == Backends.PYG_SCATTER:

                return transpose_hash_dense_matmul(self.h_pyg_scatter, self.in_dim, x) * self.s
            else:
                raise NotImplementedError

        elif isinstance(x, SparseTensor):
            if x.device() == torch.device('cpu'):
                backend = self.config.sparse_CPU
            else:
                backend = self.config.sparse_GPU

            if backend == Backends.PYTORCH:
                raise NotImplementedError
            elif backend == Backends.PYTORCH_SPARSE:
                z = torch.sparse.mm(self.ht_pytorch_sparse, x.to_torch_sparse_coo_tensor()).coalesce() * self.s
                if self.config.output_type == OutputTypes.PYTORCH_TENSOR:
                    return z.to_dense()
                elif self.config.output_type == OutputTypes.PYTORCH_SPARSE:
                    return z
                elif self.config.output_type == OutputTypes.PYG_SPARSE:
                    return SparseTensor.from_torch_sparse_coo_tensor(z)
                else:
                    raise NotImplementedError
            elif backend == Backends.PYG_SPARSE:
                z = sparse_sparse_matmul(self.ht_pyg_sparse, x).coalesce() * self.s
                if self.config.output_type == OutputTypes.PYTORCH_TENSOR:
                    return z.to_dense()
                elif self.config.output_type == OutputTypes.PYTORCH_SPARSE:
                    return z.to_torch_sparse_coo_tensor()
                elif self.config.output_type == OutputTypes.PYG_SPARSE:
                    return z
                else:
                    raise NotImplementedError
            elif backend == Backends.PYG_SCATTER:
                raise NotImplementedError
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError


DEFAULT_TENSOR_SKETCH_CONFIG = SketchConfig(
        dense_CPU=Backends.PYTORCH_FFT,
        dense_GPU=Backends.PYTORCH_FFT,
        sparse_CPU=Backends.PYTORCH_FFT,
        sparse_GPU=Backends.PYTORCH_FFT,
        output_type=OutputTypes.PYTORCH_TENSOR
)


class TensorSketch(AbstractSketch):
    def __init__(self, count_sketch_list, config=DEFAULT_TENSOR_SKETCH_CONFIG):
        super(TensorSketch, self).__init__()
        self.config = config
        self.order = len(count_sketch_list)
        self.cs_list = count_sketch_list

    def gen_sketch_funcs(self):
        return None

    def sketch_mat(self, x):
        sketches = [self.cs_list[0].sketch_mat(x)]
        if self.order == 1:
            return sketches
        else:
            if isinstance(sketches[0], torch.Tensor):
                if x.device == torch.device('cpu'):
                    backend = self.config.dense_CPU
                else:
                    backend = self.config.dense_GPU

                if backend == Backends.PYTORCH_FFT:
                    sketches[0] = torch.fft.rfft(sketches[0], dim=0)
                    for degree in range(2, self.order+1):
                        sketches.append(torch.fft.rfft(self.cs_list[degree-1].sketch_mat(x), dim=0))
                        sketches[-1] = sketches[-1] * sketches[-2]
                    return list(map(lambda _: torch.fft.irfft(_, dim=0), sketches))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    def unsketch_mat(self, x):
        # The unsketch method of tensor sketch is not implemented for now
        raise NotImplementedError

import torch
from torch_scatter import scatter_add
from torch_sparse.tensor import SparseTensor
from torch_sparse.matmul import matmul as sparse_sparse_matmul


@torch.jit.script
def hash_dense_matmul(hash_indices: torch.Tensor, out_dim: int, matrix: torch.Tensor) -> torch.Tensor:
    return scatter_add(matrix, hash_indices, dim=-2, dim_size=out_dim)


@torch.jit.script
def transpose_hash_dense_matmul(hash_indices: torch.Tensor, in_dim: int, matrix: torch.Tensor) -> torch.Tensor:
    out = matrix.index_select(-2, hash_indices)
    out = scatter_add(out, torch.arange(in_dim, dtype=torch.int64).to(hash_indices.device),
                      dim=-2, dim_size=in_dim)
    return out


@torch.jit.script
def zero_one_sparse_dense_matmul(row_indices: torch.Tensor, col_indices: torch.Tensor, dim: int,
                                 matrix: torch.Tensor) -> torch.Tensor:
    out = matrix.index_select(-2, col_indices)
    out = scatter_add(out, row_indices, dim=-2, dim_size=dim)
    return out


@torch.jit.script
def sparse_dense_matmul(row_indices: torch.Tensor, col_indices: torch.Tensor, value: torch.Tensor, dim: int,
                        matrix: torch.Tensor) -> torch.Tensor:
    out = matrix.index_select(-2, col_indices)
    out = out * value.unsqueeze(-1)
    out = scatter_add(out, row_indices, dim=-2, dim_size=dim)
    return out

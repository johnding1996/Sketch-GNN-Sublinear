import torch

from sketch import *


def top_k_sparsifying(conv_sketch, top_k):
    conv_sketch_sparsified = torch.topk(conv_sketch, k=top_k, dim=-1)
    conv_sketch_sparsified = SparseTensor(
        row=torch.arange(conv_sketch.size(0), dtype=torch.long).repeat_interleave(top_k),
        col=conv_sketch_sparsified.indices.flatten(), value=conv_sketch_sparsified.values.flatten(),
        sparse_sizes=(conv_sketch.size(0), conv_sketch.size(1)), is_sorted=False)
    return conv_sketch_sparsified


def initialize_single_layer_sketch_modules(in_dim, out_dim, order, mode, device=torch.device('cpu'),
                                           count_sketch_config=DEFAULT_COUNT_SKETCH_CONFIG,
                                           tensor_sketch_config=DEFAULT_TENSOR_SKETCH_CONFIG):
    if mode in ['all_distinct', 'order_distinct']:
        count_sketch_list = nn.ModuleList(
            [CountSketch(in_dim, out_dim, config=count_sketch_config) for _ in range(order)])
    elif mode in ['layer_distinct', 'all_same']:
        count_sketch_list = nn.ModuleList(
            [CountSketch(in_dim, out_dim, config=count_sketch_config)] * order)
    else:
        raise NotImplementedError
    tensor_sketch = TensorSketch(count_sketch_list, config=tensor_sketch_config)
    return count_sketch_list.to(device), tensor_sketch.to(device)


def initialize_sketch_modules(n_layers, in_dim, out_dim, order, mode, device=torch.device('cpu'),
                              count_sketch_config=DEFAULT_COUNT_SKETCH_CONFIG,
                              tensor_sketch_config=DEFAULT_TENSOR_SKETCH_CONFIG):
    if mode in ['all_distinct', 'layer_distinct']:
        all_sketch_modules = [initialize_single_layer_sketch_modules(in_dim, out_dim, order, mode, device,
                                                                     count_sketch_config, tensor_sketch_config)
                              for _ in range(n_layers + 1)]
    elif mode in ['order_distinct', 'all_same']:
        all_sketch_modules = [initialize_single_layer_sketch_modules(in_dim, out_dim, order, mode, device,
                                                                     count_sketch_config, tensor_sketch_config)
                              ] * (n_layers + 1)
    else:
        raise NotImplementedError
    return all_sketch_modules


def sketch_convolution_matrix(conv_mat, count_sketch_list, tensor_sketch, top_k):
    tensor_sketched_conv_mat_list = tensor_sketch(conv_mat)
    return [[top_k_sparsifying(cs(ts_conv_mat.T), top_k=top_k).cpu() for ts_conv_mat in tensor_sketched_conv_mat_list]
            for cs in count_sketch_list]


def sketch_node_feature_matrix(node_feature_mat, count_sketch_list):
    return [cs(node_feature_mat).cpu() for cs in count_sketch_list]


def preprocess_data(n_layers, in_dim, out_dim, order, top_k, mode, nf_mat, conv_mat, device=torch.device('cpu'),
                    count_sketch_config=DEFAULT_COUNT_SKETCH_CONFIG, tensor_sketch_config=DEFAULT_TENSOR_SKETCH_CONFIG):
    conv_mat = conv_mat.to(device)
    nf_mat = nf_mat.to(device)
    all_sketch_modules = initialize_sketch_modules(n_layers, in_dim, out_dim, order, mode, device,
                                                   count_sketch_config, tensor_sketch_config)
    conv_sketches = [sketch_convolution_matrix(conv_mat,
                                               count_sketch_list=all_sketch_modules[l + 1][0],
                                               tensor_sketch=all_sketch_modules[l][1],
                                               top_k=top_k)
                     for l in range(n_layers)]
    nf_sketches = sketch_node_feature_matrix(nf_mat, count_sketch_list=all_sketch_modules[0][0])
    ll_cs_list = [count_sketch.clear_cache() for count_sketch in all_sketch_modules[-1][0].cpu()]
    return nf_sketches, conv_sketches, ll_cs_list


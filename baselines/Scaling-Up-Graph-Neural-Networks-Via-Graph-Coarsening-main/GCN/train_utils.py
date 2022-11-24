import torch
import torch.nn.functional as F


def format_ordinal(n):
    assert 1 <= n
    return str(n)+('th' if 4 <= n else {1: 'st', 2: 'nd', 3: 'rd'}.get(n, 'th'))


def parse_activation(activation):
    if activation == 'ReLU':
        return F.relu
    elif activation == 'Sigmoid':
        return torch.sigmoid
    elif activation == 'None':
        return lambda x: x
    else:
        assert False


def load_nested_list(nested_list, device):
    return [item.to(device) if hasattr(item, 'to') else load_nested_list(item, device) for item in nested_list]
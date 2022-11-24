import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class SketchGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, order):
        super(SketchGCNConv, self).__init__()
        self.order = order
        self.weight = Parameter(torch.empty((in_channels, out_channels), dtype=torch.float32), requires_grad=True)
        self.bias = Parameter(torch.empty(out_channels, dtype=torch.float32), requires_grad=True)
        self.coeffs = Parameter(torch.empty(order, dtype=torch.float32), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        self.coeffs.data.fill_(0)
        self.coeffs.data[0] = 1.0

    def forward(self, nf_mats, conv_mats):
        if self.training:
            zs = [torch.fft.rfft(nf_mats[0] @ self.weight + self.bias, dim=0)]
            for degree in range(2, self.order + 1):
                zs.append(torch.fft.rfft(nf_mats[degree - 1] @ self.weight + self.bias, dim=0))
                zs[-1] = zs[-1] * zs[-2]
            zs = list(map(lambda _: torch.fft.irfft(_, dim=0), zs))
            return [sum([self.coeffs[degree - 1] * (c @ z) for degree, c, z in zip(range(1, self.order + 1), cs, zs)])
                    for cs in conv_mats]
        else:
            zs = conv_mats @ (nf_mats @ self.weight + self.bias)
            return sum([self.coeffs[degree - 1] * torch.pow(zs, degree)
                        for degree in range(1, self.order + 1)])


class SketchGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 batchnorm, dropout, order):
        super(SketchGCN, self).__init__()
        self.order = order
        self.convs = torch.nn.ModuleList()
        self.convs.append(SketchGCNConv(in_channels, hidden_channels, order))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SketchGCNConv(hidden_channels, hidden_channels, order))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SketchGCNConv(hidden_channels, out_channels, order))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, nf_mats, conv_mats):
        if self.training:
            nf_mats = self.convs[0](nf_mats, conv_mats[0])
            if self.batchnorm:
                nf_mats = [self.bns[0](nf_mat) for nf_mat in nf_mats]
            for i, conv in enumerate(self.convs[1:-1]):
                nf_mats_add = conv(nf_mats, conv_mats[i+1])
                nf_mats = [nf_mat + nf_mat_add for nf_mat, nf_mat_add in zip(nf_mats, nf_mats_add)]
                if self.batchnorm:
                    nf_mats = [self.bns[i+1](nf_mat) for nf_mat in nf_mats]
                nf_mats = [F.dropout(nf_mat, p=self.dropout, training=self.training) for nf_mat in nf_mats]
            nf_mats = self.convs[-1](nf_mats, conv_mats[-1])
            return nf_mats
        else:
            nf_mats = self.convs[0](nf_mats, conv_mats)
            if self.batchnorm:
                nf_mats = self.bns[0](nf_mats)
            for i, conv in enumerate(self.convs[1:-1]):
                nf_mats = nf_mats + conv(nf_mats, conv_mats)
                if self.batchnorm:
                    nf_mats = self.bns[i+1](nf_mats)
                nf_mats = F.dropout(nf_mats, p=self.dropout, training=self.training)
            nf_mats = self.convs[-1](nf_mats, conv_mats)
            return nf_mats

from torch import nn
from torch_geometric.nn import GCNConv

from train_utils import *


class StandardGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 batchnorm, dropout, activation):
        super(StandardGCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False, bias=True))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False, bias=True))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False, bias=True))
        self.dropout = dropout
        self.activation = parse_activation(activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, nf_mat, conv_mat):
        for i, conv in enumerate(self.convs[:-1]):
            nf_mat = conv(nf_mat, conv_mat)
            if self.batchnorm:
                nf_mat = self.bns[i](nf_mat)
            nf_mat = self.activation(nf_mat)
            nf_mat = F.dropout(nf_mat, p=self.dropout, training=self.training)
        nf_mat = self.convs[-1](nf_mat, conv_mat)
        return nf_mat


class PolyActGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 batchnorm, dropout, order):
        super(PolyActGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False, bias=True))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False, bias=True))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False, bias=True))
        self.dropout = dropout
        self.coeffs = torch.nn.Parameter(
            torch.empty((num_layers - 1, order + 1), dtype=torch.float32), requires_grad=True)
        self.order = order

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()
        self.coeffs.data.fill_(0)
        self.coeffs.data[:, 1] = 1.0

    def forward(self, nf_mat, conv_mat):
        for i, conv in enumerate(self.convs[:-1]):
            nf_mat = conv(nf_mat, conv_mat)
            if self.batchnorm:
                nf_mat = self.bns[i](nf_mat)
            nf_mat = self.coeffs[i, 0] + sum(
                [self.coeffs[i, k + 1] * torch.pow(nf_mat, k + 1) for k in range(self.order)])
            nf_mat = F.dropout(nf_mat, p=self.dropout, training=self.training)
        nf_mat = self.convs[-1](nf_mat, conv_mat)
        return nf_mat

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from train_utils import *
from torch import nn

class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class StandardGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 batchnorm, dropout, activation):
        super(StandardGCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=True, bias=True))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=True, bias=True))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=True, bias=True))
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
        # return nf_mat
        return F.log_softmax(nf_mat, dim=1)
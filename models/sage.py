import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from dgl.heterograph import DGLBlock
import dgl.function as fn
from dgl.utils import expand_as_pair
import numpy as np


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 embed_dim,
                 num_layers: int,
                 act: str = 'ReLU',
                 bn: bool = False,
                 end_up_with_fc=False,
                 bias=True):
        super(MLP, self).__init__()
        self.module_list = []
        for i in range(num_layers):
            d_in = input_dim if i == 0 else hidden_dim
            d_out = embed_dim if i == num_layers - 1 else hidden_dim
            self.module_list.append(nn.Linear(d_in, d_out, bias=bias))
            if end_up_with_fc:
                continue
            if bn:
                self.module_list.append(nn.BatchNorm1d(d_out))
            self.module_list.append(getattr(nn, act)(True))
        self.module_list = nn.Sequential(*self.module_list)

    def forward(self, x):
        return self.module_list(x)


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.res_linears = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean', bias=False, feat_drop=dropout))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.res_linears.append(torch.nn.Linear(in_feats, n_hidden))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean', bias=False, feat_drop=dropout))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
            self.res_linears.append(torch.nn.Identity())
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean', bias=False, feat_drop=dropout))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.res_linears.append(torch.nn.Identity())
        self.mlp = MLP(in_feats + n_hidden * n_layers, 2 * n_classes, n_classes, num_layers=2, bn=True,
                       end_up_with_fc=True, act='LeakyReLU')
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.profile = locals()

    def forward(self, blocks):
        collect = []
        h = blocks[0].srcdata['feat']
        h = self.dropout(h)
        num_output_nodes = blocks[-1].num_dst_nodes()
        collect.append(h[:num_output_nodes])
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_res = h[:block.num_dst_nodes()]
            h = layer(block, h)
            h = self.bns[l](h)
            h = self.activation(h)
            h = self.dropout(h)
            collect.append(h[:num_output_nodes])
            h += self.res_linears[l](h_res)
        return self.mlp(torch.cat(collect, -1))


def count_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


if __name__ == '__main__':
    sage = SAGE(128, 1024, 172, 3, torch.nn.functional.relu, 0)
    print(count_parameters(sage))

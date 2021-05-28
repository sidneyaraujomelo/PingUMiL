import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

"""
This implementation of GAT is derived from Pytorch Geometric's example
script gat.py and Pytorch Geometric's example
script ogbn_products_sage.
"""

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_heads, dropout):
        super(GAT, self).__init__()
        self.num_layers = len(hidden_channels)+1
        self.convs = torch.nn.ModuleList()
        if len(hidden_channels) > 0:
            self.convs.append(GATConv(in_channels, hidden_channels[0], heads=n_heads,
                                      dropout=dropout))
            for i in range(len(hidden_channels)-1):
                self.convs.append(GATConv(hidden_channels[i]*n_heads,
                                          hidden_channels[i+1],
                                          heads=n_heads,
                                          dropout=dropout))
            self.convs.append(GATConv(hidden_channels[-1]*n_heads, out_channels, heads=1,
                                      dropout=dropout))
        else:
            self.convs.append(GATConv(in_channels, out_channels, nheads=1,
                                      dropout=dropout))
        #for conv in self.convs:
            #torch.nn.init.kaiming_uniform_(conv.lin_l.weight)
            #torch.nn.init.kaiming_uniform_(conv.lin_r.weight)
        self.dropout = dropout
  
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
  
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def inference(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
        return x
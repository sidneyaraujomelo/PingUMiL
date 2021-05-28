import math
import torch
import numpy as np
from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_undirected
import tqdm

def tri2square(tri_idx):
    tri_idx = np.asarray(tri_idx).astype(np.float64, copy=False)
    up_bound = (1 + np.sqrt(1 + 8 * tri_idx)) / 2
    i = np.floor(up_bound)
    j = tri_idx - (i - 1) * i / 2
    return i.astype(np.int64, copy=False), j.astype(np.int64, copy=False)

def sparse_train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    #neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    """neg_row = torch.full((num_nodes-2,),0)
    neg_col = torch.arange(1, num_nodes-1)
    for i in tqdm.tqdm(range(3,num_nodes), total=num_nodes-3):
        neg_row = torch.cat(
            (
                neg_row,
                torch.full( (num_nodes-i,) , i-2)
            ), 0
        )
        neg_col = torch.cat(
            (
                neg_col,
                torch.arange(i-1, num_nodes-1)
            ), 0
        )"""
    num_neg_edges = int((num_nodes - 1)*num_nodes/2)
    ind = np.arange(num_neg_edges)
    i,j = tri2square(ind)
    # construct the symmetric upper triangular part
    neg_row = np.concatenate([i, j])
    neg_col = np.concatenate([j, i])
    neg_val = torch.ones_like(neg_row)
    neg_adj_mask = SparseTensor(row=neg_row, col=neg_col, values=neg_val, sparse_sizes=(num_nodes, num_nodes)).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data
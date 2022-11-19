import json
import os
import torch
import numpy as np
from argparse import ArgumentParser
from networkx.readwrite import json_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
from torch_cluster import random_walk
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pingumil.models import load_model
from pingumil.util.pytorchtools import EarlyStopping

dataset_folder = "dataset/ct_final_test"
dataset_prefix = "prov"
model_config = "configs/ct_sagemodel.json"

#First, we load all the data in the dataset folder.
graph_data = json.load(open(os.path.join(dataset_folder,
                                        f"{dataset_prefix}-G.json")))
graph = json_graph.node_link_graph(graph_data)
node_feats = np.load(os.path.join(dataset_folder,
                                f"{dataset_prefix}-feats.npy"))

#Now, we load the files structuring the folds for k-fold cross validation
#containing positive and negative indirect edges
fold_files = sorted(os.listdir(
    os.path.join(dataset_folder, "clf")
))
train_folds = {p : json.load(
                            open(os.path.join(dataset_folder,"clf",p))
                            ) for p in fold_files if "train" in p}
test_folds = {p : json.load(
                            open(os.path.join(dataset_folder,"clf",p))
                            ) for p in fold_files if "test" in p}

#Create data object for pytorch geometric
data = from_networkx(graph)
data.x = torch.from_numpy(node_feats).float()
#Column-wise normalization of features
data.x = data.x / data.x.max(0,keepdim=True).values
data.x[torch.isnan(data.x)] = 0

#Configuration
num_samples = [2, 2]
batch_size = 64
hidden_channels = 128
walk_length = 1
num_neg_samples = 1
epochs = 1000

subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=[-1], batch_size=batch_size, shuffle=False,
    num_workers=0)


"""class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SAGE, self).__init__()
        self.num_layers = 2
        self.convs = nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels, normalize=True))
        self.convs.append(
            SAGEConv(hidden_channels, hidden_channels, normalize=True))

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
"""

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
sage_config = json.load(open(model_config))[0]
sage_config["in_channels"] = data.num_node_features
model = load_model(sage_config)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x = data.x.to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=data.num_nodes)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0

    node_idx = torch.randperm(data.num_nodes)  # shuffle all nodes
    train_loader = NeighborSampler(
        data.edge_index, node_idx=node_idx,
        sizes=num_samples, batch_size=batch_size, shuffle=False,
        num_workers=0)

    rw = random_walk(
        data.edge_index[0], data.edge_index[1],
        node_idx, walk_length=walk_length)
    rw_idx = rw[:, 1:].flatten()
    pos_loader = NeighborSampler(
        data.edge_index, node_idx=rw_idx,
        sizes=num_samples, batch_size=batch_size * walk_length, shuffle=False,
        num_workers=0)

    # negative sampling as node2vec
    deg = degree(data.edge_index[0])
    distribution = deg ** 0.75
    neg_idx = torch.multinomial(
        distribution, data.num_nodes * num_neg_samples, replacement=True)
    neg_loader = NeighborSampler(
        data.edge_index, node_idx=neg_idx,
        sizes=num_samples, batch_size=batch_size * num_neg_samples,
        shuffle=True, num_workers=0)

    for (batch_size_, u_id, adjs_u), (_, v_id, adjs_v), (_, vn_id, adjs_vn) in\
            zip(train_loader, pos_loader, neg_loader):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs_u = [adj.to(device) for adj in adjs_u]
        z_u = model(x[u_id], adjs_u)

        adjs_v = [adj.to(device) for adj in adjs_v]
        z_v = model(x[v_id], adjs_v)

        adjs_vn = [adj.to(device) for adj in adjs_vn]
        z_vn = model(x[vn_id], adjs_vn)

        optimizer.zero_grad()
        pos_loss = -F.logsigmoid(
            (z_u.repeat_interleave(walk_length, dim=0)*z_v)
            .sum(dim=1)).mean()
        neg_loss = -F.logsigmoid(
            -(z_u.repeat_interleave(num_neg_samples, dim=0)*z_vn)
            .sum(dim=1)).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.update(batch_size_)

    pbar.close()

    loss = total_loss / len(train_loader)
    return loss

early_stopping = EarlyStopping(patience=50, verbose=True, path="embed_model.pt")
best_loss = 50
for epoch in range(1, epochs):
    loss = train(epoch)
    if loss < best_loss:
        best_loss = loss
    early_stopping(loss, model)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
    if early_stopping.early_stop:
        print("Early Stopping!")
        break
print(f'Best Loss: {best_loss:.4f}')

model.load_state_dict(torch.load(early_stopping.path))
z = model.inference(x, subgraph_loader, device).detach()
z = z.to(device)
model_config = "configs/ct_sagemodel.json"
linkpred_config = json.load(open(model_config))[1]
linkpred_config["in_channels"] = z.size(1)*2
linkpred_model = load_model(linkpred_config).to(device)

for k in range(len(train_folds.keys())):
    
    train_fold = train_folds[f"clffold-{k}-train"]
    test_fold = test_folds[f"clffold-{k}-test"]
    
    #Transform Edge structure from fold files into COO tensors
    # Also obtains the class of each edge for both train and test
    train_source_ids = [x["source"] for x in train_fold]
    train_target_ids = [x["target"] for x in train_fold]
    train_edges = torch.tensor([train_source_ids, train_target_ids])
    train_class = torch.FloatTensor([x["class"] for x in train_fold])
    test_source_ids = [x["source"] for x in test_fold]
    test_target_ids = [x["target"] for x in test_fold]
    test_edges = torch.tensor([test_source_ids, test_target_ids])
    test_class = torch.FloatTensor([x["class"] for x in test_fold])
    
    train_edges = train_edges.to(device)
    test_edges = test_edges.to(device)
    train_class = train_class.to(device)
    test_class = test_class.to(device)
    link_optimizer = torch.optim.Adam(linkpred_model.parameters(), lr=0.01,
                                      weight_decay=5e-4)
    early_stopping = EarlyStopping(patience=30, verbose=True, path="predictor.pt")
    
    best_metrics = {
        "train_loss": 50,
        "p": 0,
        "r": 0
    }

    print(f"Starting fold {k}")
    for epoch in range(1,200):
        linkpred_model.train()
        link_optimizer.zero_grad()
        out = linkpred_model(z, train_edges)
        
        train_loss = F.binary_cross_entropy_with_logits(out, train_class.unsqueeze(1))
        y_pred_tag = torch.round(torch.sigmoid(out))
        
        corrects_results_sum = torch.eq(y_pred_tag, train_class.unsqueeze(1))
        train_acc = float(corrects_results_sum.sum().item())/train_class.size()[0]
        
        train_loss.backward()
        link_optimizer.step()

        linkpred_model.eval()

        out_hat = linkpred_model(z, test_edges)
        test_loss = F.binary_cross_entropy_with_logits(out_hat, test_class.unsqueeze(1))
        out_pred = torch.round(torch.sigmoid(out_hat)).detach()
        
        test_correct = torch.eq(out_pred, test_class.unsqueeze(1))
        test_acc =  float(test_correct.sum().item()) / test_class.size()[0]
        print(f"{k}: Train loss: {train_loss} Train acc: {train_acc}")
        #print(f"Test loss: {test_loss} Test acc:{test_acc} "
        #      +f"P:{precision_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())} "
        #      +f"R:{recall_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())}")
        p = precision_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())
        r = recall_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())
        if train_loss.cpu() < best_metrics["train_loss"]:
            best_metrics["train_loss"] = train_loss.cpu()
            best_metrics["r"] = r
            best_metrics["p"] = p

        early_stopping(train_loss, linkpred_model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break
    print(best_metrics)

import torch
import torch.nn as nn
import json
import numpy as np
import os
import time
from networkx.readwrite import json_graph
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import train_test_split_edges
from tqdm import tqdm
from torch_geometric.data import NeighborSampler
from torch_cluster import random_walk
from pingumil.models import load_model
from pingumil.models.graphsage import GraphSAGE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dataset_folder = "dataset/PUC_IFRJ_UFF/pingdataset"
dataset_prefix = "prov"

def train(epoch, model, data, train_loader):
    model.train()
    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        
        optimizer.zero_grad()
        out = model(data.x[n_id], adjs)
        #print(n_id)
        loss = F.binary_cross_entropy_with_logits(out, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        labels = data.y[n_id[:batch_size]]
        total_correct += int(out.argmax(dim=-1).eq(labels.argmax(dim=-1)).sum())
        #print(f"{total_correct} {data.train_mask.sum().item()}")

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / data.train_mask.sum().item()
    return loss, approx_acc

@torch.no_grad()
def test(epoch, model, data, subgraph_loader, device):
    model.eval()
    out = model.inferences(data.x, subgraph_loader, device)
    out_test = out[data.test_mask].to(data.test_mask.device)
    y_test = data.y[data.test_mask]

    test_loss = F.binary_cross_entropy_with_logits(
        F.log_softmax(out_test, dim=1),
        y_test
    )

    test_correct = int(out_test.argmax(dim=-1).eq(y_test.argmax(dim=-1)).sum())
    test_acc = test_correct / data.test_mask.sum().item()
    if epoch == 50:
        cm = confusion_matrix(y_test.argmax(dim=-1).cpu().numpy(),
                              out_test.argmax(dim=-1).cpu().numpy())
        f = sns.heatmap(cm, annot=True, fmt="d").get_figure()
        f.savefig("heatmap_cm.png")

    return test_loss, test_acc
    

#First, we load all the data in the dataset folder.
graph_data = json.load(open(os.path.join(dataset_folder,
                            f"{dataset_prefix}-G.json")))
graph = json_graph.node_link_graph(graph_data)
node_feats = np.load(os.path.join(dataset_folder,
                                  f"{dataset_prefix}-feats.npy"),
                     allow_pickle=True)
node_labels = json.load(open(os.path.join(dataset_folder,
                                          f"{dataset_prefix}-labels.json")))
data = from_networkx(graph)
data.x = torch.from_numpy(node_feats).float()
data.y = torch.Tensor(list(node_labels.values())).float()
train_idx, test_idx = train_test_split(np.arange(data.y.shape[0]),
                                       test_size = 0.2,
                                       shuffle=True,
                                       stratify=list(node_labels.values()))
train_mask = [1 if x in train_idx else 0 for x in range(data.y.shape[0])]
data.train_mask = torch.BoolTensor(train_mask)
data.test_mask = ~data.train_mask
#data = train_test_split_edges(data)
print(data.train_mask)

sage_config = {
    "model" : "graphsage",
    "in_channels" : data.x.shape[-1],
    "hidden_channels": [256],
    "out_channels": data.y.shape[-1],
    "dropout" : 0.5
}
sage_model = load_model(sage_config)
print(sage_model)

train_loader = NeighborSampler(data.edge_index, sizes=[2,2],
                               node_idx=data.train_mask, batch_size=512,
                               shuffle=True)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
sage_model = sage_model.to(device)
optimizer = torch.optim.Adam(sage_model.parameters(), lr=0.001, weight_decay=5e-4)

start = time.time()
for epoch in range(1,51):
    tr_loss, tr_acc = train(epoch, sage_model, data, train_loader)
    #te_loss, te_acc = 0, 0
    te_loss, te_acc = test(epoch, sage_model, data, subgraph_loader, device)
    print(f"Train loss: {tr_loss}, Train acc: {tr_acc}, Test loss: {te_loss}, Test acc: {te_acc}")
print(f"Time elapsed: {time.time() - start}")
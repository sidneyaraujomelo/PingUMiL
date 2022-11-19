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
from torch_geometric.data import NeighborSampler, Data, DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
from torch_cluster import random_walk
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import wandb
from pingumil.models import load_model
from pingumil.util.pygtools import sparse_train_test_split_edges
from pingumil.experiments.donors.donors_exp import DonorsBaseExperiment
import pickle
import time

run = wandb.init(project="fsp", group="donors_tsp",
                 config={
                    "batch_size" : 1024,
                    "gnn" : "gat"
                 })
experiment = DonorsBaseExperiment(
    experiment_tag="linkpred",
    epochs=1000,
    patience=100,
    model_config="configs/donors_gatmodel.json",
    wandb=wandb,
    timestamp=time.time(),
    split_edges=False,
    save_split=False
)

print(experiment.output_file)
#Read all necessary data
data, atbsets_list, node_maps, node_feats = experiment.read_data()


dim_types = [x.size()[1] for x in node_feats]
print(dim_types)

experiment.log(f"Standardization: {experiment.standardization}\n")

#Attribute normalization
if experiment.standardization:
    #Get list of all attributes in any type of node
    atbs_list = [atb for atbsets_sublist in atbsets_list for atb in atbsets_sublist]
    #print(atbs_list)
    atbs_list = list(dict.fromkeys(atbs_list))
    #Create a dictionary mapping each attribute to its index on node_feats
    atbs_maps = {atb : {} for atb in atbs_list}
    f = lambda l,v: l.index(v) if v in l else -1
    for atb in atbs_maps.keys():
        atbs_maps[atb] = {k:f(v,atb) for k,v in enumerate(atbsets_list)}
    #For each attribute, apply normalization on node_feats according to the mapping
    for atb, atb_map in atbs_maps.items():
        #print(f"Standardization for {atb}")
        scaler = StandardScaler()
        atbdata = torch.cat(tuple([node_feats[k][:,v] for k,v in atb_map.items() if v != -1]))
        #print(atbdata)
        if ((torch.min(atbdata).item() == 0 and torch.max(atbdata).item() == 1) or torch.equal(atbdata, torch.zeros_like(atbdata))
                or torch.equal(atbdata, torch.ones_like(atbdata))):
            #print(f"Continuning for {atb}, possible One-Hot-Encoded")
            continue
        split_dim = [node_feats[k][:,v].shape[0] for k,v in atb_map.items() if v!=-1]
        atbdata_t = atbdata.reshape(-1,1)
        atbdata_std = torch.from_numpy(scaler.fit_transform(atbdata_t).reshape(1,-1))
        split_atbdata_std = torch.split(atbdata_std[0], split_dim)
        i = 0
        for k,v in atb_map.items():
            if v == -1:
                continue
            node_feats[k][:,v] = split_atbdata_std[i]
            i = i + 1
    
#Configuration
num_samples = [-1,-1]
walk_length = 1
batch_size = wandb.config.batch_size
num_neg_samples = 1
epochs = experiment.epochs
gnn_input_dim = 128

data.x_ts = node_feats
"""
data.maps = node_maps
dict_x2m = {}
#update edge_index according to node_maps
for node_map in node_maps:
    offset = len(dict_x2m)
    dict_x2m.update({k+offset:v for k,v in enumerate(node_map)})
data.edge_index[0] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[0].tolist()])
data.edge_index[1] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[1].tolist()])
#data.x = torch.zeros((data.origin.size()[0], gnn_input_dim))
"""
def find_nodemap(fnid):
    offset = 0
    for i,v in enumerate(node_maps):
        if fnid < offset + len(v):
            return i
        offset += len(v)
edge_type_function = lambda x: experiment.node_type_labels[find_nodemap(x)]

#Now, we split dataset in train/test if necessary
#if not experiment.split_edges:
#    data = train_test_split_edges(data)
print(data)
#Column-wise normalization of features
#data.x = data.x / data.x.max(0,keepdim=True).values
#data.x[torch.isnan(data.x)] = 0

subgraph_loader = NeighborSampler(
    data.train_pos_edge_index, node_idx=None,
    sizes=[-1], batch_size=batch_size, shuffle=False,
    num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gnn_config = json.load(open(experiment.model_config))[0]
gnn_config["in_channels"] = gnn_input_dim
gnn_model = load_model(gnn_config)
gnn_model = gnn_model.to(device)

typeproj_config = {
    "model": "typeprojection",
    "dim_types": dim_types,
    "dim_output": gnn_config["in_channels"]
}
typeproj_model = load_model(typeproj_config)
typeproj_model = typeproj_model.to(device)

typeproj_optimizer = torch.optim.Adam(typeproj_model.parameters(), lr=1e-3)
gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-3)
#x = data.x.to(device)
#print(x)
x_ts = [x_t.to(device) for x_t in data.x_ts]

#torch.autograd.set_detect_anomaly(True)
experiment.log(f"Device: {device}, visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}\n")
experiment.log(f"Type Projection: {typeproj_model}\n")
experiment.log(f"Graph Representation Learning Model: {gnn_model}\n")

def train(epoch):
    gnn_model.train()
    typeproj_model.train()

    pbar = tqdm(total=data.num_nodes)
    pbar.set_description(f'Epoch {epoch:02d}')

    #Type Projection Step
    #print(x)
    x_p = typeproj_model(x_ts, data.maps)
    #print(x_p)
    #print(x[0])

    total_loss = 0

    node_idx = torch.randperm(data.num_nodes)  # shuffle all nodes
    train_loader = NeighborSampler(
        data.train_pos_edge_index, node_idx=node_idx,
        sizes=num_samples, batch_size=batch_size, shuffle=False,
        num_workers=0)

    rw = random_walk(
        data.train_pos_edge_index[0], data.train_pos_edge_index[1],
        node_idx, walk_length=walk_length)
    rw_idx = rw[:, 1:].flatten()
    pos_loader = NeighborSampler(
        data.train_pos_edge_index, node_idx=rw_idx,
        sizes=num_samples, batch_size=batch_size * walk_length, shuffle=False,
        num_workers=0)

    # negative sampling as node2vec
    deg = degree(data.train_pos_edge_index[0])
    distribution = deg ** 0.75
    neg_idx = torch.multinomial(
        distribution, data.num_nodes * num_neg_samples, replacement=True)
    neg_loader = NeighborSampler(
        data.train_pos_edge_index, node_idx=neg_idx,
        sizes=num_samples, batch_size=batch_size * num_neg_samples,
        shuffle=True, num_workers=0)

    typeproj_optimizer.zero_grad()

    for (batch_size_, u_id, adjs_u), (_, v_id, adjs_v), (_, vn_id, adjs_vn) in\
            zip(train_loader, pos_loader, neg_loader):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs_u = [adj.to(device) for adj in adjs_u]
        z_u = gnn_model(x_p[u_id], adjs_u)

        adjs_v = [adj.to(device) for adj in adjs_v]
        z_v = gnn_model(x_p[v_id], adjs_v)

        adjs_vn = [adj.to(device) for adj in adjs_vn]
        z_vn = gnn_model(x_p[vn_id], adjs_vn)

        gnn_optimizer.zero_grad()
        pos_loss = -F.logsigmoid(
            (z_u.repeat_interleave(walk_length, dim=0)*z_v)
            .sum(dim=1)).mean()
        neg_loss = -F.logsigmoid(
            -(z_u.repeat_interleave(num_neg_samples, dim=0)*z_vn)
            .sum(dim=1)).mean()
        loss = pos_loss + neg_loss
        loss.backward(retain_graph=True)
        gnn_optimizer.step()
        
        wandb.log({"emb batch loss":loss.item()})
        total_loss += loss.item()
        pbar.update(batch_size_)
    typeproj_optimizer.step()
    pbar.close()

    loss = total_loss / len(train_loader)
    wandb.log({"emb loss":loss})

    return loss

gnn_early_stopping = experiment.get_early_stopping(patience=experiment.patience, verbose=True, prefix="embed")
proj_early_stopping = experiment.get_early_stopping(patience=experiment.patience, verbose=True, prefix=f"proj")
best_loss = 50
for epoch in range(1, epochs):
    loss = train(epoch)
    if loss < best_loss:
        best_loss = loss
    gnn_early_stopping(loss, gnn_model)
    proj_early_stopping(loss, typeproj_model)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
    if gnn_early_stopping.early_stop:
        print("Early Stopping!")
        break
print(f'Best Loss: {best_loss:.4f}')
experiment.log(f'Epoch: {epoch} -> Loss (Representation Learning): {best_loss:.4f}')

typeproj_model.load_state_dict(torch.load(proj_early_stopping.path))
gnn_model.load_state_dict(torch.load(gnn_early_stopping.path))
x_p = typeproj_model(x_ts, data.maps)
train_pos_edge_index = data.train_pos_edge_index.to(device)
z = gnn_model.inference(x_p, train_pos_edge_index).detach()
z = z.to(device)
linkpred_config = json.load(open(experiment.model_config))[1]
linkpred_config["in_channels"] = z.size(1)*2
linkpred_model = load_model(linkpred_config).to(device)


experiment.log(f"Link Prediction Model: {linkpred_model}\n")

link_optimizer = torch.optim.Adam(linkpred_model.parameters(), lr=1e-3)

experiment.link_prediction_step(device, linkpred_model, link_optimizer, data, z, edge_type_function, wandb)
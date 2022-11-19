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
from torch_geometric.utils import degree
from torch_cluster import random_walk
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import wandb
from pingumil.models import load_model
from pingumil.util.pygtools import sparse_train_test_split_edges
from pingumil.experiments.sshet.sshet_exp import SSHetBaseExperiment
from pingumil.util.metric import get_edge_type_list
import pickle
import time

EPS = 1e-15

run = wandb.init(project="fsp", group="sshet_tsp",
                 config={
                    "batch_size" : 2 ** 12,
                    "gnn" : "hgt",
                    "lr" : 1e-5,
                    "weight_decay" : 1e-3,
                    "lr_lp" : 1e-2
                 })
experiment = SSHetBaseExperiment(
    experiment_tag="linkpred",
    epochs=1000,
    patience=20,
    model_config="configs/donors_hgtmodel.json",
    wandb=wandb,
    timestamp=time.time()
)

print(experiment.output_file)
#Read all necessary data
data, atbsets_list, node_maps, node_feats, train_folds, test_folds = experiment.read_data()

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
data.maps = node_maps
dict_x2m = {}
#update edge_index according to node_maps
for node_map in node_maps:
    offset = len(dict_x2m)
    dict_x2m.update({k+offset:v for k,v in enumerate(node_map)})
data.edge_index[0] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[0].tolist()])
data.edge_index[1] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[1].tolist()])
data.x = torch.zeros((data.origin.size()[0], gnn_input_dim))

def find_nodemap(fnid):
    offset = 0
    for i,v in enumerate(node_maps):
        if fnid < offset + len(v):
            return i
        offset += len(v)

num_nodes = sum([x.size(0) for x in node_feats])
node_types = [find_nodemap(x) for x in range(num_nodes)]

def find_nodeunstacked(snid):
    offsets = [len(v) for i,v in enumerate(node_maps)]
    for k in range(len(offsets)):
        if snid < sum([offsets[:k+1]]):
            if k == 0:
                return snid
            return snid - sum([offsets[:k]])

edge_type_function = lambda x: experiment.node_type_labels[find_nodemap(x)]
edge_type_dict = {}
num_edge_types = 0
for i in range(len(experiment.node_type_labels)):
    edge_type_dict[i] = {}
    for j in range(i,len(experiment.node_type_labels)):
        edge_type_dict[i][j] = num_edge_types
        num_edge_types += 1

def get_edge_type_index(fnids):
    node_label_ids = [find_nodemap(x) for x in fnids]
    node_label_ids.sort()
    i, j = node_label_ids[0], node_label_ids[1]
    return edge_type_dict[i][j]

#Now, we split dataset in train/test if necessary
#if not experiment.split_edges:
#    data = train_test_split_edges(data)
print(data)
#Column-wise normalization of features
#data.x = data.x / data.x.max(0,keepdim=True).values
#data.x[torch.isnan(data.x)] = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gnn_config = json.load(open(experiment.model_config))[0]
gnn_config["in_dim"] = 128
gnn_config["num_types"] = len(dim_types)
gnn_config["num_relations"] = num_edge_types
gnn_model = load_model(gnn_config)
print(gnn_model)
gnn_model = gnn_model.to(device)


typeproj_config = {
    "model": "typeprojection",
    "dim_types": dim_types,
    "dim_output": gnn_config["n_hid"]
}
typeproj_model = load_model(typeproj_config)
typeproj_model = typeproj_model.to(device)

typeproj_optimizer = torch.optim.Adam(typeproj_model.parameters(),
                                      lr=wandb.config.lr,
                                      weight_decay=wandb.config.weight_decay)
gnn_optimizer = torch.optim.Adam(gnn_model.parameters(),
                                 lr=wandb.config.lr,
                                 weight_decay=wandb.config.weight_decay)
#x = data.x.to(device)
#print(x)
x_ts = [x_t.to(device) for x_t in data.x_ts]
node_types = torch.Tensor(node_types).to(device)
data.edge_index = data.edge_index.to(device)

edges_as_list = zip(data.edge_index[0].tolist(),
                   data.edge_index[1].tolist())
pos_edge_types = torch.Tensor(
    [get_edge_type_index([node_types[i],node_types[j]]) for i,j in edges_as_list])
pos_edge_types = pos_edge_types.to(device)

train_edge_time = torch.Tensor([0]*data.edge_index.size(-1))
train_edge_time = train_edge_time.to(device)

#torch.autograd.set_detect_anomaly(True)
experiment.log(f"Device: {device}, visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}\n")
experiment.log(f"Type Projection: {typeproj_model}\n")
experiment.log(f"Graph Representation Learning Model: {gnn_model}\n")

def train(epoch):
    gnn_model.train()
    typeproj_model.train()

    pbar = tqdm(total=data.edge_index.size(-1)*2)
    pbar.set_description(f'Epoch {epoch:02d}')

    #Type Projection Step
    #print(x)
    x_p = typeproj_model(x_ts, data.maps)
    #print(x_p)
    #print(x[0])

    total_loss = 0

    pos_train_edge_loader = DataLoader(torch.arange(data.edge_index.size()[-1]),
                                batch_size=wandb.config.batch_size, shuffle=True)


    train_neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes = data.num_nodes
    )
    #train_edges = torch.cat([data.train_pos_edge_index, train_neg_edge_index], dim=-1).to(device)
    neg_edges_as_list = zip(train_neg_edge_index[0].tolist(),
                            train_neg_edge_index[1].tolist())
    neg_edge_types = torch.Tensor(
        [get_edge_type_index([node_types[i],node_types[j]]) for i,j in neg_edges_as_list])
    neg_edge_types = neg_edge_types.to(device)
    neg_train_edge_loader = DataLoader(torch.arange(train_neg_edge_index.size()[-1]),
                                   batch_size=wandb.config.batch_size, shuffle=True)

    typeproj_optimizer.zero_grad()

    #for (batch_size_, u_id, adjs_u), (_, v_id, adjs_v), (_, vn_id, adjs_vn) in\
    #        zip(train_loader, pos_loader, neg_loader):
    for pos_batch_edge_idx, neg_batch_edge_idx in zip(pos_train_edge_loader, neg_train_edge_loader):
        z_pos = gnn_model(x_p, node_types, train_edge_time,
                          data.edge_index[:,pos_batch_edge_idx],
                          pos_edge_types[pos_batch_edge_idx])
        
        node_i, node_j = data.edge_index[:,pos_batch_edge_idx][0], data.edge_index[:,pos_batch_edge_idx][1]
        #print(node_i.size())
        #print(node_j.size())
        out = (z_pos[node_i] * z_pos[node_j]).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        z_neg = gnn_model(x_p, node_types, train_edge_time,
                          train_neg_edge_index[:,neg_batch_edge_idx],
                          neg_edge_types[neg_batch_edge_idx])
        node_i, node_j = train_neg_edge_index[:,neg_batch_edge_idx][0], train_neg_edge_index[:,neg_batch_edge_idx][1]
        #print(node_i.size())
        #print(node_j.size())
        out = (z_neg[node_i] * z_neg[node_j]).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()
        """
        adjs_v = [adj.to(device) for adj in adjs_v]
        z_v = gnn_model(x_p[v_id], adjs_v)

        adjs_vn = [adj.to(device) for adj in adjs_vn]
        z_vn = gnn_model(x_p[vn_id], adjs_vn)

        gnn_optimizer.zero_grad()
        pos_loss = -F.logsigmoid((z_u.repeat_interleave(walk_length, dim=0)*z_v).sum(dim=1)).mean()
        neg_loss = -F.logsigmoid(
            -(z_u.repeat_interleave(num_neg_samples, dim=0)*z_vn)
            .sum(dim=1)).mean()"""
        gnn_optimizer.zero_grad()
        loss = pos_loss + neg_loss
        loss.backward(retain_graph=True)
        gnn_optimizer.step()
        
        wandb.log({"emb batch loss":loss.item()})
        total_loss += loss.item()
        pbar.update(len(pos_batch_edge_idx)+len(neg_batch_edge_idx))
    typeproj_optimizer.step()
    pbar.close()

    loss = total_loss / len(pos_train_edge_loader)
    wandb.log({"emb loss":loss})

    return loss

gnn_early_stopping = experiment.get_early_stopping(patience=experiment.patience, delta=0.1, verbose=True, prefix="embed")
proj_early_stopping = experiment.get_early_stopping(patience=experiment.patience, delta=0.1, verbose=True, prefix=f"proj")
best_loss = np.inf
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
#train_pos_edge_index = data.train_pos_edge_index.to(device)
z = gnn_model(x_p, node_types, train_edge_time, data.edge_index, pos_edge_types).detach()
z = z.to(device)
linkpred_config = json.load(open(experiment.model_config))[1]
linkpred_config["in_channels"] = z.size(1)*2
linkpred_model = load_model(linkpred_config).to(device)


experiment.log(f"Link Prediction Model: {linkpred_model}\n")

experiment.link_prediction_step(device, linkpred_model, train_folds, z, test_folds, dict_x2m, edge_type_function, wandb)

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
from sklearn.preprocessing import StandardScaler
import wandb
from pingumil.models import load_model
from pingumil.util.metric import get_edge_type_list
from pingumil.util.pygtools import sparse_train_test_split_edges
from pingumil.experiments.donors.donors_exp import DonorsBaseExperiment
import pickle
import time

EPS = 1e-15

#wandb stuff
run = wandb.init(project="fsp", group="donors_mhafse",
                 config={
                    "batch_size" : 4096,
                    "lr" : 1e-5,
                    "weight_decay" : 1e-6,
                    "gnn" : "hgt"
                 })

experiment = DonorsBaseExperiment(
    experiment_tag="linkpred_mhafse",
    epochs=1000,
    patience=30,
    model_config="configs/donors_hgtmodel.json",
    wandb=wandb,
    timestamp=time.time(),
    split_edges=False,
    save_split=False
)
print(experiment.output_file)
#Read all necessary data
data, atbsets_list, node_maps, node_feats = experiment.read_data()

#Get list of all attributes in any type of node
atbs_list = [atb for atbsets_sublist in atbsets_list for atb in atbsets_sublist]
atbs_list = list(dict.fromkeys(atbs_list))
#Create a dictionary mapping each attribute to its index on node_feats
atbs_maps = {atb : {} for atb in atbs_list}
f = lambda l,v: l.index(v) if v in l else -1
for atb in atbs_maps.keys():
    atbs_maps[atb] = {k:f(v,atb) for k,v in enumerate(atbsets_list)}
#print(atbs_maps)
print(node_feats[0].shape)
node_typencs = []
for i,_ in enumerate(node_feats):
    new_node_feats = torch.zeros((node_feats[i].shape[0],len(atbs_list)))
    new_node_typencs = torch.zeros((node_feats[i].shape[0],len(atbs_list)))
    #type_encoding = []
    for atb, atb_dict in atbs_maps.items():
        atb_final_index = atbs_list.index(atb)
        #if atb_dict[i] == -1:
            #type_encoding.append(0)
        #else:
        if atb_dict[i] != -1:
            #type_encoding.append(1)
            new_node_feats[:,atb_final_index] = node_feats[i][:, atb_dict[i]]
            new_node_typencs[:,atb_final_index] = torch.ones(node_feats[i].shape[0])
    #print(type_encoding)
    node_feats[i] = new_node_feats
    node_typencs.append(new_node_typencs)

node_feats = torch.cat(node_feats)
node_typencs = torch.cat(node_typencs)

experiment.log(f"Standardization: {experiment.standardization}\n")
if experiment.standardization:
    node_feats = experiment.standardize(node_feats)
    
#Configuration
num_samples = [-1,-1]
walk_length = 1
batch_size = wandb.config.batch_size
mha_batch_size = 256
num_neg_samples = 1
epochs = experiment.epochs
gnn_input_dim = node_feats.shape[-1]

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

num_nodes = len(node_feats)
print(num_nodes)
print(data)
node_types = [find_nodemap(x) for x in range(num_nodes)]

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

data.x = node_feats
data.t = node_typencs
dim_types = [data.x.shape[-1]]
data.node_map = [0]

subgraph_loader = NeighborSampler(
    data.train_pos_edge_index, node_idx=None,
    sizes=[-1], batch_size=batch_size, shuffle=False,
    num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gnn_config = json.load(open(experiment.model_config))[0]
gnn_config["in_dim"] = 128
gnn_config["num_types"] = len(node_maps)
gnn_config["num_relations"] = num_edge_types
gnn_model = load_model(gnn_config)
print(gnn_model)
gnn_model = gnn_model.to(device)

n_head = 8
d_model = data.x.shape[-1]
d_k = data.x.shape[-1]
d_v = data.x.shape[-1]

mha_config = {
    "model": "mhattention",
    "n_head": n_head,
    "d_model": d_model,
    "d_k": d_k,
    "d_v": d_v,
    "out_dim" : 128
}
mha_model = load_model(mha_config)
mha_model = mha_model.to(device)
print(mha_model)

mha_optimizer = torch.optim.Adam(mha_model.parameters(),
    lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
gnn_optimizer = torch.optim.Adam(gnn_model.parameters(),
    lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

x = data.x.to(device)
t = data.t.to(device)

data_to_transform = torch.utils.data.DataLoader(torch.cat((x,t),dim=1),
                                                batch_size=mha_batch_size)

node_types = torch.Tensor(node_types).to(device)
data.train_pos_edge_index = data.train_pos_edge_index.to(device)
pos_edge_types = torch.Tensor([get_edge_type_index([node_types[i],node_types[j]]) for i,j in zip(data.train_pos_edge_index[0].tolist(), data.train_pos_edge_index[1].tolist())])
pos_edge_types = pos_edge_types.to(device)

train_edge_time = torch.Tensor([0]*data.train_pos_edge_index.size(-1))
train_edge_time = train_edge_time.to(device)

#torch.autograd.set_detect_anomaly(True)
experiment.log(f"Device: {device}, visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}\n")
experiment.log(f"MHA Projection: {mha_model}\n")
experiment.log(f"Graph Representation Learning Model: {gnn_model}\n")

def train(epoch):
    gnn_model.train()
    mha_model.train()

    pbar = tqdm(total=data.num_nodes)
    pbar.set_description(f'Epoch {epoch:02d}')

    #Type Projection Step
    #print(x)
    x_p = torch.zeros(x.size()[0],gnn_config["in_dim"]).to(device)
    for batch_it, batch in tqdm(enumerate(data_to_transform), total=len(data_to_transform), desc="Transforming features"):
    #for batch_it, batch in enumerate(data_to_transform):
        #print(batch_it)
        f_index = batch_it*data_to_transform.batch_size
        l_index = min((batch_it+1)*data_to_transform.batch_size, x_p.size(0))
        #print(batch.size(0), l_index)
        x_p[f_index:l_index,:],_ = mha_model(
            batch[:,:d_model].view(1,batch.size(0), d_model),
            batch[:,d_model:].view(1,batch.size(0), d_model),
            batch[:,:d_model].view(1,batch.size(0), d_model))
    #print(x_p.size())
    #print(x[0])

    total_loss = 0

    pos_train_edge_loader = DataLoader(torch.arange(data.train_pos_edge_index.size()[-1]),
                                    batch_size=wandb.config.batch_size, shuffle=True)


    train_neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes = data.num_nodes
    )
    neg_edge_types = torch.Tensor([get_edge_type_index([node_types[i],node_types[j]]) for i,j in zip(train_neg_edge_index[0].tolist(), train_neg_edge_index[1].tolist())])
    neg_edge_types = neg_edge_types.to(device)
    neg_train_edge_loader = DataLoader(torch.arange(train_neg_edge_index.size()[-1]),
                                   batch_size=wandb.config.batch_size, shuffle=True)

    mha_optimizer.zero_grad()

    for pos_batch_edge_idx, neg_batch_edge_idx in zip(pos_train_edge_loader, neg_train_edge_loader):
        #print(pos_batch_edge_idx.size())
        #print(data.train_pos_edge_index.size())
        #print(x_p.size())
        #print(node_types)
        #print(pos_edge_types[pos_batch_edge_idx])
        z_pos = gnn_model(x_p, node_types, train_edge_time,
                          data.train_pos_edge_index[:,pos_batch_edge_idx],
                          pos_edge_types[pos_batch_edge_idx])
        
        node_i, node_j = data.train_pos_edge_index[:,pos_batch_edge_idx][0], data.train_pos_edge_index[:,pos_batch_edge_idx][1]
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
        gnn_optimizer.zero_grad()
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()
        loss = pos_loss + neg_loss
        loss.backward(retain_graph=True)
        gnn_optimizer.step()
        
        wandb.log({"emb batch loss":loss.item()})
        total_loss += loss.item()
        pbar.update(len(pos_batch_edge_idx)/2)
    mha_optimizer.step()
    pbar.close()

    loss = total_loss / len(pos_train_edge_loader)
    wandb.log({"emb loss":loss})

    return loss

gnn_early_stopping = experiment.get_early_stopping(patience=experiment.patience, delta=0.05, verbose=True, prefix="embed")
mha_early_stopping = experiment.get_early_stopping(patience=experiment.patience, delta=0.05, verbose=True, prefix=f"mha")
best_loss = 50
for epoch in range(1, experiment.epochs):
    loss = train(epoch)
    if loss < best_loss:
        best_loss = loss
    gnn_early_stopping(loss, gnn_model)
    mha_early_stopping(loss, mha_model)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
    if gnn_early_stopping.early_stop:
        print("Early Stopping!")
        break
print(f'Best Loss: {best_loss:.4f}')
experiment.log(f'Epoch: {epoch} -> Loss (Representation Learning): {best_loss:.4f}')

mha_model.load_state_dict(torch.load(mha_early_stopping.path))
gnn_model.load_state_dict(torch.load(gnn_early_stopping.path))
data_to_transform = torch.utils.data.DataLoader(torch.cat((x,t),dim=1),
                                                batch_size=mha_batch_size)
x_p = torch.zeros(x.size()[0], gnn_config["in_dim"]).to(device)
for batch_it, batch in tqdm(enumerate(data_to_transform), total=len(data_to_transform), desc="Transforming features"):
#for batch_it, batch in enumerate(data_to_transform):
    #print(batch_it)
    f_index = batch_it*data_to_transform.batch_size
    l_index = min((batch_it+1)*data_to_transform.batch_size, x_p.size(0))
    #print(batch.size(0), l_index)
    x_p[f_index:l_index,:],_ = mha_model(
        batch[:,:d_model].view(1,batch.size(0), d_model),
        batch[:,d_model:].view(1,batch.size(0), d_model),
        batch[:,:d_model].view(1,batch.size(0), d_model))
train_pos_edge_index = data.train_pos_edge_index.to(device)
z = gnn_model(x_p, node_types, train_edge_time, train_pos_edge_index, pos_edge_types).detach()
z = z.to(device)
linkpred_config = json.load(open(experiment.model_config))[1]
linkpred_config["in_channels"] = z.size(1)*2
linkpred_model = load_model(linkpred_config).to(device)

experiment.log(f"Link Prediction Model: {linkpred_model}\n")

link_optimizer = torch.optim.Adam(linkpred_model.parameters(),
    lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)

experiment.link_prediction_step(device, linkpred_model, link_optimizer, data, z, edge_type_function, wandb)
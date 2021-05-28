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
from sklearn.preprocessing import StandardScaler
from pingumil.models import load_model
from pingumil.util.pytorchtools import EarlyStopping
from pingumil.experiments.sshet.sshet_exp import SSHetBaseExperiment
import pickle
import time

experiment = SSHetBaseExperiment(experiment_tag="secondprop", epochs=1000, timestamp=time.time())
print(experiment.output_file)
#Read all necessary data
data, atbsets_list, node_maps, node_feats, train_folds, test_folds = experiment.read_data()

#wandb stuff
wandb.init(project="fsp", group="sshet_mhafse")

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

print(node_feats.shape)
print(node_typencs.shape)

experiment.log(f"Standardization: {experiment.standardization}\n")
if experiment.standardization:
    node_feats = experiment.standardize(node_feats)

#Configuration
num_samples = [2, 2]
batch_size = 256
walk_length = 1
num_neg_samples = 1
epochs = 1000
sage_input_dim = node_feats.shape[-1]

dict_x2m = {}
#update edge_index according to node_maps
for node_map in node_maps:
    offset = len(dict_x2m)
    dict_x2m.update({k+offset:v for k,v in enumerate(node_map)})
data.edge_index[0] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[0].tolist()])
data.edge_index[1] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[1].tolist()])
data.x = node_feats
data.t = node_typencs

subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=[-1], batch_size=batch_size, shuffle=False,
    num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sage_config = json.load(open(model_config))[0]
sage_config["in_channels"] = sage_input_dim
sage_model = load_model(sage_config)
sage_model = sage_model.to(device)

n_head = 8
d_model = data.x.shape[-1]
d_k = data.x.shape[-1]
d_v = data.x.shape[-1]

mha_config = {
    "model": "mhattention",
    "n_head": n_head,
    "d_model": d_model,
    "d_k": d_k,
    "d_v": d_v
}
mha_model = load_model(mha_config)
mha_model = mha_model.to(device)

mha_optimizer = torch.optim.Adam(mha_model.parameters(), lr=1e-3)
sage_optimizer = torch.optim.Adam(sage_model.parameters(), lr=1e-3)
x = data.x.to(device)
t = data.t.to(device)

data_to_transform = torch.utils.data.DataLoader(torch.cat((x,t),dim=1), batch_size=batch_size)

#torch.autograd.set_detect_anomaly(True)

experiment.log(f"Device: {device}, visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}\n")
experiment.log(f"Type Projection: {mha_model}\n")
experiment.log(f"Graph Representation Learning Model: {sage_model}\n")

def train(epoch):
    sage_model.train()
    mha_model.train()

    pbar = tqdm(total=data.num_nodes)
    pbar.set_description(f'Epoch {epoch:02d}')

    #Type Projection Step
    #print(x)
    x_p = torch.zeros_like(x).to(device)
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
    print(x_p)
    #print(x[0])

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

    mha_optimizer.zero_grad()

    for (batch_size_, u_id, adjs_u), (_, v_id, adjs_v), (_, vn_id, adjs_vn) in\
            zip(train_loader, pos_loader, neg_loader):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs_u = [adj.to(device) for adj in adjs_u]
        z_u = sage_model(x_p[u_id], adjs_u)

        adjs_v = [adj.to(device) for adj in adjs_v]
        z_v = sage_model(x_p[v_id], adjs_v)

        adjs_vn = [adj.to(device) for adj in adjs_vn]
        z_vn = sage_model(x_p[vn_id], adjs_vn)

        sage_optimizer.zero_grad()
        pos_loss = -F.logsigmoid(
            (z_u.repeat_interleave(walk_length, dim=0)*z_v)
            .sum(dim=1)).mean()
        neg_loss = -F.logsigmoid(
            -(z_u.repeat_interleave(num_neg_samples, dim=0)*z_vn)
            .sum(dim=1)).mean()
        loss = pos_loss + neg_loss
        loss.backward(retain_graph=True)
        sage_optimizer.step()

        wandb.log({"emb batch loss":loss.item()})
        total_loss += loss.item()
        pbar.update(batch_size_)
    mha_optimizer.step()
    pbar.close()

    loss = total_loss / len(train_loader)
    wandb.log({"emb loss":loss})

    return loss

sage_early_stopping = experiment.get_early_stopping(patience=100, verbose=True, prefix="embed")
mha_early_stopping = experiment.get_early_stopping(patience=100, verbose=True, prefix=f"mha")
best_loss = 50000
for epoch in range(1, experiment.epochs):
    loss = train(epoch)
    if loss < best_loss:
        best_loss = loss
    sage_early_stopping(loss, sage_model)
    mha_early_stopping(loss, mha_model)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
    if sage_early_stopping.early_stop:
        print("Early Stopping!")
        break
print(f'Best Loss: {best_loss:.4f}')
experiment.log(f'Epoch: {epoch} -> Loss (Representation Learning): {best_loss:.4f}')

mha_model.load_state_dict(torch.load(mha_early_stopping.path))
sage_model.load_state_dict(torch.load(sage_early_stopping.path))
data_to_transform = torch.utils.data.DataLoader(torch.cat((x,t),dim=1), batch_size=batch_size)
x_p = torch.zeros_like(x).to(device)
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
z = sage_model.inference(x_p, subgraph_loader, device).detach()
z = z.to(device)
model_config = "configs/ct_sagemodel.json"
linkpred_config = json.load(open(model_config))[1]
linkpred_config["in_channels"] = z.size(1)*2
linkpred_model = load_model(linkpred_config).to(device)
experiment.log(f"Link Prediction Model: {linkpred_model}\n")

link_optimizer = torch.optim.Adam(linkpred_model.parameters(), lr=1e-3)

experiment.link_prediction_step(device, linkpred_model, link_optimizer, train_folds, z, test_folds, dict_x2m)
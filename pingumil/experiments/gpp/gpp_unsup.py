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
from torch_geometric.loader import NeighborSampler, DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
from torch_cluster import random_walk
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import remove_isolated_nodes
from pingumil.models import load_model
from pingumil.util.pytorchtools import EarlyStopping
from pingumil.experiments.gpp.gpp_exp import GPPBaseExperiment
import pickle
import time

experiment = GPPBaseExperiment(experiment_tag="semisup_fse", epochs=20, 
                               timestamp=time.time(), patience=5,
                               override_data=False)
print(experiment.output_file)
#Read all necessary data
dataset = experiment.read_data()

#wandb stuff
#wandb.init(project="gpp", group="unsup_test")

graph_ids = list(dataset.keys())

# supergraph feature/attribute set
atbs_list = []

#Get supergraph feature/attribute set
for graph_id, graph_data in dataset.items():
    for atbsets_list in graph_data["atbset_list"]:
        #Get list of all attributes in any type of node
        atbs_list = atbs_list+atbsets_list
        atbs_list = sorted(list(set(atbs_list)))

# Sort feature sets of all graphs that belong to the dataset (FSE)
for graph_id, graph_data in dataset.items():
    atbsets_list = graph_data["atbset_list"]
    atbs_maps = {atb : {} for atb in atbs_list}
    f = lambda l,v: l.index(v) if v in l else -1
    for atb in atbs_maps.keys():
        atbs_maps[atb] = {k:f(v,atb) for k,v in enumerate(atbsets_list)}
    node_feats = graph_data["node_feats"]
    for i,_ in enumerate(node_feats):
        new_node_feats = torch.zeros((node_feats[i].shape[0],len(atbs_list)*2))
        #type_encoding = []
        for atb, atb_dict in atbs_maps.items():
            atb_final_index = atbs_list.index(atb)
            #if atb_dict[i] == -1:
                #type_encoding.append(0)
            #else:
            if atb_dict[i] != -1:
                #type_encoding.append(1)
                new_node_feats[:,atb_final_index] = node_feats[i][:, atb_dict[i]]
                new_node_feats[:,len(atbs_list)+atb_final_index] = torch.ones(node_feats[i].shape[0])
        #print(type_encoding)
        node_feats[i] = new_node_feats
    dataset[graph_id]["node_feats"] = torch.cat(node_feats)

#Standardize data from all graphs
experiment.log(f"Standardization: {experiment.standardization}\n")
if experiment.standardization:
    dataset = experiment.standardize_all(dataset)

#Configuration
num_samples = [2, 2]
batch_size = 512
walk_length = 1
num_neg_samples = 1
sage_input_dim = 64

# #update edge_index according to node_maps
for graph_id, graph_data in dataset.items():
    dict_x2m = {}
    node_maps = graph_data["node_maps"]
    data = graph_data["graph"]
    node_feats = graph_data["node_feats"]
    #update edge_index according to node_maps
    for node_map in node_maps:
        offset = len(dict_x2m)
        dict_x2m.update({k+offset:v for k,v in enumerate(node_map)})
    data.edge_index[0] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[0].tolist()])
    data.edge_index[1] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[1].tolist()])
    data.x = node_feats
    dim_types = [data.x.shape[-1]]
    data.node_map = [0]
    data.class_map = graph_data["class_map"]
    graph_data["graph"] = data
    dataset[graph_id]["dict_x2m"] = dict_x2m
    dataset[graph_id]["dim_types"] = dim_types

'''subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=[-1], batch_size=batch_size, shuffle=False,
    num_workers=0)'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sage_config = json.load(open(experiment.model_config))[0]
sage_config["in_channels"] = sage_input_dim
print(sage_config)
sage_model = load_model(sage_config)
sage_model = sage_model.to(device)

typeproj_config = {
    "model": "typeprojection",
    "dim_types": dim_types,
    "dim_output": sage_config["in_channels"]
}
typeproj_model = load_model(typeproj_config)
typeproj_model = typeproj_model.to(device)

typeproj_optimizer = torch.optim.Adam(typeproj_model.parameters(), lr=1e-3)
sage_optimizer = torch.optim.Adam(sage_model.parameters(), lr=1e-3)
#x = data.x.to(device)

data_list = [graph_data["graph"] for graph_id, graph_data in dataset.items()]
loader = DataLoader(data_list, batch_size=batch_size)

#torch.autograd.set_detect_anomaly(True)
experiment.log(f"Device: {device}, visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}\n" +
    f"Type Projection: {typeproj_model}\n" +
    f"Graph Representation Learning Model: {sage_model}")

def train(epoch):
    graph_loss = 0
    i = 1
    for batch in loader:
        data = batch
        print(data)
        if data.edge_index.size()[-1] < batch_size:
            continue
        x = data.x.to(device)

        sage_model.train()
        typeproj_model.train()

        pbar = tqdm(total=data.num_nodes)
        pbar.set_description(f'Epoch {epoch:02d}')

        #Type Projection Step
        #print(x)
        x_p = typeproj_model([x], [0])
        #print(x_p)
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
            sizes=num_samples, batch_size=batch_size * walk_length,
            shuffle=False, num_workers=0)

        # negative sampling as node2vec
        deg = degree(data.edge_index[0])
        distribution = deg ** 0.75
        neg_idx = torch.multinomial(
            distribution, data.num_nodes * num_neg_samples, replacement=True)
        neg_loader = NeighborSampler(
            data.edge_index, node_idx=neg_idx,
            sizes=num_samples, batch_size=batch_size * num_neg_samples,
            shuffle=True, num_workers=0)

        typeproj_optimizer.zero_grad()

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

            total_loss += loss.item()
            pbar.update(batch_size_)
        typeproj_optimizer.step()
        pbar.close()

        loss = total_loss / len(train_loader)
        graph_loss = graph_loss + loss
        i = i+1
    return graph_loss/i

sage_early_stopping = experiment.get_early_stopping(patience=2, verbose=True,
                                                    prefix="embed")
proj_early_stopping = experiment.get_early_stopping(patience=2, verbose=True,
                                                    prefix=f"proj")
best_loss = 50
for epoch in range(1, experiment.epochs):
    loss = train(epoch)
    if loss < best_loss:
        best_loss = loss
    sage_early_stopping(loss, sage_model)
    proj_early_stopping(loss, typeproj_model)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
    if sage_early_stopping.early_stop:
        print("Early Stopping!")
        break
print(f'Best Loss: {best_loss:.4f}')
experiment.log(f'Epoch: {epoch} -> Loss (Representation Learning): {best_loss:.4f}')


typeproj_model.load_state_dict(torch.load(proj_early_stopping.path))
sage_model.load_state_dict(torch.load(sage_early_stopping.path))

x_multi = []
y_multi = []
for graph_id, graph_entry in dataset.items():
    graph_data = graph_entry["graph"]
    if None in graph_data.class_map.values():
        continue
    data = graph_data.to(device)
    x_p = typeproj_model([data.x], [0])
    subgraph_loader = NeighborSampler(
        data.edge_index, node_idx=None,
        sizes=[-1], batch_size=batch_size, shuffle=False,
        num_workers=0)
    z = sage_model.inference(x_p, subgraph_loader, device).detach()
    z = z.to(device)
    graph_dx2m = graph_entry["dict_x2m"]
    x_multi = x_multi + [z[graph_dx2m[int(x)]] for x in graph_data.class_map.keys()]
    y_multi = y_multi + [torch.Tensor(list(graph_data.class_map.values()))[0]]

x_multi = torch.stack(x_multi)
print(x_multi.size())
y_multi = torch.stack(y_multi)
print(y_multi.size())

assert x_multi.size(0) == y_multi.size(0)

multilabel_config = json.load(open(experiment.model_config))[1]
multilabel_config["in_channels"] = x_multi.size(1)
multilabel_config["out_channels"] = y_multi.size(1)
multilabel_model = load_model(multilabel_config).to(device)

experiment.log(f"Multilabel Classification Model: {multilabel_model}\n")

experiment.multilabel_classification_step(device, multilabel_model,
                                          x_multi, y_multi)
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
from torch_geometric.utils import negative_sampling
from torch_cluster import random_walk
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import remove_isolated_nodes
from pingumil.models import load_model
from pingumil.util.pytorchtools import EarlyStopping
from pingumil.experiments.gpp.gpp_exp import GPPBaseExperiment, EPS
import wandb
import time

#wandb stuff
run = wandb.init(project="fsp", group="gpp_fse",
                 mode="disabled",
                 config={
                    "batch_size" : 512,
                    "lr": 1e-3,
                    "lr_clf": 1e-5,
                    "weight_decay": 1e-5,
                    "gnn": "hgt"
                 })

experiment = GPPBaseExperiment(experiment_tag="semisup_fse",
                               epochs=1000, 
                               mlc_epochs=200,
                               timestamp=time.time(),
                               model_config="configs/gpp_hgtmodel.json",
                               patience=10,
                               wandb=wandb,
                               override_data=False)

print(experiment.output_file)
#Read all necessary data
dataset = experiment.read_data()

graph_ids = list(dataset.keys())
# supergraph feature/attribute set
atbs_list = []
atbsets_dict = {}

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
num_samples = [-1,-1]
walk_length = 1
batch_size = wandb.config.batch_size
num_neg_samples = 1
epochs = experiment.epochs
gnn_input_dim = 128

def find_nodemap(fnid, node_maps):
    offset = 0
    for i,v in enumerate(node_maps):
        if fnid < offset + len(v):
            return i
        offset += len(v)

all_node_types=[]
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
    num_nodes = len(node_feats)
    node_types = [find_nodemap(x, node_maps) for x in range(num_nodes)]
    data.node_types = torch.Tensor(node_types)
    all_node_types = list(set(all_node_types + node_types))
    data.class_map = graph_data["class_map"]
    graph_data["graph"] = data
    dataset[graph_id]["dict_x2m"] = dict_x2m
    dataset[graph_id]["dim_types"] = dim_types

edge_type_dict = {}
num_edge_types = 0
for i in range(len(all_node_types)):
    edge_type_dict[i] = {}
    for j in range(i,len(all_node_types)):
        edge_type_dict[i][j] = num_edge_types
        num_edge_types += 1

def get_edge_type_index(fnids):
    #print(fnids)
    node_label_ids = fnids
    node_label_ids.sort()
    i, j = node_label_ids[0], node_label_ids[1]
    return edge_type_dict[i][j]

'''subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=[-1], batch_size=batch_size, shuffle=False,
    num_workers=0)'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gnn_config = json.load(open(experiment.model_config))[0]
gnn_config["in_dim"] = 128
gnn_config["num_types"] = 1
gnn_config["num_relations"] = num_edge_types
gnn_model = load_model(gnn_config)
print(gnn_model)
gnn_model = gnn_model.to(device)

typeproj_config = {
    "model": "typeprojection",
    "dim_types": dim_types,
    "dim_output": gnn_config["in_dim"]
}
typeproj_model = load_model(typeproj_config)
typeproj_model = typeproj_model.to(device)

typeproj_optimizer = torch.optim.Adam(typeproj_model.parameters(),
                                      lr=wandb.config.lr)
gnn_optimizer = torch.optim.Adam(gnn_model.parameters(),
                                 lr=wandb.config.lr,
                                 weight_decay=wandb.config.weight_decay)
#x = data.x.to(device)

#data_list = [graph_data["graph"] for graph_id, graph_data in dataset.items()]
#loader = DataLoader(data_list, batch_size=batch_size)

#torch.autograd.set_detect_anomaly(True)
experiment.log(f"Device: {device}, visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}\n")
experiment.log(f"Type Projection: {typeproj_model}\n")
experiment.log(f"Graph Representation Learning Model: {gnn_model}\n")

def train(epoch):
    graph_loss = 0
    i = 1
    pbar = tqdm(total=len(dataset))
    pbar.set_description(f'Epoch {epoch:02d}')
    for _, graph_data in dataset.items():
        data = graph_data["graph"]
        if data.edge_index.size()[-1] < batch_size:
            continue
        x = data.x.to(device)
        node_types = data.node_types.to(device)
        
        gnn_model.train()
        typeproj_model.train()

        #Type Projection Step
        #print(x)
        x_p = typeproj_model([x], [0])
        #print(x_p)
        #print(x[0])
        total_loss = 0
        
        edges_as_list = zip(data.edge_index[0].tolist(),
                   data.edge_index[1].tolist())

        pos_edge_types = torch.Tensor(
            [get_edge_type_index([int(node_types[i]),int(node_types[j])]) for i,j in edges_as_list])
        pos_edge_types = pos_edge_types.to(device)
        
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
            [get_edge_type_index([int(node_types[i]),int(node_types[j])]) for i,j in neg_edges_as_list])
        neg_edge_types = neg_edge_types.to(device)
        neg_train_edge_loader = DataLoader(torch.arange(train_neg_edge_index.size()[-1]),
                                    batch_size=wandb.config.batch_size, shuffle=True)
        zero_node_types = torch.zeros_like(torch.Tensor(node_types)).to(device)

        typeproj_optimizer.zero_grad()

        for pos_batch_edge_idx, neg_batch_edge_idx in zip(pos_train_edge_loader, neg_train_edge_loader):
            train_edge_time = torch.Tensor([0]*data.edge_index.size(-1))
            train_edge_time = train_edge_time.to(device)
            
            z_pos = gnn_model(x_p, zero_node_types, train_edge_time,
                            data.edge_index[:,pos_batch_edge_idx].to(device),
                            pos_edge_types[pos_batch_edge_idx]
                            )
            
            node_i, node_j = data.edge_index[:,pos_batch_edge_idx][0], data.edge_index[:,pos_batch_edge_idx][1]
            #print(node_i.size())
            #print(node_j.size())
            out = (z_pos[node_i] * z_pos[node_j]).sum(dim=-1).view(-1)
            pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()
            
            z_neg = gnn_model(x_p, node_types, train_edge_time,
                            train_neg_edge_index[:,neg_batch_edge_idx].to(device),
                            neg_edge_types[neg_batch_edge_idx].to(device))
            node_i, node_j = train_neg_edge_index[:,neg_batch_edge_idx][0], train_neg_edge_index[:,neg_batch_edge_idx][1]
            #print(node_i.size())
            #print(node_j.size())
            out = (z_neg[node_i] * z_neg[node_j]).sum(dim=-1).view(-1)
            neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()
            #print(node_i.size())
            #print(node_j.size())
            out = (z_neg[node_i] * z_neg[node_j]).sum(dim=-1).view(-1)
            neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()
            gnn_optimizer.zero_grad()
            loss = pos_loss + neg_loss
            loss.backward(retain_graph=True)
            gnn_optimizer.step()

            wandb.log({"emb batch loss":loss.item()})
            total_loss += loss.item()
        pbar.update(1)
        typeproj_optimizer.step()

        loss = total_loss / len(pos_train_edge_loader)
        wandb.log({"emb loss":loss})

        graph_loss = graph_loss + loss
        i = i+1
    pbar.close()
    return graph_loss/i

gnn_early_stopping = experiment.get_early_stopping(verbose=True,
                                                   prefix="embed")
proj_early_stopping = experiment.get_early_stopping(verbose=True,
                                                   prefix=f"proj")
best_loss = np.inf
for epoch in range(1, experiment.epochs):
    loss = train(epoch)
    wandb.log({"train loss":loss})
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

x_multi = []
y_multi = []
for graph_id, graph_entry in dataset.items():
    graph_data = graph_entry["graph"]
    if None in graph_data.class_map.values():
        continue
    data = graph_data.to(device)
    node_types = data.node_types.to(device)
    x_p = typeproj_model([data.x], [0])
    train_pos_edge_index = data.edge_index.to(device)
    edges_as_list = zip(data.edge_index[0].tolist(),
                   data.edge_index[1].tolist())
    pos_edge_types = torch.Tensor(
        [get_edge_type_index([int(node_types[i]), int(node_types[j])]) for i,j in edges_as_list])
    pos_edge_types = pos_edge_types.to(device)
    zero_node_types = torch.zeros_like(torch.Tensor(node_types)).to(device)
    train_edge_time = torch.Tensor([0]*data.edge_index.size(-1))
    train_edge_time = train_edge_time.to(device)
    z = gnn_model(x_p, zero_node_types, train_edge_time, train_pos_edge_index, pos_edge_types).detach()
    z = z.to(device)
    graph_dx2m = graph_entry["dict_x2m"]
    x_multi = x_multi + [z[graph_dx2m[int(x)]] for x in graph_data.class_map.keys()]
    y_multi = y_multi + [torch.Tensor(list(graph_data.class_map.values()))[0]]

x_multi = torch.stack(x_multi)
print(x_multi.size())
y_multi = torch.stack(y_multi)
print(y_multi.size())

assert x_multi.size(0) == y_multi.size(0)

y_bartle = y_multi[:,:4]
y_dedica = y_multi[:,4:]

assert y_bartle.size(1) == 4
assert y_dedica.size(1) == 2

for label, y_gt in [("bartle", y_bartle), ("dedication", y_dedica)]:
    multilabel_config = json.load(open(experiment.model_config))[1]
    multilabel_config["in_channels"] = x_multi.size(1)
    multilabel_config["out_channels"] = y_gt.size(1)
    multilabel_model = load_model(multilabel_config).to(device)

    experiment.log(f"Multilabel Classification Model: {multilabel_model}_{label}\n")

    experiment.multilabel_classification_step(device, multilabel_model,
                                            x_multi, y_gt, list(dataset.keys()),
                                            wandb, suffix=label)

experiment.get_quality_results()
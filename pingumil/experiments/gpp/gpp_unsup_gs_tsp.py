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
import wandb
import time

#wandb stuff
run = wandb.init(project="fsp", group="gpp_tsp",
                 config={
                    "batch_size" : 512,
                    "lr": 1e-3,
                    "lr_clf": 1e-5,
                    "weight_decay": 1e-5,
                    "gnn": "gs"
                 })

experiment = GPPBaseExperiment(experiment_tag="semisup_tsp",
                               epochs=1000,
                               mlc_epochs=200,
                               timestamp=time.time(),
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
all_dim_types = []

#Get supergraph feature/attribute set, create a dictionary from attribute set to a identifier,
# and add the attribute set identifier in a adjacent graph_data["node_types"] structure
for graph_id, graph_data in dataset.items():
    for i, atbsets_list in enumerate(graph_data["atbset_list"]):
        #Get list of all attributes in any type of node
        atbs_list = atbs_list+atbsets_list
        atbs_list = sorted(list(set(atbs_list)))
        if tuple(atbsets_list) not in atbsets_dict:
            atbsets_dict[tuple(atbsets_list)] = len(atbsets_dict)
            assert graph_data["node_feats"][i].size()[1] == len(atbsets_list)
            all_dim_types.append(len(atbsets_list))
        current_node_type = atbsets_dict[tuple(atbsets_list)]
        if "node_types" not in dataset[graph_id]:
            dataset[graph_id]["node_types"] = []
        if current_node_type not in dataset[graph_id]["node_types"]:
            dataset[graph_id]["node_types"].append(current_node_type)

#Standardize data from all graphs
experiment.log(f"Standardization: {experiment.standardization}\n")
if experiment.standardization:
    scalers = {}
    # Fit an scaler for each feature/attribute
    for graph_id, graph_data in dataset.items():
        atbsets_list = graph_data["atbset_list"]
        atbs_maps = {atb : {} for atb in atbs_list}
        f = lambda l,v: l.index(v) if v in l else -1
        for atb in atbs_maps.keys():
            atbs_maps[atb] = {k:f(v,atb) for k,v in enumerate(atbsets_list)}
        node_feats = graph_data["node_feats"]
        #For each attribute, apply normalization on node_feats according to the mapping
        for atb, atb_map in atbs_maps.items():
            if atb not in scalers:
                scalers[atb] = StandardScaler()
            node2tuple = tuple([node_feats[k][:,v] for k,v in atb_map.items() if v != -1])
            if len(node2tuple) == 0:
                continue
            atbdata = torch.cat(node2tuple)
            #print(atbdata)
            if ((torch.min(atbdata).item() == 0 and torch.max(atbdata).item() == 1) or torch.equal(atbdata, torch.zeros_like(atbdata))
                    or torch.equal(atbdata, torch.ones_like(atbdata))):
                #print(f"Continuning for {atb}, possible One-Hot-Encoded")
                continue
            atbdata_t = atbdata.reshape(-1,1)
            scalers[atb] = scalers[atb].partial_fit(atbdata_t)
            
    #Now transform data from all graphs
    for graph_id, graph_data in dataset.items():
        atbsets_list = graph_data["atbset_list"]
        atbs_maps = {atb : {} for atb in atbs_list}
        f = lambda l,v: l.index(v) if v in l else -1
        for atb in atbs_maps.keys():
            atbs_maps[atb] = {k:f(v,atb) for k,v in enumerate(atbsets_list)}
        node_feats = graph_data["node_feats"]
        #For each attribute, apply normalization on node_feats according to the mapping
        for atb, atb_map in atbs_maps.items():
            if atb not in scalers:
                scalers[atb] = StandardScaler()
            node2tuple = tuple([node_feats[k][:,v] for k,v in atb_map.items() if v != -1])
            if len(node2tuple) == 0:
                continue
            atbdata = torch.cat(node2tuple)
            #print(atbdata)
            if ((torch.min(atbdata).item() == 0 and torch.max(atbdata).item() == 1) or torch.equal(atbdata, torch.zeros_like(atbdata))
                    or torch.equal(atbdata, torch.ones_like(atbdata))):
                #print(f"Continuning for {atb}, possible One-Hot-Encoded")
                continue
            split_dim = [node_feats[k][:,v].shape[0] for k,v in atb_map.items() if v!=-1]
            atbdata_t = atbdata.reshape(-1,1)
            atbdata_std = torch.from_numpy(scalers[atb].fit_transform(atbdata_t).reshape(1,-1))
            split_atbdata_std = torch.split(atbdata_std[0], split_dim)
            i = 0
            for k,v in atb_map.items():
                if v == -1:
                    continue
                node_feats[k][:,v] = split_atbdata_std[i]
                i = i + 1
        dataset[graph_id]["node_feats"] = node_feats
    
#Configuration
num_samples = [2, 2]
batch_size = wandb.config.batch_size
walk_length = 1
num_neg_samples = 1
gnn_input_dim = 64

for graph_id, graph_data in dataset.items():
    data = graph_data["graph"]
    node_feats = graph_data["node_feats"]
    node_types = graph_data["node_types"]
    data.x_ts = node_feats
    data.t_ts = node_types
    data.class_map = graph_data["class_map"]
    graph_data["graph"] = data

'''subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=[-1], batch_size=batch_size, shuffle=False,
    num_workers=0)'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sage_config = json.load(open(experiment.model_config))[0]
sage_config["in_channels"] = gnn_input_dim
print(sage_config)
sage_model = load_model(sage_config)
sage_model = sage_model.to(device)

typeproj_config = {
    "model": "typeprojection",
    "dim_types": all_dim_types,
    "dim_output": sage_config["in_channels"]
}
typeproj_model = load_model(typeproj_config)
typeproj_model = typeproj_model.to(device)

typeproj_optimizer = torch.optim.Adam(typeproj_model.parameters(),
                                      lr=wandb.config.lr)
sage_optimizer = torch.optim.Adam(sage_model.parameters(),
                                  lr=wandb.config.lr)
#x = data.x.to(device)

#data_list = [graph_data["graph"] for graph_id, graph_data in dataset.items()]
#loader = DataLoader(data_list, batch_size=batch_size)

#torch.autograd.set_detect_anomaly(True)
experiment.log(f"Device: {device}, visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}\n" +
    f"Type Projection: {typeproj_model}\n" +
    f"Graph Representation Learning Model: {sage_model}")

def train(epoch):
    graph_loss = 0
    i = 1
    for _, graph_data in dataset.items():
        data = graph_data["graph"]
        if data.edge_index.size()[-1] < batch_size:
            continue
        x_ts = [x_t.to(device) for x_t in data.x_ts]

        sage_model.train()
        typeproj_model.train()

        pbar = tqdm(total=data.num_nodes)
        pbar.set_description(f'Epoch {epoch:02d}')

        #Type Projection Step
        #print(x)
        x_p = typeproj_model(x_ts, data.t_ts)
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
            
            wandb.log({"emb batch loss":loss.item()})

            total_loss += loss.item()
            pbar.update(batch_size_)
        typeproj_optimizer.step()
        pbar.close()

        loss = total_loss / len(train_loader)
        wandb.log({"emb loss":loss})
        
        graph_loss = graph_loss + loss
        i = i+1
    return graph_loss/i

sage_early_stopping = experiment.get_early_stopping(verbose=True,
                                                    prefix="embed")
proj_early_stopping = experiment.get_early_stopping(verbose=True,
                                                    prefix=f"proj")
best_loss = np.inf
for epoch in range(1, experiment.epochs):
    loss = train(epoch)
    wandb.log({"train loss":loss})
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
    x_p = typeproj_model(data.x_ts, data.t_ts)
    subgraph_loader = NeighborSampler(
        data.edge_index, node_idx=None,
        sizes=[-1], batch_size=batch_size, shuffle=False,
        num_workers=0)
    z = sage_model.inference(x_p, subgraph_loader, device).detach()
    z = z.to(device)
    x_multi = x_multi + [z[int(x)] for x in graph_data.class_map.keys()]
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
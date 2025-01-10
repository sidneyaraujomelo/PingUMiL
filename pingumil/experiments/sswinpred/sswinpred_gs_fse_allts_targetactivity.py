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
from pingumil.experiments.sswinpred.sswinpred_exp import SSWinPredBaseExperiment
import wandb
import time

current_timestamp = 20
project_name = "winpred_binary_activity"
group_name = "gs_macro_high_cls_epochs_allts_t8v1t1_valonf1"
wandb_config = {
    "batch_size" : 1024,
    "lr": 1e-3,
    "lr_clf" : 1e-3,
    "weight_decay": 1e-4,
    "weight_decay_clf": 1e-4,
    "gnn": "gs",
    "current_timestamp" : current_timestamp
}

#wandb stuff
run = wandb.init(project=project_name, group=group_name,
                 config=wandb_config)

experiment = SSWinPredBaseExperiment(
    dataset_folder="/raid/home/smelo/PingUMiL-pytorch/dataset/SmokeSquadron/ss_wp_playeractivity/preprocessed_graphs",
    experiment_tag=project_name+"_"+group_name,
    epochs=1000,
    cls_epochs=1000,
    timestamp=time.time(),
    model_config="configs/sshitpred_sagemodel_v4.json",
    patience=100,
    wandb=wandb,
    override_data=False)
print(experiment.output_file)
#Read all necessary data
dataset = experiment.read_data()

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
    
# Reset the value of "date" of each node in each graph so that they all start at 0 seconds
# We also get the max date across all matches
max_timestamp = 0
for graph_id, graph_data in dataset.items():
    dataset[graph_id]["graph"].date = dataset[graph_id]["graph"].date - dataset[graph_id]["graph"].date[0]
    if max(dataset[graph_id]["graph"].date) > max_timestamp:
        print(graph_id, max(dataset[graph_id]["graph"].date))
        max_timestamp = max(dataset[graph_id]["graph"].date)
print(max_timestamp)

#Standardize data from all graphs
experiment.log(f"Standardization: {experiment.standardization}\n")
if experiment.standardization:
    dataset = experiment.standardize_all(dataset)
    
#Configuration
num_samples = [2, 2]
batch_size = wandb.config.batch_size
walk_length = 1
num_neg_samples = 1
sage_input_dim = 64

#update edge_index according to node_maps
for graph_id, graph_data in dataset.items():
    #maps index of the node in its node type mapping to its position in node_feats, which follows original graph indexing
    dict_m2x = {}
    node_maps = graph_data["node_maps"]
    data = graph_data["graph"]
    node_feats = graph_data["node_feats"]
    #update edge_index according to node_maps
    for node_map in node_maps:
        offset = len(dict_m2x)
        dict_m2x.update({v:k+offset for k, v in enumerate(node_map)})
    # at the end, edge_index is now sorted according to node_maps
    data.edge_index[0] = torch.LongTensor([dict_m2x[idx] for idx in data.edge_index[0].tolist()])
    data.edge_index[1] = torch.LongTensor([dict_m2x[idx] for idx in data.edge_index[1].tolist()])
    data.x = node_feats
    dim_types = [data.x.shape[-1]]
    data.node_map = [0]
    #data.class_map = graph_data["class_map"]
    graph_data["graph"] = data
    dataset[graph_id]["dict_m2x"] = dict_m2x
    dataset[graph_id]["dim_types"] = dim_types

#Setting model configuration
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

typeproj_optimizer = torch.optim.Adam(typeproj_model.parameters(),
                                      lr=wandb.config.lr)
sage_optimizer = torch.optim.Adam(sage_model.parameters(),
                                  lr=wandb.config.lr)

def train(epoch,
          loader,
          sage_model,
          typeproj_model,
          sage_optimizer,
          typeproj_optimizer):
    graph_loss = 0
    i = 1
    for batch in loader:
        data = batch
        
        #if data.filtered_edge_index.size()[-1] < batch_size:
        #    continue
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
            data.filtered_edge_index, node_idx=node_idx,
            sizes=num_samples, batch_size=batch_size, shuffle=False,
            num_workers=0)
        
        rw = random_walk(
            data.filtered_edge_index[0], data.filtered_edge_index[1],
            node_idx, walk_length=walk_length)
        rw_idx = rw[:, 1:].flatten()
        pos_loader = NeighborSampler(
            data.filtered_edge_index, node_idx=rw_idx,
            sizes=num_samples, batch_size=batch_size * walk_length,
            shuffle=False, num_workers=0)

        # negative sampling as node2vec
        deg = degree(data.filtered_edge_index[0])
        distribution = deg ** 0.75
        neg_idx = torch.multinomial(
            distribution, data.num_nodes * num_neg_samples, replacement=True)
        neg_loader = NeighborSampler(
            data.filtered_edge_index, node_idx=neg_idx,
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

#Now, let's try to get some "date"-induced subgraphs
interval = 5

current_ts_node_idx = {}
#data_list contains all the graph data to be sent to the DataLoader
data_list = []
for graph_id, graph_data in dataset.items():
    node_dates = graph_data["graph"].date
    current_ts_node_idx = torch.nonzero(node_dates < max_timestamp).squeeze()
    current_ts_nodes = [graph_data["dict_m2x"][x] for x in current_ts_node_idx.tolist()]
    # Convert the list of nodes to a set for faster membership checking
    node_set = set(current_ts_nodes)

    # Find indices where both source and target nodes are in node_list
    edge_index = graph_data["graph"].edge_index
    filtered_indices = [i for i in range(edge_index.shape[1]) if edge_index[0, i].item() in node_set and 
                        edge_index[1, i].item() in node_set]
    # Convert the list of filtered indices to a PyTorch tensor
    filtered_edge_index = edge_index[:, filtered_indices]
    graph_data["graph"].filtered_edge_index = filtered_edge_index.cpu()

#create the data object to be added to data_list
data_list = [graph_data["graph"] for _, graph_data in dataset.items()]

#Prepare training routine
loader = DataLoader(data_list, batch_size=batch_size)

typeproj_model.reset_parameters()
typeproj_optimizer = torch.optim.Adam(typeproj_model.parameters(),
                                    lr=wandb.config.lr)
sage_model.reset_parameters()
sage_optimizer = torch.optim.Adam(sage_model.parameters(),
                                lr=wandb.config.lr)

sage_early_stopping = experiment.get_early_stopping(verbose=True,
                                                    prefix="embed")
proj_early_stopping = experiment.get_early_stopping(verbose=True,
                                                    prefix=f"proj")

# training routine
best_loss = np.inf
for epoch in range(1, experiment.epochs):
    loss = train(epoch,
                    loader,
                    sage_model,
                    typeproj_model,
                    sage_optimizer,
                    typeproj_optimizer)
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

while current_timestamp < max_timestamp:
    x_multi = []
    y_multi = []
    groups = []
    for graph_id, graph_entry in dataset.items():
        graph_data = graph_entry["graph"]
        graph_dm2x = graph_entry["dict_m2x"]
        node_dates = graph_data.date
        current_ts_node_idx = torch.nonzero(node_dates < current_timestamp).squeeze()
        current_ts_nodes = [graph_dm2x[x] for x in current_ts_node_idx.tolist()]
        # Convert the list of nodes to a set for faster membership checking
        node_set = set(current_ts_nodes)
        if None in graph_entry["class_map"].values():
            continue
        data = graph_data.to(device)
        x_p = typeproj_model([data.x], [0])
        subgraph_loader = NeighborSampler(
            data.edge_index, node_idx=None,
            sizes=[-1], batch_size=batch_size, shuffle=False,
            num_workers=0)
        z = sage_model.inference(x_p, subgraph_loader, device).detach()
        z = z.to(device)
        # Get the target player nodes for Player01 and Player02
        target_node_idxs = list(graph_entry["class_map"].keys())
        target_node_idxs = [graph_dm2x[int(x)] for x in target_node_idxs]
        target_node_idxs = [x for x in target_node_idxs if x in node_set]
        assert len(target_node_idxs) > 1
        
        player_node_idxs = []
        # Get player node Player01
        target_p1_node_idxs = [(x, graph_data.date[x]) for x in target_node_idxs if graph_data.ObjectTag[x] == "Player01"]
        target_p1_node_idxs.sort(key=lambda x: x[1], reverse=True)
        target_p1_node_idx = target_p1_node_idxs[0][0]
        player_node_idxs.append(target_p1_node_idx)
        
        # Get player node Player02
        target_p2_node_idxs = [(x, graph_data.date[x]) for x in target_node_idxs if graph_data.ObjectTag[x] == "Player02"]
        target_p2_node_idxs.sort(key=lambda x: x[1], reverse=True)
        target_p2_node_idx = target_p2_node_idxs[0][0]
        player_node_idxs.append(target_p2_node_idx)
        
        experiment.log(f"Ts: {current_timestamp}, Graph: {graph_id}, target node ids: {player_node_idxs}")
        
        player_node_zs = [z[x] for x in player_node_idxs]
        x_multi = x_multi + [torch.cat(player_node_zs)]
        y_multi.append(list(graph_entry["class_map"].values())[0][0])

    x_multi = torch.stack(x_multi)
    print(x_multi.size())
    y_multi = torch.Tensor(y_multi)
    print(y_multi.size())
    
    assert x_multi.size(0) == y_multi.size(0)
    
    classification_config = json.load(open(experiment.model_config))[1]
    classification_config["in_channels"] = x_multi.size(1)
    classification_model = load_model(classification_config).to(device)

    experiment.log(f"Classification Model: {classification_model}\n")

    experiment.classification_step_tvt(device, classification_model,
                                   x_multi, y_multi, list(dataset.keys()),
                                   current_timestamp, wandb)
    # Increment timestamp
    current_timestamp = current_timestamp + interval
    
    print(f"Current timestamp {current_timestamp}")
    
    run.finish()
    
    wandb_config["current_timestamp"] = current_timestamp
    run = run = wandb.init(project=project_name, group=group_name,
                 config=wandb_config)
    experiment.reset(wandb)
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

wandb.init(project="fsp", group="donors_mhafsef",
                 config={
                    "batch_size" : 512,
                    "lr" : 1e-3,
                    "weight_decay" : 0,
                    "gnn" : "graphsage"
                 })
experiment = DonorsBaseExperiment(
    experiment_tag="linkpred_mhafsef",
    epochs=1000,
    patience=30,
    model_config="configs/ct_sagemodel.json",
    wandb=wandb,
    timestamp=time.time(),
    split_edges=False,
    save_split=False)
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

print(node_feats.shape)
print(node_typencs.shape)

experiment.log(f"Standardization: {experiment.standardization}\n")
if experiment.standardization:
    node_feats = experiment.standardize(node_feats)
x = torch.cat((node_feats,node_typencs),dim=1)

#Configuration
num_samples = [2, 2]
batch_size = wandb.config.batch_size
walk_length = 1
num_neg_samples = 1
epochs = experiment.epochs
gnn_input_dim = x.shape[-1]

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
data.x = x

subgraph_loader = NeighborSampler(
    data.train_pos_edge_index, node_idx=None,
    sizes=[-1], batch_size=batch_size, shuffle=False,
    num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gnn_config = json.load(open(experiment.model_config))[0]
gnn_config["in_channels"] = gnn_input_dim
gnn_model = load_model(gnn_config)
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
    "d_v": d_v
}
mha_model = load_model(mha_config)
mha_model = mha_model.to(device)

mha_optimizer = torch.optim.Adam(mha_model.parameters(), lr=wandb.config.lr)
gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=wandb.config.lr)
x = data.x.to(device)

data_to_transform = torch.utils.data.DataLoader(x, batch_size=batch_size)

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
    x_p = torch.zeros_like(x).to(device)
    for batch_it, batch in tqdm(enumerate(data_to_transform), total=len(data_to_transform), desc="Transforming features"):
    #for batch_it, batch in enumerate(data_to_transform):
        #print(batch_it)
        f_index = batch_it*data_to_transform.batch_size
        l_index = min((batch_it+1)*data_to_transform.batch_size, x_p.size(0))
        #print(batch.size(0), l_index)
        x_p[f_index:l_index,:],_ = mha_model(
            batch.view(1,batch.size(0), d_model),
            batch.view(1,batch.size(0), d_model),
            batch.view(1,batch.size(0), d_model))
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

    mha_optimizer.zero_grad()

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
    mha_optimizer.step()
    pbar.close()

    loss = total_loss / len(train_loader)
    wandb.log({"emb loss":loss})

    return loss

gnn_early_stopping = experiment.get_early_stopping(patience=20, verbose=True, prefix="embed")
mha_early_stopping = experiment.get_early_stopping(patience=20, verbose=True, prefix=f"mha")
best_loss = 50000
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
data_to_transform = torch.utils.data.DataLoader(x, batch_size=batch_size)
x_p = torch.zeros_like(x).to(device)
for batch_it, batch in tqdm(enumerate(data_to_transform), total=len(data_to_transform), desc="Transforming features"):
#for batch_it, batch in enumerate(data_to_transform):
    #print(batch_it)
    f_index = batch_it*data_to_transform.batch_size
    l_index = min((batch_it+1)*data_to_transform.batch_size, x_p.size(0))
    #print(batch.size(0), l_index)
    x_p[f_index:l_index,:],_ = mha_model(
        batch.view(1,batch.size(0), d_model),
        batch.view(1,batch.size(0), d_model),
        batch.view(1,batch.size(0), d_model))
z = gnn_model.inference(x_p, subgraph_loader, device).detach()
z = z.to(device)

linkpred_config = json.load(open(experiment.model_config))[1]
linkpred_config["in_channels"] = z.size(1)*2
linkpred_model = load_model(linkpred_config).to(device)
experiment.log(f"Link Prediction Model: {linkpred_model}\n")

link_optimizer = torch.optim.Adam(linkpred_model.parameters(), lr=1e-3)
experiment.link_prediction_step(device, linkpred_model, link_optimizer, data, z, edge_type_function, wandb)
"""
early_stopping = experiment.get_early_stopping(patience=100, verbose=True, prefix=f"predictor")

best_metrics = {
    "train_loss": None,
    "p": None,
    "r": None
}
#average_metrics = { k : [] for k,v in best_metrics.items() }

for epoch in range(1,epochs):
    linkpred_model.train()
    link_optimizer.zero_grad()
    train_neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes = data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    train_edges = torch.cat([data.train_pos_edge_index, train_neg_edge_index], dim=-1).to(device)
    train_labels = torch.zeros(data.train_pos_edge_index.size(1)+train_neg_edge_index.size(1), dtype=torch.float, device=device)
    train_labels[:data.train_pos_edge_index.size(1)] = 1.
    #train_edge_data = Data(x=z, edge_index=train_edges, y=train_labels.flatten())
    train_edge_loader = DataLoader(torch.arange(train_edges.size()[-1]), batch_size=32, shuffle=True)
    print(f"{train_edges.size()} {train_labels.size()}")
    total_train_loss = 0
    total_train_correct_results = 0
    for batch_edge_idx in train_edge_loader:
        #print(batch_edge_idx)
        out = linkpred_model(z, train_edges[:,batch_edge_idx]).flatten()
        train_loss = F.binary_cross_entropy_with_logits(out, train_labels[batch_edge_idx])
        y_pred_tag = torch.round(torch.sigmoid(out))    
        #print(y_pred_tag.size())
        #print(train_labels[batch_edge_idx].size())

        corrects_results_sum = torch.eq(y_pred_tag, train_labels[batch_edge_idx])
        total_train_correct_results += corrects_results_sum.sum().item()
    
        train_loss.backward()
        link_optimizer.step()
        total_train_loss += train_loss.item()
        wandb.log({"linkpred train batch loss":train_loss.item()})
    linkpred_loss = total_train_loss/len(train_edge_loader)
    train_acc = float(total_train_correct_results)/train_labels.size()[0]
    
    wandb.log({
        "linkpred train loss":linkpred_loss,
        "train acc":train_acc
    })

    linkpred_model.eval()

    test_edges = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=-1).to(device)
    test_labels = torch.zeros(data.test_pos_edge_index.size(1)+data.test_neg_edge_index.size(1), dtype=torch.float, device=device)
    test_labels[:data.test_pos_edge_index.size(1)] = 1.
    print(test_edges.size())
    #test_edge_data = Data(x=z, edge_index=test_edges, y=test_labels)
    test_edge_loader = DataLoader(torch.arange(test_edges.size()[-1]), batch_size=32)
    
    total_test_loss = 0
    total_test_correct_results = 0
    total_out_pred = None
    for batch_edge_idx in test_edge_loader:
        out_hat = linkpred_model(z, test_edges[:,batch_edge_idx]).flatten()
        
        test_loss = F.binary_cross_entropy_with_logits(out_hat, test_labels[batch_edge_idx])
        total_test_loss += test_loss.item()

        out_pred = torch.round(torch.sigmoid(out_hat)).detach()
        if total_out_pred != None:
            total_out_pred = torch.cat((total_out_pred, out_pred), dim=-1)
        else:
            total_out_pred = out_pred
        
        total_test_correct_results += torch.eq(out_pred, test_labels[batch_edge_idx]).sum().item()

        wandb.log({"linkpred test batch loss":test_loss.item()})

    test_acc =  float(total_test_correct_results) / test_labels.size()[0]

    wandb.log({
        "linkpred test loss":total_test_loss/len(test_edge_loader),
        "test acc":test_acc
    })

    print(f"{epoch}: Train loss: {total_train_loss/len(train_edge_loader)} Train acc: {train_acc}")
    print(f"{epoch}: Test loss: {total_test_loss/len(test_edge_loader)} Test acc: {test_acc}")
    #print(f"Test loss: {test_loss} Test acc:{test_acc} "
    #      +f"P:{precision_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())} "
    #      +f"R:{recall_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())}")
    print(total_out_pred)
    print(total_out_pred.cpu().numpy())
    print(test_labels)
    p = precision_score(total_out_pred.cpu().numpy(),test_labels.cpu().numpy())
    r = recall_score(total_out_pred.cpu().numpy(),test_labels.cpu().numpy())
    
    if best_metrics["train_loss"] == None or train_loss.cpu() < best_metrics["train_loss"]:
        best_metrics["train_loss"] = train_loss.cpu().item()
        best_metrics["r"] = r
        best_metrics["p"] = p

    early_stopping(train_loss, linkpred_model)
    if early_stopping.early_stop:
        print("Early Stopping!")
        break
#for k,v in best_metrics.items():
#    average_metrics[k].append(v)
#experiment.log(f"{best_metrics}\n")
#for k,v in average_metrics.items():
#    experiment.log(f"{k}:{torch.mean(torch.Tensor(v)).item()}({torch.var(torch.Tensor(v)).item()})\n")
for k,v in best_metrics.items():
    experiment.log(f"{k}:{v}")
wandb.log({
    "r" : best_metrics["r"],
    "p" : best_metrics["p"]
})

experiment.finish()
"""
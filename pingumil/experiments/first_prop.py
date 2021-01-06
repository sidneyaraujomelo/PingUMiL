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
import pickle
import time


dataset_folder = "dataset/SmokeSquadron/ss_het"
dataset_prefix = "prov"
model_config = "configs/ct_sagemodel.json"
timestamp = time.time()
output_file = f"sshet_linkpred_firstprop_{timestamp}.txt"
standardization = True

print(output_file)

with open(output_file, "a") as fp:
    fp.write(f"Experiment {timestamp}\n")


if os.path.exists(os.path.join(dataset_folder, f"{dataset_prefix}-G.data")):
    data = torch.load(os.path.join(dataset_folder, f"{dataset_prefix}-G.data"))
else:
    #First, we load all the data in the dataset folder.
    graph_json = json.load(open(os.path.join(dataset_folder,
                                            f"{dataset_prefix}-G.json")))
    graph = json_graph.node_link_graph(graph_json)
    #Create data object for pytorch geometric (takes a long time)
    data = from_networkx(graph)
    torch.save(data, os.path.join(dataset_folder, f"{dataset_prefix}-G.data"))
    print(data)

#Load attribute set list that describes each set
atbsets_list = json.load(open(os.path.join(dataset_folder, "prov-atbset_list.json")))
print(atbsets_list)

#Now, we load the attribute set map files
node_maps = []
node_maps_files = sorted([x for x in os.listdir(dataset_folder) if x.endswith("-map.json")])
node_maps = [json.load(open(os.path.join(dataset_folder, x))) for x in node_maps_files]

#Now, we load the attribute set feats files
node_feats = []
node_feats_files = sorted([x for x in os.listdir(dataset_folder) if x.endswith("-feats.npy")])
node_feats = [torch.from_numpy(np.load(os.path.join(dataset_folder, x))).float() for x in node_feats_files]

#Check if everything is sound
assert len(node_feats) == len(node_maps)

for k in range(len(node_feats)):
    assert len(node_maps[k])==node_feats[k].size()[0]

#Get list of all attributes in any type of node
atbs_list = [atb for atbsets_sublist in atbsets_list for atb in atbsets_sublist]
atbs_list = list(dict.fromkeys(atbs_list))
#Create a dictionary mapping each attribute to its index on node_feats
atbs_maps = {atb : {} for atb in atbs_list}
f = lambda l,v: l.index(v) if v in l else -1
for atb in atbs_maps.keys():
    atbs_maps[atb] = {k:f(v,atb) for k,v in enumerate(atbsets_list)}
print(atbs_maps)
print(node_feats[0].shape)
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

node_feats = torch.cat(node_feats)

with open(output_file, "a") as fp:
    fp.write(f"Standardization: {standardization}\n")
if standardization:
    scaler = StandardScaler()
    node_feats = torch.from_numpy(scaler.fit_transform(node_feats)).type(torch.FloatTensor)

#Now, we load the files structuring the folds for k-fold cross validation
#containing positive and negative indirect edges
fold_files = sorted(os.listdir(
    os.path.join(dataset_folder, "clf")
))
train_folds = {p : json.load(
                            open(os.path.join(dataset_folder,"clf",p))
                            ) for p in fold_files if "train" in p}
test_folds = {p : json.load(
                            open(os.path.join(dataset_folder,"clf",p))
                            ) for p in fold_files if "test" in p}

#Configuration
num_samples = [2, 2]
batch_size = 64
walk_length = 1
num_neg_samples = 1
epochs = 1000
sage_input_dim = 64

dict_x2m = {}
#update edge_index according to node_maps
for node_map in node_maps:
    offset = len(dict_x2m)
    dict_x2m.update({k+offset:v for k,v in enumerate(node_map)})
data.edge_index[0] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[0].tolist()])
data.edge_index[1] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[1].tolist()])
data.x = node_feats
dim_types = [data.x.shape[-1]]
print(dim_types)
data.node_map = [0]

subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=[-1], batch_size=batch_size, shuffle=False,
    num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sage_config = json.load(open(model_config))[0]
sage_config["in_channels"] = sage_input_dim
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
x = data.x.to(device)

#torch.autograd.set_detect_anomaly(True)
with open(output_file, "a") as fp:
    fp.write(f"Device: {device}, visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}\n")
    fp.write(f"Type Projection: {typeproj_model}\n")
    fp.write(f"Graph Representation Learning Model: {sage_model}\n")

def train(epoch):
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
    return loss

sage_early_stopping = EarlyStopping(patience=50, verbose=True, path=f"embed_model_{timestamp}.pt")
proj_early_stopping = EarlyStopping(patience=50, verbose=True, path=f"proj_model_{timestamp}.pt")
best_loss = 50
for epoch in range(1, epochs):
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
with open(output_file, "a") as fp:
    fp.write(f'Epoch: {epoch} -> Loss (Representation Learning): {best_loss:.4f}')

typeproj_model.load_state_dict(torch.load(proj_early_stopping.path))
sage_model.load_state_dict(torch.load(sage_early_stopping.path))
x_p = typeproj_model([x], [0])
z = sage_model.inference(x_p, subgraph_loader, device).detach()
z = z.to(device)
model_config = "configs/ct_sagemodel.json"
linkpred_config = json.load(open(model_config))[1]
linkpred_config["in_channels"] = z.size(1)*2
linkpred_model = load_model(linkpred_config).to(device)

with open(output_file, "a") as fp:
    fp.write(f"Link Prediction Model: {linkpred_model}\n")

average_metrics = { k : [] for k in ["train_loss", "p", "r"] }

for k in range(len(train_folds.keys())):
    
    train_fold = train_folds[f"clffold-{k}-train"]
    test_fold = test_folds[f"clffold-{k}-test"]
    
    #Transform Edge structure from fold files into COO tensors
    # Also obtains the class of each edge for both train and test
    train_source_ids = [dict_x2m[x["source"]] for x in train_fold]
    train_target_ids = [dict_x2m[x["target"]] for x in train_fold]
    train_edges = torch.tensor([train_source_ids, train_target_ids])
    train_class = torch.FloatTensor([x["class"] for x in train_fold])
    test_source_ids = [dict_x2m[x["source"]] for x in test_fold]
    test_target_ids = [dict_x2m[x["target"]] for x in test_fold]
    test_edges = torch.tensor([test_source_ids, test_target_ids])
    test_class = torch.FloatTensor([x["class"] for x in test_fold])
    
    train_edges = train_edges.to(device)
    test_edges = test_edges.to(device)
    train_class = train_class.to(device)
    test_class = test_class.to(device)
    link_optimizer = torch.optim.Adam(linkpred_model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=50, verbose=True, path=f"predictor_{timestamp}.pt")
    
    best_metrics = {
        "train_loss": 5000,
        "p": 0,
        "r": 0
    }
    
    print(f"Starting fold {k}")
    for epoch in range(1,epochs*10):
        linkpred_model.train()
        link_optimizer.zero_grad()
        out = linkpred_model(z, train_edges)
        
        train_loss = F.binary_cross_entropy_with_logits(out, train_class.unsqueeze(1))
        y_pred_tag = torch.round(torch.sigmoid(out))
        
        corrects_results_sum = torch.eq(y_pred_tag, train_class.unsqueeze(1))
        train_acc = float(corrects_results_sum.sum().item())/train_class.size()[0]
        
        train_loss.backward()
        link_optimizer.step()

        linkpred_model.eval()

        out_hat = linkpred_model(z, test_edges)
        test_loss = F.binary_cross_entropy_with_logits(out_hat, test_class.unsqueeze(1))
        out_pred = torch.round(torch.sigmoid(out_hat)).detach()
        
        test_correct = torch.eq(out_pred, test_class.unsqueeze(1))
        test_acc =  float(test_correct.sum().item()) / test_class.size()[0]
        print(f"{k}: Train loss: {train_loss} Train acc: {train_acc}")
        #print(f"Test loss: {test_loss} Test acc:{test_acc} "
        #      +f"P:{precision_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())} "
        #      +f"R:{recall_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())}")
        p = precision_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())
        r = recall_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())
        if train_loss.cpu() < best_metrics["train_loss"]:
            best_metrics["train_loss"] = train_loss.cpu().item()
            best_metrics["r"] = r
            best_metrics["p"] = p

        early_stopping(train_loss, linkpred_model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break
    for k,v in best_metrics.items():
        average_metrics[k].append(v)

    with open(output_file, "a") as fp:
        fp.write(f"{best_metrics}\n")
with open(output_file, "a") as fp:
    for k,v in average_metrics.items():
        fp.write(f"{k}:{torch.mean(torch.Tensor(v)).item()}({torch.var(torch.Tensor(v)).item()})\n")
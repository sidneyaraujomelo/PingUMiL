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
from torch_geometric.utils import add_self_loops
from torch_cluster import random_walk
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import remove_isolated_nodes
from pingumil.models import load_model
from pingumil.util.pytorchtools import EarlyStopping
from pingumil.experiments.ssdeathpred.ssdeathpred_exp import SSDeathPredBaseExperiment
import wandb
import time

project_name = "deathpred"
group_name = "sup_gs"
wandb_config = {
    "batch_size" : 2048,
    "lr": 1e-5,
    "lr_clf" : 1e-5,
    "weight_decay": 1e-5,
    "weight_decay_clf": 1e-5,
    "gnn": "gs",
}

#wandb stuff
run = wandb.init(project=project_name, group=group_name,
                 config=wandb_config)

experiment = SSDeathPredBaseExperiment(experiment_tag=group_name,
                               epochs=1000,
                               cls_epochs=1000,
                               timestamp=time.time(),
                               patience=100,
                               wandb=wandb,
                               model_config="configs/ssdeathpred_sagemodel_v4.json",
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
    
#Standardize data from all graphs
experiment.log(f"Standardization: {experiment.standardization}\n")
if experiment.standardization:
    dataset = experiment.standardize_all(dataset)
    
#Configuration
num_samples = [2, 2, 2]
batch_size = wandb.config.batch_size
walk_length = 1
num_neg_samples = 1
sage_input_dim = 64

def get_graph_name(graph_name_xml):
    graph_name = graph_name_xml.split(".")[0]
    graph_entries = [x for x in dataset.keys() if graph_name in x]
    assert len(graph_entries) == 1
    return graph_entries[0]

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
    data.edge_index = add_self_loops(data.edge_index)[0]
    data.edge_index[0] = torch.LongTensor([dict_m2x[idx] for idx in data.edge_index[0].tolist()])
    data.edge_index[1] = torch.LongTensor([dict_m2x[idx] for idx in data.edge_index[1].tolist()])
    data.x = node_feats
    dim_types = [data.x.shape[-1]]
    data.node_map = [0]
    data.player01_idxs = []
    data.player01_y = []
    data.player02_idxs = []
    data.player02_y = []
    data.graph_name = [graph_id]*data.x.size()[0]
    graph_data["graph"] = data
    dataset[graph_id]["dict_m2x"] = dict_m2x
    dataset[graph_id]["dim_types"] = dim_types

def update_dataset_with_cls_data(df, dataset):
    for _, row in df.iterrows():
        graph_id = get_graph_name(row["source"])
        graph_data = dataset[graph_id]
        graph_dm2x = graph_data["dict_m2x"]
        graph_id2dm = graph_data["id_map"]
        player01_idx = graph_dm2x[graph_id2dm[str(int(row["node_idxplayer01"]))]]
        player02_idx = graph_dm2x[graph_id2dm[str(int(row["node_idxplayer02"]))]]
        player01_willdie = row["willdieplayer01"]
        player02_willdie = row["willdieplayer02"]
        dataset[graph_id]["graph"].player01_idxs.append(player01_idx)
        dataset[graph_id]["graph"].player01_y.append(player01_willdie)
        dataset[graph_id]["graph"].player02_idxs.append(player02_idx)
        dataset[graph_id]["graph"].player02_y.append(player02_willdie)

update_dataset_with_cls_data(experiment.train_df, dataset)
update_dataset_with_cls_data(experiment.test_df, dataset)

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

classification_config = json.load(open(experiment.model_config))[1]
classification_config["in_channels"] = sage_config["out_channels"]
classification_model = load_model(classification_config).to(device)

classification_model = classification_model.to(device)

def learning_routine(epoch, data_dict, sage_model, typeproj_model, clf_model,
                     sage_optimizer, typeproj_optimizer, clf_optimizer, stage="train"):
    i = 0
    total_loss = 0
    total_acc = 0
    total_player01_f1 = 0
    total_player02_f1 = 0
    for graph_id, graph_data in data_dict.items():
        data = graph_data["graph"]
        x = data.x.to(device)
        if stage == "train":
            sage_model.train()
            typeproj_model.train()
            clf_model.train()
        else:
            sage_model.eval()
            typeproj_model.eval()
            clf_model.eval()
            
        typeproj_optimizer.zero_grad()
        sage_optimizer.zero_grad()
        clf_optimizer.zero_grad()
        
        pbar = tqdm(total=data.num_nodes)
        pbar.set_description(f'Epoch {epoch:02d}')

        #Type Projection Step
        x_p = typeproj_model([x], [0])

        #node_idx = torch.randperm(data.num_nodes)  # shuffle all nodes
        train_emb_loader = NeighborSampler(
            data.edge_index, sizes=num_samples, batch_size=batch_size, shuffle=False,
            num_workers=0)

        z = []
        
        for batch_size_, u_id, adjs_u in train_emb_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs_u = [adj.to(device) for adj in adjs_u]
            output = sage_model(x_p[u_id], adjs_u)
            z.append(output)
            pbar.update(batch_size_)
        pbar.close()
        
        z = torch.cat(z, dim=0)
        assert z.size()[0] == x_p.size()[0]
        #print(data.graph_name)
        X_z_p1 = []
        for cls_x in data.player01_idxs:
            player01_node_zs = z[cls_x,:]
            X_z_p1.append(player01_node_zs)
        X_z_p1 = torch.stack(X_z_p1)
        
        X_z_p2 = []
        for cls_x in data.player02_idxs:
            player02_node_zs = z[cls_x,:]
            X_z_p2.append(player02_node_zs)
        X_z_p2 = torch.stack(X_z_p2)
        
        X_z = torch.concat((X_z_p1,X_z_p2))
        print(X_z.shape)
        y_hat = classification_model(X_z, sigmoid=False)
        y_train = torch.Tensor(data.player01_y+data.player02_y)
        y_train = y_train.view(-1, 1).to(device)
        pos_weight = torch.tensor([3.0]).to(device)
        loss = F.binary_cross_entropy_with_logits(y_hat, y_train, pos_weight=pos_weight)
        
        #Separate y_trains for each player
        player01_y_train = y_train[:X_z_p1.size()[0]]
        player01_y_hat = y_hat[:X_z_p1.size()[0]]
        player02_y_train = y_train[X_z_p1.size()[0]:]
        player02_y_hat = y_hat[X_z_p1.size()[0]:]
        
        #Detaching all tensors
        np_y_train = y_train.cpu().detach().numpy()
        np_y_pred_train = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()
        
        acc, _, _, _ = experiment.calculate_metrics(np_y_train, np_y_pred_train)
        
        loss.backward(retain_graph=True)
        clf_optimizer.step()
        sage_optimizer.step()
        typeproj_optimizer.step()

        wandb.log({f"graph {stage} loss": loss})
        wandb.log({f"graph {stage} acc": acc})
        
        total_loss = total_loss + loss
        total_acc = total_acc + acc
        
        np1_y_train = player01_y_train.cpu().detach().numpy()
        np1_y_pred_train = torch.round(torch.sigmoid(player01_y_hat)).cpu().detach().numpy()
        np2_y_train = player02_y_train.cpu().detach().numpy()
        np2_y_pred_train = torch.round(torch.sigmoid(player02_y_hat)).cpu().detach().numpy()
        
        _, _, _, p1f1 = experiment.calculate_metrics(np1_y_train, np1_y_pred_train)
        _, _, _, p2f1 = experiment.calculate_metrics(np2_y_train, np2_y_pred_train)
        
        total_player01_f1 = total_player01_f1 + p1f1
        total_player02_f1 = total_player01_f1 + p2f1
        i = i+1
    wandb.log({f"total {stage} loss": total_loss})
    wandb.log({f"average {stage} acc": total_acc/i})
    if (stage == "val"):
        wandb.log({f"avg {stage} player_01 f1": total_player01_f1/i})
        wandb.log({f"avg {stage} player_02 f1": total_player02_f1/i})
    return total_loss, i

def train(epoch,
          data_dicts,
          sage_model,
          typeproj_model,
          clf_model,
          sage_optimizer,
          typeproj_optimizer,
          clf_optimizer):
    train_data_dict, val_data_dict = data_dicts
    #TRAIN ROUTINE
    total_train_loss, num_train_graphs = learning_routine(epoch,
                                                          train_data_dict,
                                                          sage_model,
                                                          typeproj_model,
                                                          clf_model,
                                                          sage_optimizer,
                                                          typeproj_optimizer,
                                                          clf_optimizer,
                                                          "train")
    #Validation routine
    total_val_loss, num_val_graphs = learning_routine(epoch,
                                                      val_data_dict,
                                                      sage_model,
                                                      typeproj_model,
                                                      clf_model,
                                                      sage_optimizer,
                                                      typeproj_optimizer,
                                                      clf_optimizer,
                                                      "val")
        
    
    return total_train_loss/num_train_graphs, total_val_loss/num_val_graphs

def test_routine(data_dict,
         sage_model,
         typeproj_model,
         clf_model):
    
    #TODO: Increment the test routine to deal with both player01 and player 02 output csvs
    test_output_dicts = []
    test_pred_dicts = []
    test_acc = []
    test_p = []
    test_r = []
    test_f1 = []
    i=0
    for graph_name, graph_data in data_dict.items():
        data = graph_data["graph"]
        x = data.x.to(device)
        
        sage_model.eval()
        typeproj_model.eval()
        clf_model.eval()

        #Type Projection Step
        x_p = typeproj_model([x], [0])

        #node_idx = torch.randperm(data.num_nodes)  # shuffle all nodes
        train_emb_loader = NeighborSampler(
            data.edge_index, sizes=num_samples, batch_size=batch_size, shuffle=False,
            num_workers=0)

        z = []
        
        for batch_size_, u_id, adjs_u in train_emb_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs_u = [adj.to(device) for adj in adjs_u]
            output = sage_model(x_p[u_id], adjs_u)
            z.append(output)
        
        z = torch.cat(z, dim=0)
        assert z.size()[0] == x_p.size()[0]
        #print(data.graph_name)
        X_z_p1 = []
        for cls_x in data.player01_idxs:
            player01_node_zs = z[cls_x,:]
            X_z_p1.append(player01_node_zs)
        X_z_p1 = torch.stack(X_z_p1)
        
        X_z_p2 = []
        for cls_x in data.player02_idxs:
            player02_node_zs = z[cls_x,:]
            X_z_p2.append(player02_node_zs)
        X_z_p2 = torch.stack(X_z_p2)
        
        X_z = torch.concat((X_z_p1,X_z_p2))
        y_hat = classification_model(X_z, sigmoid=False)
        y_test = torch.Tensor(data.player01_y+data.player02_y)
        y_test = y_test.view(-1, 1).to(device)
        
        #Detaching all tensors
        np_y_test = y_test.cpu().detach().numpy()
        np_y_pred_test = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()
        
        acc, p, r, f1 = experiment.calculate_metrics(np_y_test, np_y_pred_test)
        test_acc.append(acc)
        test_p.append(p)
        test_r.append(r)
        test_f1.append(f1)

        #Separate y_trains for each player
        player01_y_test = y_test[:X_z_p1.size()[0]]
        player01_y_hat = y_hat[:X_z_p1.size()[0]]
        player02_y_test = y_test[X_z_p1.size()[0]:]
        player02_y_hat = y_hat[X_z_p1.size()[0]:]
        
        #Detaching all tensors
        #np_y_train = y_train.cpu().detach().numpy()
        #np_y_pred_train = torch.round(torch.sigmoid(y_hat)).cpu().detach().numpy()
        np1_y_test = player01_y_test.cpu().detach().numpy()
        np1_y_log_test = torch.sigmoid(player01_y_hat).cpu().detach().numpy().flatten()
        np1_y_pred_test = torch.round(torch.sigmoid(player01_y_hat)).cpu().detach().numpy().flatten()
        np2_y_test = player02_y_test.cpu().detach().numpy()
        np2_y_log_test = torch.sigmoid(player02_y_hat).cpu().detach().numpy().flatten()
        np2_y_pred_test = torch.round(torch.sigmoid(player02_y_hat)).cpu().detach().numpy().flatten()
        
        p1_output_dict = experiment.create_output_dict()
        p1_output_dict = experiment.add_metrics_to_output_dict(p1_output_dict, None, None, [(np1_y_test, np1_y_pred_test, "test")])

        p2_output_dict = experiment.create_output_dict()
        p2_output_dict = experiment.add_metrics_to_output_dict(p2_output_dict, None, None, [(np2_y_test, np2_y_pred_test, "test")])
        
        output_dicts = [p1_output_dict, p2_output_dict]
        
        for k, player_dict in enumerate(output_dicts):
            wandb.log({f"graph test p{k+1} acc": player_dict["accuracy_test"][0]})
            wandb.log({f"graph test p{k+1} p": player_dict["precision_test"][0]})
            wandb.log({f"graph test p{k+1} r": player_dict["recall_test"][0]})
            wandb.log({f"graph test p{k+1} f1": player_dict["f1_test"][0]})
        
        test_output_dicts.append(output_dicts)
        
        pred_p1_dict = experiment.create_prediction_dict(data.player01_idxs, [graph_name]*X_z_p1.size()[0], np1_y_test, np1_y_log_test, np1_y_pred_test)
        pred_p2_dict = experiment.create_prediction_dict(data.player02_idxs, [graph_name]*X_z_p2.size()[0], np2_y_test, np2_y_log_test, np2_y_pred_test)
        
        test_pred_dicts.append((pred_p1_dict, pred_p2_dict))
        
        i = i+1
    
    
    wandb.log({f"test acc": sum(test_acc)/i})
    wandb.log({f"test p": sum(test_p)/i})
    wandb.log({f"test r": sum(test_r)/i})
    wandb.log({f"test f1": sum(test_f1)/i})
    for k in range(2):
        for metric in ["accuracy", "precision", "recall", "f1"]:
            player_metric = [x[k][f"{metric}_test"][0] for x in test_output_dicts]
            wandb.log({f"player{k+1} {metric} mean" : np.mean(player_metric)})
            wandb.log({f"player{k+1} {metric} std" : np.var(player_metric)})
    return (test_acc, test_p, test_r, test_f1), i, test_output_dicts, test_pred_dicts

typeproj_model.reset_parameters()
typeproj_optimizer = torch.optim.Adam(typeproj_model.parameters(),
                                    lr=wandb.config.lr,
                                    weight_decay=wandb.config.weight_decay)
sage_model.reset_parameters()
sage_optimizer = torch.optim.Adam(sage_model.parameters(),
                                lr=wandb.config.lr,
                                weight_decay=wandb.config.weight_decay)

classification_model.reset_parameters()
classification_optimizer = torch.optim.Adam(classification_model.parameters(),
                                            lr = wandb.config.lr_clf,
                                            weight_decay=wandb.config.weight_decay_clf)

sage_early_stopping = experiment.get_early_stopping(verbose=True,
                                                    prefix="embed")
proj_early_stopping = experiment.get_early_stopping(verbose=True,
                                                    prefix=f"proj")
clf_early_stopping = experiment.get_early_stopping(verbose=True,
                                                   prefix="clf")

# Separate train, validation and test data
test_graphs = experiment.test_df["source"].unique()
train_index, val_index = next(experiment.split_train_df(9, "willdieplayer01"))
train_graphs = experiment.train_df.loc[train_index, "source"].unique()
val_graphs = experiment.train_df.loc[val_index, "source"].unique()
assert len(train_graphs) + len(val_graphs) == len(experiment.train_df["source"].unique())
assert len(train_index) + len(val_index) == len(experiment.train_df.index)

#data_list contains all the graph data to be sent to the DataLoader
train_data_dict = {k:v for k, v in dataset.items() if k in list(map(get_graph_name, train_graphs))}
test_data_dict = {k:v for k, v in dataset.items() if k in list(map(get_graph_name, test_graphs))}
val_data_dict = {k:v for k, v in dataset.items() if k in list(map(get_graph_name, val_graphs))}

# training routine
best_loss = np.inf
for epoch in range(1, experiment.epochs):
    losses = train(epoch,
                   (train_data_dict, val_data_dict),
                    sage_model,
                    typeproj_model,
                    classification_model,
                    sage_optimizer,
                    typeproj_optimizer,
                    classification_optimizer)
    train_loss, val_loss = losses
    wandb.log({"train loss":train_loss})
    wandb.log({"val loss":val_loss})
    if val_loss < best_loss:
        best_loss = val_loss
    sage_early_stopping(val_loss, sage_model)
    proj_early_stopping(val_loss, typeproj_model)
    clf_early_stopping(val_loss, classification_model)
    print(f'Epoch {epoch:02d}, Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}')
    if sage_early_stopping.early_stop:
        print("Early Stopping!")
        break
print(f'Best Loss: {best_loss:.4f}')
experiment.log(f'Epoch: {epoch} -> Loss (Supervised Learning): {best_loss:.4f}')

typeproj_model.load_state_dict(torch.load(proj_early_stopping.path))
sage_model.load_state_dict(torch.load(sage_early_stopping.path))
classification_model.load_state_dict(torch.load(clf_early_stopping.path))

metrics, n_graphs, test_output_dicts, test_pred_dicts = test_routine(test_data_dict, sage_model, typeproj_model, classification_model)

experiment.output_pred_dicts(test_pred_dicts)

run.finish()
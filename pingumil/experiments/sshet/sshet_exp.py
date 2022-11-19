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
from pingumil.util.metric import get_edge_type_list
from pingumil.util.pytorchtools import EarlyStopping
import pickle
import time

class SSHetBaseExperiment():
    def __init__(
            self,
            dataset_folder="dataset/SmokeSquadron/ss_het",
            dataset_prefix="prov",
            model_config="configs/ct_sagemodel.json",
            experiment_tag="base",
            timestamp=None,
            standardization=True,
            epochs=1000,
            patience=100,
            wandb=None):
        self.dataset_folder = dataset_folder
        self.dataset_prefix = dataset_prefix
        self.model_config = model_config
        self.experiment_tag = experiment_tag
        self.wandb = wandb
        if timestamp:
            self.timestamp=timestamp
        else:
            self.timestamp = time.time()
        if not self.wandb:
            self.run_name = self.timestamp
        else:
            self.run_name = self.wandb.run.name
        self.standardization = standardization
        self.epochs = epochs
        self.patience = patience
        self.log_folder = f"experiment_log/{self.dataset_prefix}_{self.experiment_tag}/{self.run_name}"
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        self.output_file = f"{self.log_folder}/{self.run_name}.txt"
        self.node_type_labels = {
            0 : "item",
            1 : "player",
            2 : "smoke",
            3 : "missle",
            4 : "fireworks"
        }
        self.log(f"Experiment {self.timestamp}")

    def log(self, message, mode="a"):
        with open(self.output_file, mode) as fp:
            fp.write(message+"\n")
    
    def read_data(self):
        if os.path.exists(os.path.join(self.dataset_folder, f"{self.dataset_prefix}-G.data")):
            data = torch.load(os.path.join(self.dataset_folder, f"{self.dataset_prefix}-G.data"))
        else:
            #First, we load all the data in the dataset folder.
            graph_json = json.load(open(os.path.join(self.dataset_folder,
                                                    f"{self.dataset_prefix}-G.json")))
            graph = json_graph.node_link_graph(graph_json)
            #Create data object for pytorch geometric (takes a long time)
            data = from_networkx(graph)
            torch.save(data, os.path.join(self.dataset_folder, f"{self.dataset_prefix}-G.data"))
            print(data)

        #Load attribute set list that describes each set
        atbsets_list = json.load(open(os.path.join(self.dataset_folder, "prov-atbset_list.json")))
        print(atbsets_list)

        #Now, we load the attribute set map files
        node_maps = []
        node_maps_files = sorted([x for x in os.listdir(self.dataset_folder) if x.endswith("-map.json")])
        node_maps = [json.load(open(os.path.join(self.dataset_folder, x))) for x in node_maps_files]

        #Now, we load the attribute set feats files
        node_feats = []
        node_feats_files = sorted([x for x in os.listdir(self.dataset_folder) if x.endswith("-feats.npy")])
        node_feats = [torch.from_numpy(np.load(os.path.join(self.dataset_folder, x))).float() for x in node_feats_files]

        #Check if everything is sound
        assert len(node_feats) == len(node_maps)

        for k in range(len(node_feats)):
            assert len(node_maps[k])==node_feats[k].size()[0]
        
        #Now, we load the files structuring the folds for k-fold cross validation
        #containing positive and negative indirect edges
        fold_files = sorted(os.listdir(
            os.path.join(self.dataset_folder, "clf")
        ))
        train_folds = {p : json.load(
                                    open(os.path.join(self.dataset_folder,"clf",p))
                                    ) for p in fold_files if "train" in p}
        test_folds = {p : json.load(
                                    open(os.path.join(self.dataset_folder,"clf",p))
                                    ) for p in fold_files if "test" in p}

        return data, atbsets_list, node_maps, node_feats, train_folds, test_folds

    def standardize(self, data):
        scaler = StandardScaler()
        stdzed_data = torch.from_numpy(scaler.fit_transform(data)).type(torch.FloatTensor)
        return stdzed_data
    
    def link_prediction_step(self, device, linkpred_model, train_folds, input_data, test_folds, dict_x2m, edge_type_function=None, wandb=None):
        self.log(f"Link Prediction Model: {linkpred_model}")
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
            early_stopping = self.get_early_stopping(patience=self.patience,
                                           verbose=True,
                                           prefix="predictor")
            link_optimizer = torch.optim.Adam(linkpred_model.parameters(),
                                              lr=wandb.config.lr_lp)
            
            best_metrics = {
                "train_loss": np.inf,
                "p": 0,
                "r": 0
            }
            
            print(f"Starting fold {k}")
            for epoch in range(1,self.epochs):
                edge_type_results = {}
                linkpred_model.train()
                link_optimizer.zero_grad()
                out = linkpred_model(input_data, train_edges)
                
                train_loss = F.binary_cross_entropy_with_logits(out, train_class.unsqueeze(1))
                y_pred_tag = torch.round(torch.sigmoid(out))
                
                corrects_results_sum = torch.eq(y_pred_tag, train_class.unsqueeze(1))
                total_train_correct_results = corrects_results_sum.sum().item()

                #Per edge type accuracy
                edge_type_list = get_edge_type_list(
                    train_edges,
                    edge_type_function
                )
                for edge_type in set(edge_type_list):
                    edge_type_idx = [i for i,v in enumerate(edge_type_list) if v == edge_type]
                    if edge_type not in edge_type_results:
                        edge_type_results[edge_type] = {"correct" : 0, "total" : 0}
                    edge_type_results[edge_type]["total"] += len(edge_type_idx)
                    edge_correct_pred = corrects_results_sum[edge_type_idx].sum().item()
                    edge_type_results[edge_type]["correct"] += edge_correct_pred

                train_acc = float(corrects_results_sum.sum().item())/train_class.size()[0]
                
                train_loss.backward()
                link_optimizer.step()
                wandb.log({
                    "linkpred train loss":train_loss.item(),
                    "train acc":train_acc
                })
                for edge_type, value in edge_type_results.items():
                    edge_type_acc = float(value["correct"]/value["total"])
                    wandb.log({
                        f"train {edge_type} acc" : edge_type_acc
                })

                linkpred_model.eval()

                out_hat = linkpred_model(input_data, test_edges)
                test_loss = F.binary_cross_entropy_with_logits(out_hat, test_class.unsqueeze(1))
                out_pred = torch.round(torch.sigmoid(out_hat)).detach()
                
                test_correct_results = torch.eq(out_pred, test_class.unsqueeze(1))
                test_acc =  float(test_correct_results.sum().item()) / test_class.size()[0]

                #Per edge type accuracy
                edge_type_list = get_edge_type_list(
                    test_edges,
                    edge_type_function
                )
                for edge_type in set(edge_type_list):
                    edge_type_idx = [i for i,v in enumerate(edge_type_list) if v == edge_type]
                    if edge_type not in edge_type_results:
                        edge_type_results[edge_type] = {"correct" : 0, "total" : 0}
                    edge_type_results[edge_type]["total"] += len(edge_type_idx)
                    edge_correct_pred = test_correct_results[edge_type_idx].sum().item()
                    edge_type_results[edge_type]["correct"] += edge_correct_pred

                wandb.log({
                    "linkpred test loss":test_loss,
                    "test acc":test_acc
                })
                for edge_type, value in edge_type_results.items():
                    edge_type_acc = float(value["correct"]/value["total"])
                    wandb.log({
                        f"test {edge_type} acc" : edge_type_acc
                    })
                print(f"{epoch}: Train loss: {train_loss.item()} Train acc: {train_acc}")
                print(f"{epoch}: Test loss: {test_loss.item()} Test acc: {test_acc}")
                #print(f"Test loss: {test_loss} Test acc:{test_acc} "
                #      +f"P:{precision_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())} "
                #      +f"R:{recall_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())}")
                p = precision_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())
                r = recall_score(out_pred.cpu().numpy(),test_class.unsqueeze(1).cpu().numpy())
                if best_metrics["train_loss"] == None or train_loss.cpu().item() < best_metrics["train_loss"]:
                    best_metrics["train_loss"] = train_loss.cpu().item()
                    best_metrics["r"] = r
                    best_metrics["p"] = p

                early_stopping(train_loss, linkpred_model)
                if early_stopping.early_stop:
                    print("Early Stopping!")
                    break
            for k,v in best_metrics.items():
                average_metrics[k].append(v)

            self.log(f"{best_metrics}")
        for k,v in average_metrics.items():
            wandb.log({
                f"{k}_avg":torch.mean(torch.Tensor(v)).item(),
                f"{k}_var":torch.var(torch.Tensor(v)).item()
            })
        self.finish()

    def get_early_stopping(self, patience, verbose, prefix, delta=0):
        return EarlyStopping(patience=patience,
                             verbose=verbose,
                             delta=delta,
                             path=f"{self.log_folder}/{prefix}_{self.timestamp}.pt")
    
    def finish(self):
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.timestamp))
        self.log(f"Experiment time: {elapsed}")
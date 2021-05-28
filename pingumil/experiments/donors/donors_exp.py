import json
import os
import torch
import wandb
import numpy as np
from argparse import ArgumentParser
from networkx.readwrite import json_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import NeighborSampler, Data, DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree, train_test_split_edges, negative_sampling
from torch_cluster import random_walk
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from pingumil.models import load_model
from pingumil.util.metric import get_edge_type_list
from pingumil.util.pytorchtools import EarlyStopping
import pickle
import time

class DonorsBaseExperiment():
    def __init__(
            self,
            dataset_folder="dataset/donors/donors",
            dataset_prefix="donor",
            model_config="configs/ct_sagemodel.json",
            experiment_tag="base",
            timestamp=None,
            wandb=None,
            standardization=True,
            split_edges=False,
            save_split=False,
            epochs=1000,
            patience=100):
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
        self.split_edges = split_edges
        self.save_split = save_split
        self.log_folder = f"experiment_log/{self.dataset_prefix}_{self.experiment_tag}/{self.run_name}"
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        self.output_file = f"{self.log_folder}/{self.run_name}.txt"
        self.node_type_labels = {
            0 : "donation",
            1 : "donor",
            2 : "project",
            3 : "teacher"
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
        atbsets_list = json.load(open(os.path.join(self.dataset_folder, f"{self.dataset_prefix}-featset_list.json")))
        print(atbsets_list)

        #Now, we load the attribute set map files
        node_maps = []
        node_maps_files = sorted([x for x in os.listdir(self.dataset_folder) if x.endswith("-map.json")])
        node_maps = [json.load(open(os.path.join(self.dataset_folder, x))) for x in node_maps_files]

        #Now, we load the attribute set feats files
        node_feats = []
        node_feats_files = sorted([x for x in os.listdir(self.dataset_folder) if x.endswith("-values.npy")])
        node_feats = [torch.from_numpy(np.load(os.path.join(self.dataset_folder, x), allow_pickle=True)).float() for x in node_feats_files]

        #Check if everything is sound
        assert len(node_feats) == len(node_maps)

        for k in range(len(node_feats)):
            assert len(node_maps[k])==node_feats[k].size()[0]

        data.maps = node_maps
        if self.split_edges:
            dict_x2m = {}
            #update edge_index according to node_maps
            for node_map in node_maps:
                offset = len(dict_x2m)
                dict_x2m.update({k+offset:v for k,v in enumerate(node_map)})
            data.edge_index[0] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[0].tolist()])
            data.edge_index[1] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[1].tolist()])
            #data.x = torch.zeros((data.origin.size()[0], sage_input_dim))
            #Now, we split dataset in train/test
            data = train_test_split_edges(data)
            if self.save_split:
                torch.save(data, os.path.join(self.dataset_folder, f"{self.dataset_prefix}-G.data"))

        return data, atbsets_list, node_maps, node_feats

    def standardize(self, data):
        scaler = StandardScaler()
        stdzed_data = torch.from_numpy(scaler.fit_transform(data)).type(torch.FloatTensor)
        return stdzed_data
    
    def link_prediction_step(self, device, linkpred_model, link_optimizer, data, z, edge_type_function, wandb):
        early_stopping = self.get_early_stopping(patience=self.patience, verbose=True, prefix=f"predictor")
        best_metrics = {
            "train_loss": None,
            "p": None,
            "r": None
        }
        #average_metrics = { k : [] for k,v in best_metrics.items() }

        for epoch in range(1,self.epochs):
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
            train_edge_loader = DataLoader(torch.arange(train_edges.size()[-1]),
                                           batch_size=self.wandb.config.batch_size, shuffle=True)
            print(f"{train_edges.size()} {train_labels.size()}")
            total_train_loss = 0
            total_train_correct_results = 0
            edge_type_results = {}
            for batch_edge_idx in train_edge_loader:
                #print(batch_edge_idx)
                out = linkpred_model(z, train_edges[:,batch_edge_idx]).flatten()
                train_loss = F.binary_cross_entropy_with_logits(out, train_labels[batch_edge_idx])
                y_pred_tag = torch.round(torch.sigmoid(out))    
                #print(y_pred_tag.size())
                #print(train_labels[batch_edge_idx].size())

                corrects_results_sum = torch.eq(y_pred_tag, train_labels[batch_edge_idx])
                total_train_correct_results += corrects_results_sum.sum().item()

                #Per edge type accuracy
                edge_type_list = get_edge_type_list(
                    train_edges[:,batch_edge_idx],
                    edge_type_function
                )
                for edge_type in set(edge_type_list):
                    edge_type_idx = [i for i,v in enumerate(edge_type_list) if v == edge_type]
                    if edge_type not in edge_type_results:
                        edge_type_results[edge_type] = {"correct" : 0, "total" : 0}
                    edge_type_results[edge_type]["total"] += len(edge_type_idx)
                    edge_correct_pred = corrects_results_sum[edge_type_idx].sum().item()
                    edge_type_results[edge_type]["correct"] += edge_correct_pred
            
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
            for edge_type, value in edge_type_results.items():
                edge_type_acc = float(value["correct"]/value["total"])
                wandb.log({
                    f"train {edge_type} acc" : edge_type_acc
                })


            linkpred_model.eval()

            test_edges = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=-1).to(device)
            test_labels = torch.zeros(data.test_pos_edge_index.size(1)+data.test_neg_edge_index.size(1), dtype=torch.float, device=device)
            test_labels[:data.test_pos_edge_index.size(1)] = 1.
            print(test_edges.size())
            #test_edge_data = Data(x=z, edge_index=test_edges, y=test_labels)
            test_edge_loader = DataLoader(torch.arange(test_edges.size()[-1]),
                                          batch_size=self.wandb.config.batch_size)
            
            total_test_loss = 0
            total_test_correct_results = 0
            total_out_pred = None
            edge_type_results = {}
            for batch_edge_idx in test_edge_loader:
                out_hat = linkpred_model(z, test_edges[:,batch_edge_idx]).flatten()
                
                test_loss = F.binary_cross_entropy_with_logits(out_hat, test_labels[batch_edge_idx])
                total_test_loss += test_loss.item()

                out_pred = torch.round(torch.sigmoid(out_hat)).detach()
                if total_out_pred != None:
                    total_out_pred = torch.cat((total_out_pred, out_pred), dim=-1)
                else:
                    total_out_pred = out_pred
                test_correct_results = torch.eq(out_pred, test_labels[batch_edge_idx])
                total_test_correct_results += test_correct_results.sum().item()

                #Per edge type accuracy
                edge_type_list = get_edge_type_list(
                    test_edges[:,batch_edge_idx],
                    edge_type_function
                )
                for edge_type in set(edge_type_list):
                    edge_type_idx = [i for i,v in enumerate(edge_type_list) if v == edge_type]
                    if edge_type not in edge_type_results:
                        edge_type_results[edge_type] = {"correct" : 0, "total" : 0}
                    edge_type_results[edge_type]["total"] += len(edge_type_idx)
                    edge_correct_pred = test_correct_results[edge_type_idx].sum().item()
                    edge_type_results[edge_type]["correct"] += edge_correct_pred

                wandb.log({"linkpred test batch loss":test_loss.item()})

            test_acc =  float(total_test_correct_results) / test_labels.size()[0]

            wandb.log({
                "linkpred test loss":total_test_loss/len(test_edge_loader),
                "test acc":test_acc
            })
            for edge_type, value in edge_type_results.items():
                edge_type_acc = float(value["correct"]/value["total"])
                wandb.log({
                    f"test {edge_type} acc" : edge_type_acc
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
            
            if best_metrics["train_loss"] == None or total_train_loss < best_metrics["train_loss"]:
                best_metrics["train_loss"] = total_train_loss
                best_metrics["r"] = r
                best_metrics["p"] = p

            early_stopping(train_loss, linkpred_model)
            if early_stopping.early_stop:
                print("Early Stopping!")
                break
        for k,v in best_metrics.items():
            self.log(f"{k}:{v}")
        wandb.log({
            "r" : best_metrics["r"],
            "p" : best_metrics["p"]
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



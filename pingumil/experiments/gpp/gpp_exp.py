import json
import os
import torch
from glob import glob
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
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import LeaveOneOut
from pingumil.models import load_model
from pingumil.util.metric import get_edge_type_list
from pingumil.util.pytorchtools import EarlyStopping
import pickle
import time

class GPPBaseExperiment():
    def __init__(
            self,
            dataset_folder="/raid/home/smelo/PingUMiL-pytorch/dataset/GPP/preprocessed_graphs_test",
            dataset_prefix="gpp",
            model_config="configs/gpp_sagemodel.json",
            experiment_tag="base",
            timestamp=None,
            standardization=True,
            epochs=1000,
            patience=100,
            wandb=None,
            override_data=False):
        self.dataset_folder = dataset_folder
        self.dataset_prefix = dataset_prefix
        self.model_config = model_config
        self.experiment_tag = experiment_tag
        self.wandb = wandb
        self.override_data = override_data
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
        self.log(f"Experiment {self.timestamp}")

    def log(self, message, mode="a"):
        with open(self.output_file, mode) as fp:
            fp.write(message+"\n")
    
    def read_data(self):
        graph_folders = glob(os.path.join(self.dataset_folder,"*"))
        dataset = {x: {} for x in graph_folders if not os.path.isfile(x)}
        for graph_folder in graph_folders:
            if os.path.isfile(graph_folder):
                continue
            print(graph_folder)
            graph_json_path = glob(os.path.join(graph_folder, f"{self.dataset_prefix}*-G.json"))[0]
            graph_json_file = os.path.basename(graph_json_path)
            graph_id = graph_json_file[len(self.dataset_prefix):-7]
            if (not self.override_data and 
                os.path.exists(os.path.join(self.dataset_folder, graph_folder,
                                            f"{self.dataset_prefix}{graph_id}-G.data"))):
                data = torch.load(os.path.join(self.dataset_folder, graph_folder,
                                            f"{self.dataset_prefix}{graph_id}-G.data"))
            else:
                #First, we load all the data in the dataset folder.
                graph_json = json.load(open(os.path.join(self.dataset_folder, graph_folder,
                                                        f"{self.dataset_prefix}{graph_id}-G.json")))
                graph = json_graph.node_link_graph(graph_json)
                #Create data object for pytorch geometric (takes a long time)
                data = from_networkx(graph)
                torch.save(data, os.path.join(self.dataset_folder, graph_folder,
                                            f"{self.dataset_prefix}{graph_id}-G.data"))
                #print(data)
            dataset[graph_folder]["graph"] = data

            #Load attribute set list that describes each set
            atbsets_list = json.load(open(os.path.join(self.dataset_folder, graph_folder,
                                                    f"{self.dataset_prefix}{graph_id}-atbset_list.json")))
            #print(atbsets_list)
            dataset[graph_folder]["atbset_list"] = atbsets_list

            #Now, we load the attribute set map files
            node_maps = []
            node_maps_files = sorted(
                [x for x in glob(os.path.join(self.dataset_folder, graph_folder, f"*{graph_id}*-map.json"))])
            node_maps = [json.load(open(x)) for x in node_maps_files]
            dataset[graph_folder]["node_maps"] = node_maps

            #Now, we load the attribute set feats files
            node_feats = []
            node_feats_files = sorted(
                [x for x in glob(os.path.join(self.dataset_folder, graph_folder, f"*{graph_id}*-feats.npy"))])
            node_feats = [torch.from_numpy(np.load(x)).float() for x in node_feats_files]
            dataset[graph_folder]["node_feats"] = node_feats

            #Now, we load the classes from the class maps
            class_maps_file = glob(os.path.join(self.dataset_folder, graph_folder, f"*{graph_id}*-class_map.json"))[0]
            class_map = json.load(open(class_maps_file))
            dataset[graph_folder]["class_map"] = class_map

            #Check if everything is sound
            assert len(node_feats) == len(node_maps)

            for k in range(len(node_feats)):
                assert len(node_maps[k])==node_feats[k].size()[0]
        return dataset

    def standardize_all(self, dataset):
        all_data = torch.cat([v["node_feats"] for k,v in dataset.items()])
        scaler = StandardScaler()
        stdzed_data = torch.from_numpy(scaler.fit_transform(all_data)).type(torch.FloatTensor)
        k = 0
        for graph_folder, graph_data in dataset.items():
            stdzed_graph_data = stdzed_data[k:k+graph_data["node_feats"].shape[0],:]
            dataset[graph_folder]["node_feats"] = stdzed_graph_data
            k = k + graph_data["node_feats"].shape[0]
        stdzed_confirm_data = torch.cat([v["node_feats"] for k,v in dataset.items()])
        assert stdzed_confirm_data.shape == all_data.shape
        return dataset

    def get_early_stopping(self, patience, verbose, prefix, delta=0):
        return EarlyStopping(patience=patience,
                             verbose=verbose,
                             delta=delta,
                             path=f"{self.log_folder}/{prefix}_{self.timestamp}.pt")
    
    def multilabel_classification_step(self, device, model, x, y):
        self.log(f"Multilabel Classification Model: {model}")
        average_metrics = { k : [] for k in ["train_loss", "p", "r", "test_loss"] }
        
        x = x.to(device)
        y = y.to(device)
        
        best_metrics = {
            "train_loss": np.inf,
            "p": 0,
            "r": 0
        }
        
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(x):
            early_stopping = self.get_early_stopping(patience=self.patience,
                                                 verbose=True,
                                                 prefix="predictor")
            model.reset_parameters()
            classification_optimizer = torch.optim.Adam(
                model.parameters(), lr=1e-3)
            for epoch in range(1,self.epochs):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                model.train()
                optimizer.zero_grad()
                
                y_hat = model(x_train, sigmoid=True)
                train_loss = F.binary_cross_entropy(y_hat, y_train)
                y_pred = torch.round(y_hat)
                
                train_loss.backward()
                optimizer.step()
                
                model.eval()
                y_hat_test = model(x_test, sigmoid=True)
                test_loss = F.binary_cross_entropy(y_hat_test, y_test)
                y_pred_test = torch.round(y_hat_test)
                
                print(f"{epoch}: Train loss: {train_loss.item()}")
                print(f"{epoch}: Test loss: {test_loss.item()}")
                
                p = precision_score(y_pred_test.detach().numpy(),
                                    y_test.detach().numpy(), average="micro")
                r = recall_score(y_pred_test.detach().numpy(),
                                 y_test.detach().numpy(), average="micro")
                if (best_metrics["train_loss"] == None or 
                    train_loss.cpu().item() < best_metrics["train_loss"]):
                    best_metrics["train_loss"] = train_loss.cpu().item()
                    best_metrics["r"] = r
                    best_metrics["p"] = p
                    best_metrics["test_loss"] = test_loss.cpu().item()

                early_stopping(train_loss, model)
                if early_stopping.early_stop:
                    print("Early Stopping!")
                    break
                
            for k,v in best_metrics.items():
                average_metrics[k].append(v)
        for k,v in average_metrics.items():
            self.log(f"{k}_avg: {torch.mean(torch.Tensor(v)).item()}")
            self.log(f"{k}_var {torch.var(torch.Tensor(v)).item()}")
        self.finish()
                
    def finish(self):
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.timestamp))
        self.log(f"Experiment time: {elapsed}")

if __name__ ==  "__main__":
    k = GPPBaseExperiment()
    dataset = k.read_data()
    print(dataset)
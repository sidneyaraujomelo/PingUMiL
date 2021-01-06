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

class SSHetBaseExperiment():
    def __init__(
            dataset_folder="dataset/SmokeSquadron/ss_het",
            dataset_prefix="prov",
            model_config="configs/ct_sagemodel.json",
            experiment_tag="base",
            standardization=True):
        dataset_folder = "dataset/SmokeSquadron/ss_het"
        dataset_prefix = "prov"
        model_config = "configs/ct_sagemodel.json"
        self.timestamp = time.time()
        self.output_file = f"sshet_linkpred_{experiment_tag}_{timestamp}.txt"
        self.standardization = True
        self.log(f"Experiment {self.timestamp}")

    def log(self, message, mode="a"):
        with open(self.output_file, mode) as fp:
            fp.write(message)
    
    def read_data():
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

        return data, atbsets_list, node_maps, node_feats

    def standardize(data):
        scaler = StandardScaler()
        stdzed_data = torch.from_numpy(scaler.fit_transform(data)).type(torch.FloatTensor)
        return stdzed_data
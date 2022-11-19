import networkx as nx
import json
import os
import numpy as np
from util.util import create_path

class PingDataset():
    def __init__(self):
        self.g = nx.Graph()
        self.featmap = {}
        self.labelmap = {}

    def save_to(self, path):
        create_path(path)
        json_g = nx.readwrite.json_graph.node_link_data(self.g)
        with open(os.path.join(path, "prov-G.json"), "w") as fp:
            fp.write(json.dumps(json_g))
        with open(os.path.join(path, "prov-labels.json"), "w") as fp:
            fp.write(json.dumps(self.labelmap))
        np_features = np.array([v for k, v in self.featmap.items()])
        np.save(os.path.join(path, "prov-feats.npy"), np_features)


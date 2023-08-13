import networkx as nx
import json
import os
import numpy as np
from collections import OrderedDict
from config.export import (create_output_json_graph,
                           create_output_json_idmap,
                           create_output_json_classmap,
                           create_output_json_attributeset_list,
                           create_output_json_attributeset_map,
                           create_output_json_features_map)

class PingDataset():
    def __init__(self):
        self.g = nx.Graph()
        self.node_atb_sets = []
        self.idmap = OrderedDict()
        self.classmap = OrderedDict()
        self.featmap = OrderedDict()

    def export(self, output_prefix, params):
        json_data = nx.readwrite.json_graph.node_link_data(self.g)
        create_output_json_graph(json_data, output_prefix, params)

        # Write id map json
        print("Writing Json Id Map File")
        create_output_json_idmap(self.idmap, output_prefix, params)

        # Write class_map json
        print("Writing Json Class Map File")
        create_output_json_classmap(self.classmap, output_prefix, params)

        # Write node attribute sets json
        print("Writing Json Node Attribute Sets File")
        create_output_json_attributeset_list(self.node_atb_sets, output_prefix, params)

        #Write feature numpy per attribute sets
        for atb_set_id, featmap in self.featmap.items():
            create_output_json_attributeset_map(list(featmap.keys()),
                                                f"{output_prefix}_atbset_{atb_set_id}",
                                                params)
            create_output_json_features_map(featmap,
                                            f"{output_prefix}_atbset_{atb_set_id}",
                                            params)


import networkx as nx
from config.export import *

class ParsedGraphExporter():
    def __init__(self, params):
        self.params = params
    
    def export(self, output_prefix, hnx_parser):
        json_data = nx.readwrite.json_graph.node_link_data(hnx_parser.g)
        create_output_json_graph(json_data, output_prefix)

        # Write id map json
        print("Writing Json Id Map File")
        create_output_json_idmap(hnx_parser.idmap, output_prefix)

        # Write class_map json
        print("Writing Json Class Map File")
        create_output_json_classmap(hnx_parser.classmap, output_prefix)

        # Write node attribute sets json
        print("Writing Json Node Attribute Sets File")
        create_output_json_attributeset_list(hnx_parser.node_atb_sets, output_prefix)
        
        #Write feature numpy per attribute sets
        for atb_set_id, featmap in hnx_parser.featmap.items():
            create_output_json_attributeset_map(list(featmap.keys()), f"{output_prefix}_atbset_{atb_set_id}")
            create_output_json_features_map(featmap, f"{output_prefix}_atbset_{atb_set_id}")
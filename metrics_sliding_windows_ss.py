import os
import json
import networkx as nx
from config.util import loadProvenanceXML
from prov_hnx_parser import ProvHnxParser

input_file = "../smokesquadrondataset/original provenance graphs/j-2019-05-29-14-11-49.xml"
xmlFile = loadProvenanceXML(input_file)
parse_config = json.load(open("config/parse_config.json","r"))
data_config = json.load(open("config/smokesquad_config.json","r"))
assert len(data_config["attrib_name_list"]) == len(data_config["attrib_type_list"])
assert len(data_config["attrib_name_list"]) == len(data_config["attrib_default_value_list"])
print(parse_config, data_config)
G = nx.Graph()
G.graph['name'] = os.path.basename(input_file)
parser = ProvHnxParser(xmlFile, G, parse_config, data_config, True)
parser.parse()
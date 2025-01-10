from config.util import getEdgeSourceAndTargetIDs, getVertexByID, get_text_from_node, sumEdgeTypeDictionaries
from config.config import *
import math
import os
from argparse import ArgumentParser
from random import choice
import xml.etree.ElementTree as ET
from tqdm import tqdm
from glob import glob
import json

def general_statistics(input_path, preprocessed=False):
    if preprocessed:
        general_statistics_preprocessed(input_path)
    else:
        general_statistics_xml(input_path)

def general_statistics_preprocessed(input_path):
    filenames = glob(f"{input_path}**/*-G.json", recursive=True)
    num_graphs = len(filenames)
    num_nodes = 0
    num_edges = 0
    for filename in tqdm(filenames):
        graph = json.load(open(filename, "r"))
        num_nodes = num_nodes + len(graph["nodes"])
        num_edges = num_edges + len(graph["links"])
    print(f""" 
          Dataset from {input_path}
          Graphs: {num_graphs}
          Nodes: {num_nodes}
          Edges: {num_edges}
          Average edge per graph: {num_edges/num_graphs}
          """)

def general_statistics_xml(input_path):
    if os.path.isfile(input_path):
        filenames = input_path
    else:
        filenames = [x for x in os.listdir(input_path) if os.path.isfile(os.path.join(input_path,x)) and x.endswith(".xml")]
    num_graphs = len(filenames)
    num_nodes = 0
    num_edges = 0
    for filename in tqdm(filenames):
        tree = ET.parse(os.path.join(input_path, filename))
        root = tree.getroot()
        for element in root:
            if element.tag == "vertices":
                num_nodes = num_nodes + len(element)
            elif element.tag == "edges":
                num_edges = num_edges + len(element)
            else:
                continue
    print(f""" 
          Dataset from {input_path}
          Graphs: {num_graphs}
          Nodes: {num_nodes}
          Edges: {num_edges}
          Average edge per graph: {num_edges/num_graphs}
          """)

def getPingUMiLEdgeTypes(tree):
    dict_edge_types = {}

    root = tree.getroot()

    for element in root:
        if (element.tag == "edges"):
            for edge_element in element:
                source_id, target_id = getEdgeSourceAndTargetIDs(edge_element)
                source_vertex = getVertexByID(root,"vertex_{}".format(source_id))
                target_vertex = getVertexByID(root,"vertex_{}".format(target_id))
                target_label = get_text_from_node(target_vertex, "label")
                source_label = get_text_from_node(source_vertex, "label")

                edge_type = "{}->{}".format(target_label, source_label)
                if edge_type not in dict_edge_types:
                    dict_edge_types[edge_type] = 1
                else:
                    dict_edge_types[edge_type] = dict_edge_types[edge_type] + 1
    return dict_edge_types

def get_attribute_list(element):
    atb_list = []
    for attribute in element:
        atb_list.append(attribute.find("name").text)
    return atb_list

def get_node_attribute_sets(tree):
    node_atb_sets = []
    
    root = tree.getroot()
    for element in root:
        if element.tag != "vertices":
            continue
        for vertex_element in element:
            attribute_list = []
            for tag_element in vertex_element:
                if tag_element.tag != "attributes":
                    attribute_list.append(tag_element.tag)
                else:
                    attribute_list = attribute_list + get_attribute_list(tag_element)
            if attribute_list not in node_atb_sets:
                node_atb_sets.append(attribute_list)
    return node_atb_sets

def edge_sum_method(input_path):
    if os.path.isfile(input_path):
        tree = ET.parse(input_path)
        d_et = getPingUMiLEdgeTypes(tree)
    else:
        d_et = {}
        list_of_files = [x for x in os.listdir(input_path) if os.path.isfile(os.path.join(input_path,x)) and x.endswith(".xml")]
        print(list_of_files)
        for file in list_of_files:
            print(file)
            tree = ET.parse(os.path.join(input_path, file))
            td_et = getPingUMiLEdgeTypes(tree)
            d_et = sumEdgeTypeDictionaries(d_et, td_et)

    total_edges = 0
    for k,v in d_et.items():
        print("{}: {}".format(k,v))
        total_edges = total_edges + v
    print(total_edges)

def node_atb_sets_method(input_path):
    if os.path.isfile(input_path):
        trees = [ET.parse(input_path)]
    else:
        filenames = [x for x in os.listdir(input_path) if os.path.isfile(os.path.join(input_path,x)) and x.endswith(".xml")]
        trees = [ET.parse(os.path.join(input_path, filename)) for filename in filenames]
    node_atb_sets = []
    for tree in tqdm(trees):
        node_sets = get_node_attribute_sets(tree)
        #print(node_sets)
        node_atb_sets = node_atb_sets + [x for x in node_sets if x not in node_atb_sets]
    for node_atb_set in node_atb_sets:
        print(",".join(node_atb_set))
    
if __name__ == "__main__":
    parser = ArgumentParser("Set of methods for PinGML.")
    parser.add_argument("input_path", help="Path to file or directory")
    parser.add_argument("--preprocessed", action="store_true",
                        help="Checking for raw xml graph data or preprocessed graph data.")
    parser.add_argument("--statistics", action="store_true",
                        help="Get statistics for input graphs.")
    parser.add_argument("--edge_sum", action="store_true",
                        help="Get all edge types and counts them")
    parser.add_argument("--node_atb_sets", action="store_true",
                        help="Get all node types by attribute sets and counts them")
    args = parser.parse_args()

    input_path = args.input_path
    preprocessed = args.preprocessed if args.preprocessed else False

    if args.statistics:
        general_statistics(input_path, preprocessed)
    if args.edge_sum:
        edge_sum_method(input_path)
    if args.node_atb_sets:
        node_atb_sets_method(input_path)
        
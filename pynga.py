from config.util import getEdgeSourceAndTargetIDs, getVertexByID, getTextFromNode, sumEdgeTypeDictionaries
from config.config import *
from config.preprocess import *
import math
import os
from argparse import ArgumentParser
from random import choice
import xml.etree.ElementTree as ET
from tqdm import tqdm

def getPingUMiLEdgeTypes(tree):
    dictEdgeTypes = {}

    root = tree.getroot()

    for element in root:
        if (element.tag == "edges"):
            for edgeElement in element:
                sourceID, targetID = getEdgeSourceAndTargetIDs(edgeElement)
                sourceVertex = getVertexByID(root,"vertex_{}".format(sourceID))
                targetVertex = getVertexByID(root,"vertex_{}".format(targetID))
                targetLabel = getTextFromNode(targetVertex, "label")
                sourceLabel = getTextFromNode(sourceVertex, "label")

                edgeType = "{}->{}".format(targetLabel, sourceLabel)
                if edgeType not in dictEdgeTypes:
                    dictEdgeTypes[edgeType] = 1
                else:
                    dictEdgeTypes[edgeType] = dictEdgeTypes[edgeType] + 1
    return dictEdgeTypes

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
    parser.add_argument("--edge_sum", action="store_true",
                        help="Get all edge types and counts them")
    parser.add_argument("--node_atb_sets", action="store_true",
                        help="Get all node types by attribute sets and counts them")
    args = parser.parse_args()

    input_path = args.input_path

    if args.edge_sum:
        edge_sum_method(input_path)
    if args.node_atb_sets:
        node_atb_sets_method(input_path)
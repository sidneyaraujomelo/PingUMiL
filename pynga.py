from config.util import getEdgeSourceAndTargetIDs, getVertexByID, getTextFromNode, sumEdgeTypeDictionaries
from config.config import *
from config.preprocess import *
import math
import os
from argparse import ArgumentParser
from random import choice
import xml.etree.ElementTree as ET

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

if __name__ == "__main__":
    parser = ArgumentParser("Remove edges on Provenance data.")
    parser.add_argument("input_path", help="Path to file or directory")
    args = parser.parse_args()

    input_path = args.input_path

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
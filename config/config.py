import os
import errno
import json
import numpy as np
import random
import os
import pickle as pkl

from config.exp_config import *
from config.smokesquad_config import *


random.seed()

if (len(attrib_name_list) != len(attrib_type_list)):
    print("Attributes name and type list are not the same size.")
    print(len(attrib_name_list))
    print(len(attrib_type_list))
    print([x for x in zip(attrib_name_list,attrib_type_list)])

if (len(attrib_name_list) != len(attrib_default_value_list)):
    print("Attributes name and type list are not the same size.")
    print(len(attrib_name_list))
    print(len(attrib_default_value_list))
    print([x for x in zip(attrib_name_list,attrib_default_value_list)])

"""METHODS """
# Returns Graph file with integer id nodes
def get_output_graph_filename(output_suffix=None):
    token = output_prefix + (output_suffix if output_suffix else "")
    if kFoldGenerated:
        return output_folder + token + "/" + token + "-G.json"
    return output_folder + token + "-G.json"


# Returns Dictionary translating vertex id from provenance xml to vertex id in graph_file
def get_output_id_map_filename(output_suffix=None):
    token = output_prefix + (output_suffix if output_suffix else "")
    if kFoldGenerated:
        return output_folder + token + "/" + token + "-id_map.json"
    return output_folder + token + "-id_map.json"


# Returns Dictionary relating vertex id from provenance xml to classes (?)
def get_output_class_map_filename(output_suffix=None):
    token = output_prefix + (output_suffix if output_suffix else "")
    if kFoldGenerated:
        return output_folder + token + "/" + token + "-class_map.json"
    return output_folder + token + "-class_map.json"


# This one transforms featsmap dictionary into numpy matrixes (should be reviewed for more precision) and returns it
def get_output_feats_filename(output_suffix=None):
    token = output_prefix + (output_suffix if output_suffix else "")
    if kFoldGenerated:
        return output_folder + token + "/" + token + "-feats.npy"
    return output_folder + token + "-feats.npy"


#Defines the set of a node given a float number between 0 and 1
def get_set_node(a):
    if (a <= train_ratio):
        return train_dict
    elif (a <= train_ratio + test_ratio + 0.001):
        return test_dict
    return valid_dict

#Defines the set of a node given a float number between 0 and 1 for GAT data
def get_set_node_gat(a):
    if (a <= gat_test_ratio):
        return 'test'
    elif (a <= gat_test_ratio + gat_valid_ratio):
        return 'val'
    return 'none'

#Defines the set of a node given a float number between 0 and 1 for GAT data
def get_set_node_gat_multigraph(a):
    if (a <= train_ratio):
        return 'train'
    elif (a <= train_ratio + test_ratio + 0.001):
        return 'test'
    return 'val'


#Defines the set of an edge given a float number between 0 and 1
def get_set_edge(a):
    if (a <= train_ratio):
        return train_edge_dict
    elif (a <= train_ratio + test_ratio + 0.001):
        return test_edge_dict
    return valid_edge_dict


def findNodeWithTag(element, tag):
    if element.tag == tag:
        return element
    else:
        for child in element:
            return findNodeWithTag(child, tag)


def findAttributeNodeWithName(element, name):
    for attribute in element.iter("attribute"):
        if attribute.find('name').text == name:
            return attribute
    return None

def prepareDictionaries():
    # Dictionary of type Classes->{Class1, Class2, Class3}
    cat_attrib_list = []
    for i, atb_type in enumerate(attrib_type_list):
        if atb_type != 'categoric':
            continue
        sup_dict[attrib_name_list[i]] = []
        cat_attrib_list.append(attrib_name_list[i])
    for i, tag_type in enumerate(tag_type_list):
        if tag_type != 'categoric':
            continue
        sup_dict[tag_name_list[i]] = []
        cat_attrib_list.append(tag_name_list[i])
    return cat_attrib_list


def populateCategoricDictionaries(root,cat_attrib_list):
    # Traverse the tree until find vertices node
    vertices = findNodeWithTag(root, "vertices")
    # Iterates over vertex nodes
    for vertex in vertices.iter("vertex"):
        # Get tags of the vertex
        for cat_attrib in cat_attrib_list:
            # Find tag value for a given attribute name
            if (vertex.find(cat_attrib) != None):
                # If value exists
                tag_value = vertex.find(cat_attrib).text
                if (tag_value == None):
                    tag_value = tag_default_value_list[tag_name_list.index(cat_attrib)]
                if (tag_value not in sup_dict[cat_attrib]):
                    sup_dict[cat_attrib].append(tag_value)
        # Get attributes node
        attributes = vertex.find("attributes")
        # iterates over the list of wanted attributes
        for cat_attrib in cat_attrib_list:
            #Skip if it tries to search for a provenance tag
            if cat_attrib in tag_name_list:
                continue
            # Find attribute node for a given attribute name
            attrib_node = findAttributeNodeWithName(attributes, cat_attrib)
            if (attrib_node == None):
                current_node_atb_value = attrib_default_value_list[attrib_name_list.index(cat_attrib)]
            else:
                # Get the value of the attribute node for a given attribute name
                current_node_atb_value = attrib_node.find('value').text
            # Adds the value to the dictionary if it's not there yet
            if (current_node_atb_value not in sup_dict[cat_attrib]):
                sup_dict[cat_attrib].append(current_node_atb_value)


def buildOneHotVectorRepresentationForCategoricAttributes(cat_attrib_list):
    # Now we have all categories for every categoric attribute in the sup_dict
    # Iterate again over all categoric attribute
    for cat_attrib in cat_attrib_list:
        # Create a dictionary for every attribute
        category_representations = {}
        # Iterate over every possible value of the attribute, generating its 1-hot-vector representation
        for idx, category in enumerate(sup_dict[cat_attrib]):
            category_representations[category] = [0] * len(sup_dict[cat_attrib])
            category_representations[category][idx] = 1
        # Categoric_att_dict now points to every categoric attribute entry
        # And each Categoric Attribute Entry points to its 1-hot-vector representation
        categoric_att_dict[cat_attrib] = category_representations


def buildCategoricDictionary(root):
    print("Building Dictionary of Categoric Attributes and One-Hot-Vector representations")
    # Dictionary of type Classes->{Class1, Class2, Class3}
    cat_attrib_list = prepareDictionaries()

    populateCategoricDictionaries(root, cat_attrib_list)

    buildOneHotVectorRepresentationForCategoricAttributes(cat_attrib_list)

    print("One-hot-vector representations of every categoric attribute: ")
    print(categoric_att_dict)


def buildCategoricDictionaryForList(xmls):
    print("Building Dictionary of Categoric Attributes and One-Hot-Vector representations")
    cat_attrib_list = prepareDictionaries()

    # iterate over xml files
    for root in xmls:
        populateCategoricDictionaries(root, cat_attrib_list)

    buildOneHotVectorRepresentationForCategoricAttributes(cat_attrib_list)

    print("One-hot-vector representations of every categoric attribute: ")
    print(categoric_att_dict)


def getOneHotVectorForAttribute(attribute_name, attribute_value):
    #print(attribute_name)
    return categoric_att_dict[attribute_name][attribute_value]


def dictToNpArrayofArrays(featsmap):
    arr = np.array([v for k, v in featsmap.items()])
    print(arr)
    return arr

def createEdgeFile(filename):
    path = '/'.join(filename.split('/')[:-1])+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(filename, 'w')
    return f

#Common function for creating and writing Json to files!
def _createFile(filename, data):
    path = '/'.join(filename.split('/')[:-1])+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(filename, 'w')
    f.write(json.dumps(data))
    f.close()

#Common function for creating and writing Pkl to files!
def createPklFile(filename, data):
    path = '/'.join(filename.split('/')[:-1])+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(filename, 'wb')
    pkl.dump(data, f)  

#Create the Json Graph
def createOutputJsonGraph(json_data, a=None):
    _createFile(get_output_graph_filename(a), json_data)

#Create the Json Id Map
def createOutputJsonIdMap(idmap, a=None):
    _createFile(get_output_id_map_filename(a), idmap)

#Create the Json Class Map
def createOutputJsonClassMap(classmap, a=None):
    _createFile(get_output_class_map_filename(a), classmap)

#Create the numpy Features map
def createOutputJsonFeaturesMap(featsmap, a=None):
    # TRANSFORM FEATSMAP DICTIONARY INTO NUMPY MATRIXES
    npdata = dictToNpArrayofArrays(featsmap)
    filename = get_output_feats_filename(a) 
    f = open(filename, 'w')
    np.save(filename, npdata)

def createOutputJson(filename, data):
    _createFile(filename, data)

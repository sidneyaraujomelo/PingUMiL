import os
import xml.etree.ElementTree as ET
import torch

def generate_class_weights(x):
    if torch.is_tensor(x):
        n_samples = x.size(0)
        n_classes = x.size(1)
    elif type(x) == list:
        n_samples = len(x)
        n_classes = len(x[0])
    class_count = [0]*n_classes
    
    for v in x:
        for k in range(n_classes):
            if v[k] > 0:
                class_count[k] = class_count[k]+1

    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    return class_weights
    
def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_provenance_xml(file_path):
    e = ET.parse(file_path).getroot()
    return e

def find_node_with_tag(element, tag):
    if element.tag == tag:
        return element
    else:
        for child in element:
            return find_node_with_tag(child, tag)

def find_attribute_node_with_name(element, name):
    for attribute in element.iter("attribute"):
        if attribute.find('name').text == name:
            return attribute
    return None

def populate_categoric_dictionaries(root, graph_config, sup_dict):
    # Traverse the tree until find vertices node
    vertices = find_node_with_tag(root, "vertices")
    # Iterates over vertex nodes
    for vertex in vertices.iter("vertex"):
        # Get tags of the vertex
        for tag in graph_config["tags"]:
            if tag["type"] != "categoric":
                continue
            # Find tag value for a given attribute name
            if (vertex.find(tag["name"]) != None):
                # If value exists
                tag_value = vertex.find(tag["name"]).text
                if (tag_value == None):
                    tag_value = tag["default"]
                if (tag_value not in sup_dict[tag["name"]]):
                    sup_dict[tag["name"]].append(tag_value)
        # Get attributes node
        node_attributes = vertex.find("attributes")
        # iterates over the list of wanted attributes
        for attribute in graph_config["attributes"]:
            if attribute["type"] != "categoric":
                continue
            # Find attribute node for a given attribute name
            attrib_node = find_attribute_node_with_name(node_attributes,
                                                        attribute["name"])
            if (attrib_node == None):
                current_node_atb_value = attribute["default"]
            else:
                # Get the value of the attribute node for a given attribute name
                current_node_atb_value = attrib_node.find('value').text
            # Adds the value to the dictionary if it's not there yet
            if (current_node_atb_value not in sup_dict[attribute["name"]]):
                sup_dict[attribute["name"]].append(current_node_atb_value)

def build_onehot_categoric_attributes(cat_attrib_list, sup_dict):
    # Now we have all categories for every categoric attribute in the sup_dict
    # Iterate again over all categoric attribute
    encoded_att_dict = {}
    for cat_attrib in cat_attrib_list:
        # Create a dictionary for every attribute
        category_representations = {}
        # Iterate over every possible value of the attribute, generating its 1-hot-vector representation
        for idx, category in enumerate(sup_dict[cat_attrib]):
            category_representations[category] = [0] * len(sup_dict[cat_attrib])
            category_representations[category][idx] = 1
        # encoded_att_dict now points to every categoric attribute entry
        # And each Categoric Attribute Entry points to its 1-hot-vector representation
        encoded_att_dict[cat_attrib] = category_representations
    return encoded_att_dict

def build_categoric_dictionary(xmls, graph_config):
    print("Building Dictionary of Categoric Attributes and One-Hot-Vector representations")
    cat_attrib_list = {attribute["name"]:{} for attribute in graph_config["attributes"]+graph_config["tags"] if attribute["type"]=="categoric"}
    sup_dictionary = {attribute:[] for attribute in cat_attrib_list}

    # iterate over xml files
    for root in xmls:
        populate_categoric_dictionaries(root, graph_config, sup_dictionary)

    encoded_att_dict = build_onehot_categoric_attributes(cat_attrib_list,
                                                         sup_dictionary)

    print("One-hot-vector representations of every categoric attribute: ")
    print(encoded_att_dict)
    return encoded_att_dict

def loadExtraEdgeFiles(input_prefix, extraedgesfiles):
    if (len(extraedgesfiles)==0):
        extraedgesfiles = [f for f in os.listdir(input_prefix) if os.path.isfile(os.path.join(input_prefix,f)) and f.endswith('.txt')]
    extraedgesfiles.sort()
    print(extraedgesfiles)
    edgefiles = [open(input_prefix + filename) for filename in extraedgesfiles]
    return edgefiles, extraedgesfiles


""" Given a list of attributes, search all attribute nodes for
the attributes of that list """

def get_attributes_values(atbs_node, atbs_name_list):
    atbs_dict = {}
    # Iterates 'atribute' node
    for atb_node in atbs_node:
        # Get atribute name
        atb_name = atb_node.find("name").text;
        # Checks if name is on wanted attributes list
        if atb_name in atbs_name_list:
            atb_value = atb_node.find("value").text
            atbs_dict[atb_name] = atb_value
    return atbs_dict

""" Given a list of tags, search all tag values on a node for the tags of that list """
def get_tags_values(node, tag_name_list):
    tagsvalues_dict = {}
    #Iterates over tag_name_list and adds values to dictionary
    for tag_name in tag_name_list:
        #Get value of tag
        tagsvalues_dict[tag_name] = node.find(tag_name).text
    return tagsvalues_dict

"""Obtaining Node text value from first element with a given tag"""
def get_text_from_node(node, tag):
    a = node.find(tag).text
    return a

"""Obtaining Node text value from first element with a given tag"""
def setTextToNode(node, tag, new_text):
    node.find(tag).text = new_text

""" Given a root, obtains the number of nodes """
def getNumberOfNodes(e):
    for element in e:
        if (element.tag == "vertices"):
            return (len(element.findall("vertex")))
    return 0

""" Given a root, obtains the number of edges """
def getNumberOfEdges(e):
    for element in e:
        if (element.tag == "edges"):
            return (len(element.findall("edge")))
    return 0

""" Given a root, obtain a vertex based on its ID """
def getVertexByID(e, id_vertex):
    vertexes = e.find("vertices").findall("vertex")
    for vertex in vertexes:
        if (vertex.find("ID").text == id_vertex):
            return vertex
    return None

""" Given a vertex node, returns its ID (only the number) """
def getVertexIntID(node):
    return int(node.find("ID").text.split("_")[-1])

""" Given a vertex node, sets its ID (only the number) """
def setVertexIntID(node, id):
    prefix = node.find("ID").text.split("_")[0]
    node.find("ID").text = prefix+"_"+str(id)

""" Given an edge, returns its sourceID (only the number) """
def getEdgeSourceID(edge):
    return int(edge.find("sourceID").text.split("_")[-1])

""" Given an edge, returns its targetID (only the number) """
def getEdgeTargetID(edge):
    return int(edge.find("targetID").text.split("_")[-1])

""" Given an edge, returns both its sourceID and targetID (number only) """
def getEdgeSourceAndTargetIDs(edge):
    return getEdgeSourceID(edge), getEdgeTargetID(edge)

""" Merge two edge label dictionaries, putting everything of b in a and then returning a """
def mergeEdgeLabelDictionaries(a, b):
    for key,value in b.items():
        if key not in a:
            a[key] = value
        else:
            for v in value:
                if v not in a[key]:
                    a[key].append(v)
    return a

""" Sums two edge count dictionaries"""
def sumEdgeCountDictionaries(a,b):
    for source,targets_dict in b.items():
        if source not in a:
            a[source] = targets_dict
        else:
            for target, count in targets_dict.items():
                if target not in a[source]:
                    a[source][target] = count
                else:
                    a[source][target] = a[source][target]+count
    return a

""" Sums two edge type dictionaries"""
def sumEdgeTypeDictionaries(a,b):
    for edge_type,v in b.items():
        if edge_type not in a:
            a[edge_type] = v
        else:
            a[edge_type] = a[edge_type]+v
    return a

""" increments edge type counter in a edge type dictionary """
def incrementEdgeTypeInDictionary(d, source, target):
    if source not in d:
        d[source] = {}
        d[source][target] = 1
    elif target not in d[source]:
        d[source][target] = 1
    else:
        d[source][target] = d[source][target]+1
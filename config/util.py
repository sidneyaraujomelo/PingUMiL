import xml.etree.ElementTree as ET
import os

def createPath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def loadProvenanceXML(file_path):
    e = ET.parse(file_path).getroot()
    return e

def loadProvenanceXMLList(input_prefix, input_files, ignore_files=None):
    if ignore_files is None:
        ignore_files = []
    # If no specific input file is given, we should read all files from input_prefix
    if len(input_files) == 0:
        if isinstance(input_prefix, list):
            for input_prefix_folder in input_prefix:
                all_files = [f for f in os.listdir(input_prefix_folder)
                             if os.path.isfile(os.path.join(input_prefix_folder, f))]
                new_input_files = [f for f in all_files
                                   if f.endswith('.xml') and f not in ignore_files]
                new_input_files = [os.path.join(input_prefix_folder, f) for f in new_input_files]
                input_files = input_files + new_input_files
        else:
            input_files = [os.path.join(input_prefix_folder, f) for f in os.listdir(input_prefix)
                          if os.path.isfile(os.path.join(input_prefix, f))
                          and f.endswith('.xml') and f not in ignore_files]
    input_files.sort()
    print(input_files)
    xmls = [ET.parse(filepath).getroot() for filepath in input_files]
    input_files = [os.path.basename(x) for x in input_files]
    return xmls, input_files

def loadExtraEdgeFiles(input_prefix, extraedgesfiles):
    if (len(extraedgesfiles)==0):
        extraedgesfiles = [f for f in os.listdir(input_prefix) if os.path.isfile(os.path.join(input_prefix, f)) and f.endswith('.txt')]
    extraedgesfiles.sort()
    print(extraedgesfiles)
    edgefiles = [open(input_prefix + filename) for filename in extraedgesfiles]
    return edgefiles, extraedgesfiles


""" Given a list of attributes, search all attribute nodes for
the attributes of that list """

def get_attributes(atbs_node, atbs_name_list):
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
def get_tag_values(node, tag_name_list):
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
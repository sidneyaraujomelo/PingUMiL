import os
import pickle as pkl
import numpy as np
import json
from config.config import dictToNpArrayofArrays

# Returns Graph file with integer id nodes
def get_output_graph_filename(params, output_suffix=None):
    token = params["output_prefix"] + (output_suffix if output_suffix else "")
    if params["kFoldGenerated"]:
        return params["output_folder"] + token + "/" + token + "-G.json"
    return params["output_folder"] + token + "-G.json"


# Returns Dictionary translating vertex id from provenance xml to vertex id in graph_file
def get_output_id_map_filename(params, output_suffix=None):
    token = params["output_prefix"]+ (output_suffix if output_suffix else "")
    if params["kFoldGenerated"]:
        return params["output_folder"] + token + "/" + token + "-id_map.json"
    return params["output_folder"] + token + "-id_map.json"


# Returns Dictionary relating vertex id from provenance xml to classes (?)
def get_output_class_map_filename(params, output_suffix=None):
    token = params["output_prefix"]+ (output_suffix if output_suffix else "")
    if params["kFoldGenerated"]:
        return params["output_folder"] + token + "/" + token + "-class_map.json"
    return params["output_folder"] + token + "-class_map.json"

# Returns the name of the file that will contain the List of attribute sets
def get_output_atbset_list_filename(params, output_suffix=None):
    token = params["output_prefix"]+ (output_suffix if output_suffix else "")
    if params["kFoldGenerated"]:
        return params["output_folder"] + token + "/" + token + "-atbset_list.json"
    return params["output_folder"] + token + "-atbset_list.json"

# Returns the name of the file that will list the respective node id for each
# line in a given attribute set matrix (featmap).
def get_output_atbset_map_filename(params, output_suffix=None):
    token = params["output_prefix"]+ (output_suffix if output_suffix else "")
    if params["kFoldGenerated"]:
        return params["output_folder"] + token + "/" + token + "-map.json"
    return params["output_folder"] + token + "-map.json"

# This one transforms featsmap dictionary into numpy matrixes (should be reviewed for more precision) and returns it
def get_output_feats_filename(params, output_suffix=None):
    token = params["output_prefix"]+ (output_suffix if output_suffix else "")
    if params["kFoldGenerated"]:
        return params["output_folder"] + token + "/" + token + "-feats.npy"
    return params["output_folder"] + token + "-feats.npy"

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
def create_output_json_graph(json_data, a=None, params=None):
    _createFile(get_output_graph_filename(params, a), json_data)

#Create the Json Id Map
def create_output_json_idmap(idmap, a=None, params=None):
    _createFile(get_output_id_map_filename(params, a), idmap)

#Create the Json Class Map
def create_output_json_classmap(classmap, a=None, params=None):
    _createFile(get_output_class_map_filename(params, a), classmap)

#Create the Json AttributeSet List
def create_output_json_attributeset_list(atbsetmap, a=None, params=None):
    _createFile(get_output_atbset_list_filename(params, a), atbsetmap)

#Create the Json AttributeSet Map
def create_output_json_attributeset_map(atbsetmap, a=None, params=None):
    _createFile(get_output_atbset_map_filename(params, a), atbsetmap)

#Create the numpy Features map
def create_output_json_features_map(featsmap, a=None, params=None):
    # TRANSFORM FEATSMAP DICTIONARY INTO NUMPY MATRIXES
    npdata = dictToNpArrayofArrays(featsmap)
    filename = get_output_feats_filename(params, a) 
    f = open(filename, 'w')
    np.save(filename, npdata)

def createOutputJson(filename, data):
    _createFile(filename, data)
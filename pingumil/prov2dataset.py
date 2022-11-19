import json
import glob
import os
from argparse import ArgumentParser
import networkx as nx
from util.util import *
from pingdataset import PingDataset

def load_provenance_xml_files(prov_folder: str):
    """
    This method loads all provenance xml files found in the path and returns
    the root of the XMLs and the name of their original files
    """
    xml_files = glob.glob(f"{prov_folder}/*.xml")
    prov_xml_files = [load_provenance_xml(x) for x in xml_files]
    xml_filenames = [os.path.basename(x) for x in xml_files]
    return prov_xml_files, xml_filenames

def parse_xml2nx(pingdata, root, graph_config, categoric_dict,
                 current_id_count=0, original_file=""):
    tag_name_list = [tag["name"] for tag in graph_config["tags"]]
    attrib_name_list = [tag["name"] for tag in graph_config["attributes"]]
    for element in root:
        """ TAG VERTICES """
        if (element.tag == "vertices"):
            for vertex in element:
                features = []
                label = None
                """ VERTEX ID """
                node_id_text = get_text_from_node(vertex, "ID")
                original_node_id = int(node_id_text.split("_")[-1])
                node_id = original_node_id+current_id_count
                pingdata.g.add_node(node_id)
                pingdata.g.nodes[node_id]["o_id"] = original_node_id
                if original_file != "":
                    pingdata.g.nodes[node_id]["o_file"] = original_file

                # Time to iterate over Vertex Data 
                # Tag_val_dict contains the values for all tags in the current
                #node
                tag_val_dict = get_tags_values(vertex, tag_name_list)
                
                for tag in graph_config["tags"]:
                    tag_name = tag["name"]
                    tag_value = None
                    if tag["type"] == "numeric":    
                        if tag_name in tag_val_dict:
                            tag_value = float(tag_val_dict[tag_name])
                        else:
                            tag_value = float(tag["default"])
                        features.append(tag_value)
                    elif tag["type"] == "categoric":
                        if tag_name in tag_val_dict:
                            val = tag_val_dict[tag_name]
                        else:
                            val = tag["default"]
                        #The added value to the tag is the one-hot-encoding of "val"
                        tag_value = categoric_dict[tag_name][val]
                        if (tag_name == graph_config["label"]):
                            label = tag_value
                        else:
                            features = features + tag_value

                # Time to iterate over Attributes Data 
                # attrib_val_dict contains the values for all tags in the current
                #node
                attrib_node = vertex.find("attributes")
                attrib_val_dict = get_attributes_values(attrib_node,
                                                        attrib_name_list)
                for atb in graph_config["attributes"]:
                    atb_name = atb["name"]
                    if atb["type"] == "numeric":    
                        if atb_name in attrib_val_dict:
                            atb_value = float(attrib_val_dict[atb_name])
                        else:
                            atb_value = float(atb["default"])
                        features.append(atb_value)
                    elif atb["type"] == "categoric":
                        if atb_name in attrib_val_dict:
                            val = attrib_val_dict[atb_name]
                        else:
                            val = atb["default"]
                        #The added value to the tag is the one-hot-encoding of "val"
                        atb_value = categoric_dict[atb_name][val]
                        if (atb_name == graph_config["label"]):
                            label = atb_value
                        else:
                            features = features + atb_value
                pingdata.featmap[node_id] = features
                pingdata.labelmap[node_id] = label

        # XML TAG EDGES
        elif (element.tag == "edges"):
            #Let's iterate over edge elements
            for edge in element: 
                # Get source vertex
                source_node = get_text_from_node(edge, "sourceID")
                source_node = int(source_node.split("_")[-1]) + current_id_count

                # Get target vertex
                target_node = get_text_from_node(edge, "targetID")
                target_node = int(target_node.split("_")[-1]) + current_id_count

                # Add edges
                pingdata.g.add_edge(source_node, target_node)
    return pingdata

def prov_list_to_nx(roots, graph_config, categoric_dict, files):
    pingdata = PingDataset()
    current_id_count = 0
    for i, root in enumerate(roots):
        current_id_count = len(pingdata.g.nodes)
        pingdata = parse_xml2nx(pingdata, root, graph_config, categoric_dict,
                                current_id_count, files[i])
    return pingdata

def main():
    argparser = ArgumentParser()
    argparser.add_argument("prov_folder", type=str, help="Path to graphs folder")
    argparser.add_argument("graph_config", type=str, help="Path to graph json descriptor")
    argparser.add_argument("--join", action="store_true")

    parser = argparser.parse_args()
    if not os.path.exists(parser.prov_folder):
        print(f"{parser.prov_folder} does not exist!")
    if not os.path.exists(parser.graph_config):
        print(f"{parser.graph_config} does not exist!")
    elif not os.path.isfile(parser.graph_config):
        print(f"{parser.graph_config} is not a file!")

    graph_config = json.load(open(parser.graph_config))

    prov_trees, prov_files = load_provenance_xml_files(parser.prov_folder)
    categoric_dict = build_categoric_dictionary(prov_trees, graph_config)

    #print(categoric_dict)
    if parser.join:
        pingdata = prov_list_to_nx(prov_trees, graph_config, categoric_dict,
                                   prov_files)
        pingdata.save_to(os.path.join(parser.prov_folder,"..","pingdataset"))
    else:
        for i, prov in enumerate(prov_trees):
            pingdata = PingDataset()
            pingdata = parse_xml2nx(pingdataset, prov, graph_config,
                                    categoric_dict, original_file=prov_files[i])
            pingdata.save_to(os.path.join(parser.prov_folder,"..","pingdatasets"))

if __name__=="__main__":
    main()
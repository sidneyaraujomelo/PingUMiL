import networkx as nx
from collections import OrderedDict
import random
from config.util import get_text_from_node, get_tag_values, get_attributes
from config.config import get_ohv_for_attribute, find_node_with_tag, find_attribute_node_with_name
from config.graph_splitter import NodeSplitter, EdgeSplitter
from parsedgraphexporter import ParsedGraphExporter

def writeAttributeToGraph(G, node_id, attrib_name, attrib_val, attrib_written_in_node=False):
    if (attrib_written_in_node == True):
        G.nodes[node_id][attrib_name] = attrib_val

class ProvHnxParser():
    def __init__(self, provs, g, parse_config, data_config, use_graph_name=False):
        if type(provs) == list:
            self.provs = provs
        else:
            self.provs = [provs]
        self.num_provs = len(self.provs)
        self.g = g
        self.parse_config = parse_config
        self.data_config = data_config
        self.use_graph_name = use_graph_name
        self.node_atb_sets = []
        self.idmap = OrderedDict()
        self.classmap = OrderedDict()
        self.featmap = OrderedDict()
        self.categoric_att_dict = {}
        self.sup_dict = {}
        self.build_categoric_dictionary_for_list()
        self.graph_exporter = ParsedGraphExporter(self.parse_config)
        self.node_splitter = NodeSplitter(**self.parse_config["node_split"])
        self.edge_splitter = EdgeSplitter(**self.parse_config["edge_split"])
    
    def prepare_dictionaries(self):
        attrib_type_list = self.data_config["attrib_type_list"]
        attrib_name_list = self.data_config["attrib_name_list"]
        tag_type_list = self.data_config["tag_type_list"]
        tag_name_list = self.data_config["tag_name_list"]
        # Dictionary of type Classes->{Class1, Class2, Class3}
        cat_attrib_list = []
        for i, atb_type in enumerate(attrib_type_list):
            if atb_type != 'categoric':
                continue
            self.sup_dict[attrib_name_list[i]] = []
            cat_attrib_list.append(attrib_name_list[i])
        for i, tag_type in enumerate(tag_type_list):
            if tag_type != 'categoric':
                continue
            self.sup_dict[tag_name_list[i]] = []
            cat_attrib_list.append(tag_name_list[i])
        return cat_attrib_list

    def populate_categoric_dictionaries(self, root, cat_attrib_list):
        tag_name_list = self.data_config["tag_name_list"]
        tag_default_value_list = self.data_config["tag_default_value_list"]
        attrib_name_list = self.data_config["attrib_name_list"]
        attrib_default_value_list = self.data_config["attrib_default_value_list"]
        # Traverse the tree until find vertices node
        vertices = find_node_with_tag(root, "vertices")
        # Iterates over vertex nodes
        for vertex in vertices.iter("vertex"):
            # Get tags of the vertex
            for cat_attrib in cat_attrib_list:
                # Find tag value for a given attribute name
                if vertex.find(cat_attrib) is not None:
                    # If value exists
                    tag_value = vertex.find(cat_attrib).text
                    if tag_value is None:
                        tag_value = tag_default_value_list[tag_name_list.index(cat_attrib)]
                    if tag_value not in self.sup_dict[cat_attrib]:
                        self.sup_dict[cat_attrib].append(tag_value)
            # Get attributes node
            attributes = vertex.find("attributes")
            # iterates over the list of wanted attributes
            for cat_attrib in cat_attrib_list:
                #Skip if it tries to search for a provenance tag
                if cat_attrib in tag_name_list:
                    continue
                # Find attribute node for a given attribute name
                attrib_node = find_attribute_node_with_name(attributes, cat_attrib)
                if (attrib_node is None):
                    current_node_atb_value = attrib_default_value_list[attrib_name_list.index(cat_attrib)]
                else:
                    # Get the value of the attribute node for a given attribute name
                    current_node_atb_value = attrib_node.find('value').text
                # Adds the value to the dictionary if it's not there yet
                if (current_node_atb_value not in self.sup_dict[cat_attrib]):
                    self.sup_dict[cat_attrib].append(current_node_atb_value)

    def build_ohv_representation_for_categoric_attributes(self, cat_attrib_list):
        # Now we have all categories for every categoric attribute in the sup_dict
        # Iterate again over all categoric attribute
        for cat_attrib in cat_attrib_list:
            # Create a dictionary for every attribute
            category_representations = {}
            # Iterate over every possible value of the attribute, generating its 1-hot-vector representation
            for idx, category in enumerate(self.sup_dict[cat_attrib]):
                category_representations[category] = [0] * len(self.sup_dict[cat_attrib])
                category_representations[category][idx] = 1
            # Categoric_att_dict now points to every categoric attribute entry
            # And each Categoric Attribute Entry points to its 1-hot-vector representation
            self.categoric_att_dict[cat_attrib] = category_representations

    def build_categoric_dictionary_for_list(self):
        print("Building Dictionary of Categoric Attributes and One-Hot-Vector representations")
        cat_attrib_list = self.prepare_dictionaries()

        # iterate over xml files
        for root in self.provs:
            self.populate_categoric_dictionaries(root, cat_attrib_list)

        self.build_ohv_representation_for_categoric_attributes(cat_attrib_list)

        print("One-hot-vector representations of every categoric attribute: ")
        print(self.categoric_att_dict)
    
    def __parseVertex(self, vertex, current_prov):
        attrib_written_in_node = self.parse_config["attrib_written_in_node"]
        attrib_type_list = self.data_config["attrib_type_list"]
        attrib_name_list = self.data_config["attrib_name_list"]
        tag_type_list = self.data_config["tag_type_list"]
        tag_name_list = self.data_config["tag_name_list"]
        tag_default_value_list = self.data_config["tag_default_value_list"]
        label_attrib_name = self.data_config["label_attrib_name"]
        label_conditions = self.data_config["label_conditions"]

        """ VERTEX ID """
        node_id = get_text_from_node(vertex, "ID")
        node_id = int(node_id.split("_")[-1])
        node_id_int = len(self.idmap)
        self.idmap[node_id_int] = node_id_int

        self.g.add_node(node_id_int)

        node_set = self.node_splitter.get_set_element(random.random())
        #print(node_set)
        # Iterate over the array 'node_set' containing bool values
        # for "test" and "val"
        for idx, value in node_set.items():
            self.g.nodes[node_id_int][idx] = value

        # This information is added so that we know the origin of
        # this node
        self.g.nodes[node_id_int]['origin'] = current_prov

        feature_name = []
        feature_values = []
        label = None

        # Time to iterate over Vertex Data
        tag_val_dict = get_tag_values(vertex, tag_name_list)
        for idx, tag_name in enumerate(tag_name_list):
            if (tag_name == label_attrib_name):
                if(tag_val_dict[tag_name] != None):
                    tag_value = tag_val_dict[tag_name]
                    label = get_ohv_for_attribute(self.categoric_att_dict,
                                                  tag_name,
                                                  tag_value)
                    writeAttributeToGraph(self.g, node_id_int,
                                          tag_name, tag_value, attrib_written_in_node)
                else:
                    default_value = tag_default_value_list[
                        tag_name_list.index(tag_name)]
                    label = get_ohv_for_attribute(self.categoric_att_dict,
                                                  tag_name,
                                                  default_value)
                    writeAttributeToGraph(self.g, node_id_int,
                                            tag_name,
                                            default_value,
                                            attrib_written_in_node)
                continue
            if (tag_type_list[idx] == 'numeric'):
                if (tag_name in tag_val_dict):
                    tag_value = float(
                        tag_val_dict[tag_name].replace(',','.'))
                    feature_name.append(tag_name)
                    feature_values.append(tag_value)
                    writeAttributeToGraph(self.g, node_id_int,
                                            tag_name, tag_value, attrib_written_in_node)
            elif (tag_type_list[idx] == 'categoric'):
                tag_value = tag_val_dict[tag_name]
                onehotvectorrep = get_ohv_for_attribute(self.categoric_att_dict,
                                                        tag_name,
                                                        tag_value)
                writeAttributeToGraph(self.g, node_id_int,
                                        tag_name, tag_value, attrib_written_in_node)
                for i, x in enumerate(onehotvectorrep):
                    feature_values.append(float(x))  
                    feature_name.append(f"{tag_name}_{i}")
        """ Time to iterate over Attributes """
        attrib_node = vertex.find("attributes")
        attrib_val_dict = get_attributes(attrib_node, attrib_name_list)
        for idx, attrib_name in enumerate(attrib_name_list):
            if (attrib_name == label_attrib_name):
                attrib_value = attrib_val_dict[attrib_name]
                label = get_ohv_for_attribute(self.categoric_att_dict,
                                              label_attrib_name,
                                              attrib_value)
                writeAttributeToGraph(self.g, node_id_int,
                                        label_attrib_name,
                                        attrib_value, attrib_written_in_node)
                continue
            if (attrib_type_list[idx] == 'numeric'):
                if (attrib_name in attrib_val_dict):
                    attrib_value = float(
                        attrib_val_dict[attrib_name].replace(',','.'))
                    feature_values.append(attrib_value)
                    feature_name.append(attrib_name)
                    writeAttributeToGraph(self.g, node_id_int,
                                            attrib_name, attrib_value, attrib_written_in_node)
                #else:
                #    features.append(attrib_default_value_list[idx])
                #    writeAttributeToGraph(G, node_id_int, attrib_name, attrib_default_value_list[idx])
            elif (attrib_type_list[idx] == 'categoric'):
                onehotvectorrep = []
                if attrib_name in attrib_val_dict:
                    attrib_value = attrib_val_dict[attrib_name]
                    onehotvectorrep = get_ohv_for_attribute(self.categoric_att_dict,
                                                            attrib_name,
                                                            attrib_value)
                    writeAttributeToGraph(self.g, node_id_int,
                                            attrib_name, attrib_value, attrib_written_in_node)
                #else:
                    #print(attrib_default_value_list)
                    #onehotvectorrep = get_ohv_for_attribute(attrib_name, attrib_default_value_list[idx])
                    #writeAttributeToGraph(G, node_id_int, attrib_name, attrib_default_value_list[idx])
                for i, x in enumerate(onehotvectorrep):
                    feature_values.append(float(x))
                    feature_name.append(f"{attrib_name}_{i}")
                    # print(features)

                    # G.node[node_id_int]['feature'] = features
        # Now, we have both feature values in feature_values and the
        # set of features names in feature_name. The feature names
        # becomes a set which will be identifier of a dictionary of
        # featmaps for a give node attribute set (type).
        # Each featmap maps the node id of type X to its feature
        # values.
        if feature_name not in self.node_atb_sets:
            self.node_atb_sets.append(feature_name)
            atb_set_id = len(self.node_atb_sets)-1
            self.featmap[atb_set_id] = OrderedDict()
        else:
            atb_set_id = self.node_atb_sets.index(feature_name)
        self.featmap[atb_set_id][node_id_int] = feature_values
        # First, we check if label condition is set. If yes, we check if this node
        # should be included in classmap.
        if label_conditions is not None and len(label_conditions) > 0:
            all_cond_satisf = True
            for k, v in label_conditions.items():
                if k in tag_name_list:
                    if tag_val_dict[k] != v:
                        all_cond_satisf = False
                elif k in attrib_name_list:
                    if attrib_val_dict[k] != v:
                        all_cond_satisf = False
            if all_cond_satisf:
                if label is None and label_attrib_name == 'function':
                    # to be implemented
                    label_func = lambda x:x
                    try:
                        label = label_func(self.g.graph['name'])
                    except KeyError:
                        label = None
                    print(f"Current file: {self.g.graph['name']}, current label: {label}")
                    #assert 1==2
                self.classmap[node_id_int] = label
    
    def __parseEdge(self, edge, current_prov):
        # Get source vertex
        source_node = get_text_from_node(edge, "sourceID")
        source_node = int(source_node.split("_")[-1]) #+ currentIdCount

        # Get target vertex
        target_node = get_text_from_node(edge, "targetID")
        target_node = int(target_node.split("_")[-1]) #+ currentIdCount

        self.g.add_edge(source_node, target_node)

        # Now it's time to randomly choose if the edge is going to be used for training, test or validation
        # We use 70% training, 20% test and 10% validation in our experiments, this can be changed in config.
        if (self.num_provs == 1):
            edge_set = self.edge_splitter.get_set_element(random.random())
        else:
            edge_set = self.edge_splitter.get_set_element(current_prov/self.num_provs)
        #print(node_set)
        # Iterate over the array 'edge_set' containing bool values for "test" and "val"
        for idx, value in edge_set.items():
            #print(G.graph['name'])
            self.g.edges[source_node,target_node][idx] = value
    
    def __parseXML(self, root, current_id_count=0, current_prov=0):
        #print(root)
        for element in root:
            """ TAG VERTICES """
            if (element.tag == "vertices"):
                for vertex in element:
                    self.__parseVertex(vertex, current_prov)
                    
            # TAG EDGES
            elif (element.tag == "edges"):
                for edge in element:
                    self.__parseEdge(edge, current_prov)

    def parse(self):
        attrib_name_list = self.data_config["attrib_name_list"]
        current_id_count = 0
        print(attrib_name_list)
        for current_prov, root in enumerate(self.provs):
            current_id_count = len(self.idmap)
            print(current_id_count) 
            self.__parseXML(root, current_id_count, current_prov)
        for node_atb_set in self.node_atb_sets:
            print(node_atb_set)
        print(len(self.featmap))
        for i, fmap in self.featmap.items():
            print(f"{i}:{len(fmap)}")

    def save(self):
        output_prefix = None
        if self.use_graph_name:
            output_prefix = self.g.graph['name']
        # Write output json Graph
        print("Writing Json Graph File")
        self.graph_exporter.export(output_prefix, self)
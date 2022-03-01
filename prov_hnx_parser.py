from config.util import *
from config.config import *
import networkx as nx
from collections import OrderedDict

def writeAttributeToGraph(G, node_id, attrib_name, attrib_val):
    if (attrib_written_in_node == True):
        G.nodes[node_id][attrib_name] = attrib_val

class ProvHnxParser():
    def __init__(self, provs, g, use_graph_name=False):
        if type(provs) == list:
            self.provs = provs
        else:
            self.provs = [provs]
        self.num_provs = len(self.provs)
        self.g = g
        self.use_graph_name = use_graph_name
        self.node_atb_sets = []
        self.idmap = OrderedDict()
        self.classmap = OrderedDict()
        self.featmap = OrderedDict()
    
    def __parseVertex(self, vertex, current_prov):
        """ VERTEX ID """
        node_id = getTextFromNode(vertex, "ID")
        node_id = int(node_id.split("_")[-1])
        node_id_int = len(self.idmap)
        self.idmap[node_id_int] = node_id_int

        self.g.add_node(node_id_int)

        node_set = get_set_node(random.random())
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
        tag_val_dict = getTagValues(vertex, tag_name_list)
        for idx, tag_name in enumerate(tag_name_list):
            if (tag_name == label_attrib_name):
                if(tag_val_dict[tag_name] != None):
                    tag_value = tag_val_dict[tag_name]
                    label = getOneHotVectorForAttribute(tag_name,
                                                        tag_value)
                    writeAttributeToGraph(self.g, node_id_int,
                                            tag_name, tag_value)
                else:
                    default_value = tag_default_value_list[
                        tag_name_list.index(tag_name)]
                    label = getOneHotVectorForAttribute(tag_name,
                                                        default_value)
                    writeAttributeToGraph(self.g, node_id_int,
                                            tag_name,
                                            default_value)
                continue
            if (tag_type_list[idx] == 'numeric'):
                if (tag_name in tag_val_dict):
                    tag_value = float(
                        tag_val_dict[tag_name].replace(',','.'))
                    feature_name.append(tag_name)
                    feature_values.append(tag_value)
                    writeAttributeToGraph(self.g, node_id_int,
                                            tag_name, tag_value)
            elif (tag_type_list[idx] == 'categoric'):
                tag_value = tag_val_dict[tag_name]
                onehotvectorrep = getOneHotVectorForAttribute(
                    tag_name, tag_value)
                writeAttributeToGraph(self.g, node_id_int,
                                        tag_name, tag_value)
                for i, x in enumerate(onehotvectorrep):
                    feature_values.append(float(x))  
                    feature_name.append(f"{tag_name}_{i}")
        """ Time to iterate over Attributes """
        attrib_node = vertex.find("attributes")
        attrib_val_dict = getAttributes(attrib_node, attrib_name_list)
        for idx, attrib_name in enumerate(attrib_name_list):
            if (attrib_name == label_attrib_name):
                attrib_value = attrib_val_dict[attrib_name]
                label = getOneHotVectorForAttribute(label_attrib_name,
                                                    attrib_value)
                writeAttributeToGraph(self.g, node_id_int,
                                        label_attrib_name,
                                        attrib_value)
                continue
            if (attrib_type_list[idx] == 'numeric'):
                if (attrib_name in attrib_val_dict):
                    attrib_value = float(
                        attrib_val_dict[attrib_name].replace(',','.'))
                    feature_values.append(attrib_value)
                    feature_name.append(attrib_name)
                    writeAttributeToGraph(self.g, node_id_int,
                                            attrib_name, attrib_value)
                #else:
                #    features.append(attrib_default_value_list[idx])
                #    writeAttributeToGraph(G, node_id_int, attrib_name, attrib_default_value_list[idx])
            elif (attrib_type_list[idx] == 'categoric'):
                onehotvectorrep = []
                if attrib_name in attrib_val_dict:
                    attrib_value = attrib_val_dict[attrib_name]
                    onehotvectorrep = getOneHotVectorForAttribute(attrib_name,
                                                                    attrib_value)
                    writeAttributeToGraph(self.g, node_id_int,
                                            attrib_name, attrib_value)
                #else:
                    #print(attrib_default_value_list)
                    #onehotvectorrep = getOneHotVectorForAttribute(attrib_name, attrib_default_value_list[idx])
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
        self.classmap[node_id_int] = label
    
    def __parseEdge(self, edge, current_prov):
        # Get source vertex
        source_node = getTextFromNode(edge, "sourceID")
        source_node = int(source_node.split("_")[-1]) #+ currentIdCount

        # Get target vertex
        target_node = getTextFromNode(edge, "targetID")
        target_node = int(target_node.split("_")[-1]) #+ currentIdCount

        self.g.add_edge(source_node, target_node)

        # Now it's time to randomly choose if the edge is going to be used for training, test or validation
        # We use 70% training, 20% test and 10% validation in our experiments, this can be changed in config.
        if (self.num_provs==1):
            node_set=get_set_edge(random.random())
        else:
            node_set = get_set_edge(current_prov/self.num_provs)
        #print(node_set)
        # Iterate over the array 'node_set' containing bool values for "test" and "val"
        for idx, value in node_set.items():
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
        json_data = nx.readwrite.json_graph.node_link_data(self.g)
        createOutputJsonGraph(json_data, output_prefix)

        # Write id map json
        print("Writing Json Id Map File")
        createOutputJsonIdMap(self.idmap, output_prefix)

        # Write class_map json
        print("Writing Json Class Map File")
        createOutputJsonClassMap(self.classmap, output_prefix)

        # Write node attribute sets json
        print("Writing Json Node Attribute Sets File")
        createOutputJsonAttributeSetList(self.node_atb_sets, output_prefix)
        
        #Write feature numpy per attribute sets
        for atb_set_id, featmap in self.featmap.items():
            createOutputJsonAttributeSetMap(list(featmap.keys()), f"{output_prefix}_atbset_{atb_set_id}")
            createOutputJsonFeaturesMap(featmap, f"{output_prefix}_atbset_{atb_set_id}")
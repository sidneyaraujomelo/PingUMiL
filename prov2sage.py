from config.util import *

from config.config import *

import networkx as nx

# GLOBAL VARIABLES
idmap = {}
classmap = {}
featmap = {}

def writeAttributeToGraph(G, node_id, attrib_name, attrib_val):
    if (attrib_written_in_node == True):
        G.node[node_id][attrib_name] = attrib_val

def parseXML2nx(e, G, currentIdCount=0, currentXml=1, numXmls=1):
    print(currentIdCount)
    for element in e:
        """ TAG VERTICES """
        if (element.tag == "vertices"):
            for vertex in element:

                """ VERTEX ID """
                node_id = get_text_from_node(vertex, "ID")
                node_id = int(node_id.split("_")[-1])

                node_id_int = len(idmap)
                idmap[node_id_int] = node_id_int

                if (node_id != node_id_int):
                    print("Node_id diferente de node_id_int, erro de consistencia!")
                    return -1
                # print("Criou node id: "+str(node_id))
                G.add_node(node_id_int)

                # Now it's time to randomly choose if the node is going to be used for training, test or validation
                # We use 70% training, 20% test and 10% validation in our experiments, this can be changed in config.
                """if (numXmls==1):
                    node_set=get_set_node(random.random())
                else:
                    #If this is the last graph, then it should be out of training -> all its nodes should be validation nodes
                    if (leaveOneGraphOutOfTraining and currentXml == numXmls):
                        node_set=get_set_node(2)
                    else:
                        node_set = get_set_node(random.random())"""
                node_set = get_set_node(random.random())
                #print(node_set)
                # Iterate over the array 'node_set' containing bool values for "test" and "val"
                for idx, value in node_set.items():
                    G.node[node_id_int][idx] = value

                #This information is added so that we know the origin of this node
                G.node[node_id_int]['origin'] = currentXml

                features = []
                label = None

                """ Time to iterate over Vertex Data """
                tag_val_dict = get_tag_values(vertex, tag_name_list)
                for idx, tag_name in enumerate(tag_name_list):
                    if (tag_name == label_attrib_name):
                        if(tag_val_dict[tag_name] != None):
                            label = get_ohv_for_attribute(tag_name, tag_val_dict[tag_name])
                            writeAttributeToGraph(G, node_id_int, tag_name, tag_val_dict[tag_name])
                        else:
                            label = get_ohv_for_attribute(tag_name, tag_default_value_list[tag_name_list.index(tag_name)])
                            writeAttributeToGraph(G, node_id_int, tag_name, tag_default_value_list[tag_name_list.index(tag_name)])
                        continue
                    if (tag_type_list[idx] == 'numeric'):
                        if (tag_name in tag_val_dict):
                            features.append(float(tag_val_dict[tag_name].replace(',','.')))
                            writeAttributeToGraph(G, node_id_int, tag_name, tag_val_dict[tag_name])
                        else:
                            features.append(tag_default_value_list[idx])
                            writeAttributeToGraph(G, node_id_int, tag_name, tag_default_value_list[idx])
                    elif (tag_type_list[idx] == 'categoric'):
                        onehotvectorrep = get_ohv_for_attribute(tag_name, tag_val_dict[tag_name]) 
                        writeAttributeToGraph(G, node_id_int, tag_name, tag_val_dict[tag_name])
                        for x in onehotvectorrep:
                            features.append(float(x))  
                """ Time to iterate over Attributes """
                attrib_node = vertex.find("attributes")
                attrib_val_dict = get_attributes(attrib_node, attrib_name_list)
                for idx, attrib_name in enumerate(attrib_name_list):
                    if (attrib_name == label_attrib_name):
                        label = get_ohv_for_attribute(label_attrib_name, attrib_val_dict[attrib_name])
                        writeAttributeToGraph(G, node_id_int, label_attrib_name, attrib_val_dict[attrib_name])
                        continue
                    if (attrib_type_list[idx] == 'numeric'):
                        if (attrib_name in attrib_val_dict):
                            features.append(float(attrib_val_dict[attrib_name].replace(',','.')))
                            writeAttributeToGraph(G, node_id_int, attrib_name, attrib_val_dict[attrib_name])
                        else:
                            features.append(attrib_default_value_list[idx])
                            writeAttributeToGraph(G, node_id_int, attrib_name, attrib_default_value_list[idx])
                    elif (attrib_type_list[idx] == 'categoric'):
                        onehotvectorrep = []
                        if attrib_name in attrib_val_dict:
                            onehotvectorrep = get_ohv_for_attribute(attrib_name, attrib_val_dict[attrib_name])
                            writeAttributeToGraph(G, node_id_int, attrib_name, attrib_val_dict[attrib_name])
                        else:
                            #print(attrib_default_value_list)
                            onehotvectorrep = get_ohv_for_attribute(attrib_name, attrib_default_value_list[idx])
                            writeAttributeToGraph(G, node_id_int, attrib_name, attrib_default_value_list[idx])
                        for x in onehotvectorrep:
                            features.append(float(x))
                            # print(features)

                            # G.node[node_id_int]['feature'] = features

                featmap[node_id_int] = features
                classmap[node_id_int] = label

        # TAG EDGES
        elif (element.tag == "edges"):
            for edge in element:
                # Get source vertex
                source_node = get_text_from_node(edge, "sourceID")
                source_node = int(source_node.split("_")[-1]) #+ currentIdCount

                # Get target vertex
                target_node = get_text_from_node(edge, "targetID")
                target_node = int(target_node.split("_")[-1]) #+ currentIdCount

                G.add_edge(source_node, target_node)

                # Now it's time to randomly choose if the edge is going to be used for training, test or validation
                # We use 70% training, 20% test and 10% validation in our experiments, this can be changed in config.
                if (numXmls==1):
                    node_set=get_set_edge(random.random())
                else:
                    node_set = get_set_edge(currentXml/numXmls)
                #print(node_set)
                # Iterate over the array 'node_set' containing bool values for "test" and "val"
                for idx, value in node_set.items():
                    #print(G.graph['name'])
                    G.edges[source_node,target_node][idx] = value

def prov2nx(e, G):
    parseXML2nx(e,G,0,1,1)    

def edgesFileToList(edgefile, currentIdCount):
    edgesValues = []
    for line in edgefile:
        source = int(line.split(" ")[0])+currentIdCount
        target = int(line.split(" ")[1])+currentIdCount
        edgesValues.append([source, target])
    return edgesValues

def writeNewEdgeFileToList(edgesValues, output_filename):
    output = createEdgeFile(output_folder+output_filename)
    for edges in edgesValues:
        output.write("{} {}\n".format(edges[0], edges[1]))
    output.close() 

def provList2nx(xmls, G, extraedges=None, negativeedges=None):
    currentIdCount = 0
    currentXml = 1
    numXmls = len(xmls)
    edgesValues = []
    negativeEdgesValues = []

    if (extraedges == None and negativeedges == None):
        for e in xmls:
            currentIdCount = len(idmap)
            print(currentIdCount)
            parseXML2nx(e,G,currentIdCount, currentXml, numXmls)
            currentXml = currentXml+1  
    elif (extraedges != None and negativeedges == None):
        for e in xmls:
            currentIdCount = len(idmap)
            #edgesOfCurrentGraph = []
            print("{} {}".format(currentIdCount,extraedges[currentXml-1]))
            edgesValues = edgesValues+edgesFileToList(extraedges[currentXml-1], currentIdCount)
            parseXML2nx(e,G,currentIdCount, currentXml, numXmls)
            currentXml = currentXml+1

        #Write new edgeValues
        writeNewEdgeFileToList(edgesValues, "extra_edges.txt")   
    elif (extraedges != None and negativeedges != None):
        for e in xmls:
            currentIdCount = 0
            #edgesOfCurrentGraph = []
            #print("{} {}".format(currentIdCount,extraedges[currentXml-1]))
            if (includeExtraEdges):
                edgesValues = edgesValues+edgesFileToList(extraedges[currentXml-1], currentIdCount)
            if (includeNegativeEdges):
                negativeEdgesValues = negativeEdgesValues+edgesFileToList(negativeedges[currentXml-1], currentIdCount)
            parseXML2nx(e,G,currentIdCount, currentXml, numXmls)
            currentXml = currentXml+1

        #Write new edgeValues 
        if (includeExtraEdges):
            writeNewEdgeFileToList(edgesValues, "extra_edges.txt")

        #Write new negativeEdgeValues
        if (includeNegativeEdges):
            writeNewEdgeFileToList(negativeEdgesValues, "negative_edges.txt")        


def outputJsonGraphFiles(G, a):
    # Write output json Graph
    print("Writing Json Graph File")
    json_data = nx.readwrite.json_graph.node_link_data(G)
    create_output_json_graph(json_data,a)

    # Write id map json
    print("Writing Json Id Map File")
    create_output_json_idmap(idmap,a)

    # Write class_map json
    print("Writing Json Class Map File")
    create_output_json_classmap(classmap,a)

    # Write class_map json
    print("Writing Json Features Map File")
    create_output_json_features_map(featmap,a)

def main(argv=None):
    

    if includeExtraEdges:
        print("Loading extra edge files")
        edgesFiles = loadExtraEdgeFiles(input_prefix, extraedgesfiles)
    if includeNegativeEdges:
        print("Loading negative edge files")
        negativeEdgesFiles = loadExtraEdgeFiles(input_prefix, negativeedgefiles)

    print("Loading Provenance XML file")

    if kFoldGenerated:
        print("Preparing k fold generated data")
        if onlyOneFile:
            pass

        else:
            xmlFiles =  loadProvenanceXMLList(input_prefix, inputfiles)
            build_categoric_dictionary_for_list(xmlFiles)
            for i in range(kFoldNumber):
                #print ("Fold no. "+str(i+1))
                G = nx.Graph()
                G.graph['name'] = output_prefix+str(i)
                if includeExtraEdges:
                    provList2nx(xmlFiles, G, edgesFiles)
                else:
                    provList2nx(xmlFiles, G)
                outputJsonGraphFiles(G, str(i))
                idmap.clear()
                classmap.clear()
                featmap.clear()
                if leaveOneGraphOutOfTraining:
                    xmlFiles = xmlFiles[-2:-1]+xmlFiles[:-2]+xmlFiles[-1:]
                else:
                    xmlFiles = xmlFiles[-1:]+xmlFiles[:-1]
                
    else:

        G = nx.Graph()

        if onlyOneFile:
            e = loadProvenanceXML(input_prefix+inputfile)
            build_categoric_dictionary(e)
            G.graph['name'] = output_prefix
            print("Creating NX Graph from Provenance Graph")
            prov2nx(e, G)

        else:
            xmlFiles, nameXmlFiles = loadProvenanceXMLList(input_prefix, inputfiles)
            build_categoric_dictionary_for_list(xmlFiles)
            G.graph['name'] = output_prefix
            print("Creating NX Graph from Provenance Graph Collection")
            if includeExtraEdges and includeNegativeEdges:
                provList2nx(xmlFiles, G, edgesFiles, negativeEdgesFiles)
            elif includeExtraEdges:
                provList2nx(xmlFiles, G, edgesFiles)
            else:
                provList2nx(xmlFiles, G)


        # Write output json Graph
        print("Writing Json Graph File")
        json_data = nx.readwrite.json_graph.node_link_data(G)
        create_output_json_graph(json_data)

        # Write id map json
        print("Writing Json Id Map File")
        create_output_json_idmap(idmap)

        # Write class_map json
        print("Writing Json Class Map File")
        create_output_json_classmap(classmap)

        # Write class_map json
        print("Writing Json Features Map File")
        create_output_json_features_map(featmap)

main()

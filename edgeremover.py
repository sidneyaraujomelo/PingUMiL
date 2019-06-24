from config.util import *
from config.config import *
from config.preprocess import *
import math
from argparse import ArgumentParser
from random import choice

DEBUG = 0

def createNegativeExamples(root, origin, positiveExamples, types, search_window):
    listOfSetEdges = []
    listOfAllNegativeExamples = []
    dictOfEdgesCount = {}
    if isinstance(types,dict):
        all_possible_targetLabels = list(types.keys())
    elif isinstance(types,str):
        all_possible_targetLabels = [types.split("->")[0]]
        types = { types.split("->")[0] : types.split("->")[1]}
    else:
        print("This edge type is not supported")
        return
    vertexes = root.find("vertices").findall("vertex")
    for i in range(len(vertexes)):
        cur_v_label = vertexes[i].find("label").text
        cur_v_id = getVertexIntID(vertexes[i])
        if (cur_v_label in all_possible_targetLabels):
            #print("Searching negative example for label with ID {} and label {}".format(cur_v_id, cur_v_label))
            for j in range(2,search_window):
                if (i + j > len(vertexes)-1):
                    continue
                candidate_v_label = vertexes[i+j].find("label").text
                candidate_v_id = getVertexIntID(vertexes[i+j])
                if ([candidate_v_id, cur_v_id] in positiveExamples):
                    #print("Capturou exemplo positivo {}".format([candidate_v_id, cur_v_id]))
                    continue
                elif (candidate_v_label in types[cur_v_label]):
                    #print("Capturou exemplo negativo {} com labels {}->{}".format([candidate_v_id, cur_v_id], cur_v_label, candidate_v_label))
                    listOfAllNegativeExamples.append([candidate_v_id, cur_v_id])
                    listOfSetEdges.append(SetEdge(origin, candidate_v_id, cur_v_id, candidate_v_label, cur_v_label))
                    if (cur_v_label not in dictOfEdgesCount):
                        dictOfEdgesCount[cur_v_label] = {candidate_v_label: 1}
                    elif (candidate_v_label not in dictOfEdgesCount[cur_v_label]):
                        dictOfEdgesCount[cur_v_label][candidate_v_label] = 1
                    else:
                        dictOfEdgesCount[cur_v_label][candidate_v_label] = dictOfEdgesCount[cur_v_label][candidate_v_label] + 1

    print("Lista de todos os possíveis target labels de arestas inventadas: {}".format(all_possible_targetLabels))
    print("Numero de arestas inventadas: {} \nConjunto de arestas:\n{}".format(len(listOfAllNegativeExamples),listOfAllNegativeExamples))
    print("Lista de Sequencias de labels de arestas inventadas: {}".format(dictOfEdgesCount))

    return listOfAllNegativeExamples, dictOfEdgesCount, listOfSetEdges, dictOfEdgesCount

def removeEdgesByType(root, origin, target_edge_type):
    listOfSetEdges = []
    listOfTargetEdgesIds = []
    dictOfEdgeTypes = {}
    dictOfEdgesCount = {}
    elementsToRemove = []
    first_vertex = getVertexIntID(root.find("vertices").find("vertex"))
    for element in root:
        if (element.tag == "edges"):
            for edgeElement in element:
                sourceID, targetID = getEdgeSourceAndTargetIDs(edgeElement)
                sourceVertex = getVertexByID(root,"vertex_{}".format(sourceID))
                targetVertex = getVertexByID(root,"vertex_{}".format(targetID))
                targetLabel = getTextFromNode(targetVertex, "label")
                sourceLabel = getTextFromNode(sourceVertex, "label")

                edgeType = "{}->{}".format(targetLabel, sourceLabel)
                if (edgeType == target_edge_type):
                    #print("{} {} {}".format(sourceID, targetID, abs(sourceID-targetID)))
                    listOfTargetEdgesIds.append([sourceID, targetID])
                    elementsToRemove.append(edgeElement)
                    #element.remove(edgeElement)
                    
                    listOfSetEdges.append(SetEdge(origin, sourceID, targetID, sourceLabel, targetLabel))

                    if (targetLabel not in dictOfEdgeTypes):
                        dictOfEdgeTypes[targetLabel] = [sourceLabel]
                    elif (sourceLabel not in dictOfEdgeTypes[targetLabel]):
                        dictOfEdgeTypes[targetLabel].append(sourceLabel)

                    if (targetLabel not in dictOfEdgesCount):
                        dictOfEdgesCount[targetLabel] = {sourceLabel: 1}
                    elif (sourceLabel not in dictOfEdgesCount[targetLabel]):
                        dictOfEdgesCount[targetLabel][sourceLabel] = 1
                    else:
                        dictOfEdgesCount[targetLabel][sourceLabel] = dictOfEdgesCount[targetLabel][sourceLabel] + 1

    #De fato remover as arestas
    element = root.find("edges")
    for edgeElement in elementsToRemove:
        element.remove(edgeElement)

    print("Numero de arestas removidas: {} \nConjunto de arestas:\n{}".format(len(listOfTargetEdgesIds),listOfTargetEdgesIds))
    print("Lista de Sequencias de labels de arestas removidas: {}".format(dictOfEdgeTypes))
    print("Contagem de nós: {}".format(dictOfEdgesCount))
    return listOfTargetEdgesIds, dictOfEdgeTypes, listOfSetEdges, dictOfEdgesCount

def removeIndirectEdges(root, origin):
    listOfSetEdges = []
    listOfTargetEdgesIds = []
    dictOfEdgeTypes = {}
    dictOfEdgesCount = {}
    elementsToRemove = []
    first_vertex = getVertexIntID(root.find("vertices").find("vertex"))
    for element in root:
        if (element.tag == "edges"):
            for edgeElement in element:
                sourceID, targetID = getEdgeSourceAndTargetIDs(edgeElement)
                if (abs(sourceID-targetID) > 1 and sourceID != first_vertex and targetID != first_vertex):
                    #print("{} {} {}".format(sourceID, targetID, abs(sourceID-targetID)))
                    listOfTargetEdgesIds.append([sourceID, targetID])
                    elementsToRemove.append(edgeElement)
                    #element.remove(edgeElement)
                    sourceVertex = getVertexByID(root,"vertex_{}".format(sourceID))
                    targetVertex = getVertexByID(root,"vertex_{}".format(targetID))
                    targetLabel = getTextFromNode(targetVertex, "label")
                    sourceLabel = getTextFromNode(sourceVertex, "label")

                    listOfSetEdges.append(SetEdge(origin, sourceID, targetID, sourceLabel, targetLabel))

                    if (targetLabel not in dictOfEdgeTypes):
                        dictOfEdgeTypes[targetLabel] = [sourceLabel]
                    elif (sourceLabel not in dictOfEdgeTypes[targetLabel]):
                        dictOfEdgeTypes[targetLabel].append(sourceLabel)

                    if (targetLabel not in dictOfEdgesCount):
                        dictOfEdgesCount[targetLabel] = {sourceLabel: 1}
                    elif (sourceLabel not in dictOfEdgesCount[targetLabel]):
                        dictOfEdgesCount[targetLabel][sourceLabel] = 1
                    else:
                        dictOfEdgesCount[targetLabel][sourceLabel] = dictOfEdgesCount[targetLabel][sourceLabel] + 1

    #De fato remover as arestas
    element = root.find("edges")
    for edgeElement in elementsToRemove:
        element.remove(edgeElement)

    print("Numero de arestas removidas: {} \nConjunto de arestas:\n{}".format(len(listOfTargetEdgesIds),listOfTargetEdgesIds))
    print("Lista de Sequencias de labels de arestas removidas: {}".format(dictOfEdgeTypes))
    print("Contagem de nós: {}".format(dictOfEdgesCount))
    return listOfTargetEdgesIds, dictOfEdgeTypes, listOfSetEdges, dictOfEdgesCount

def main():
    parser = ArgumentParser("Remove edges on Provenance data.")
    parser.add_argument("path", help="Path to files")
    parser.add_argument("mode_preprocess", help="None | Indirect | EdgeType", default="None")
    parser.add_argument("--negative_samples", help="False | True", default="True")
    parser.add_argument("--edge_type", help="if mode_process == EdgeType, this argument must be an EdgeType such as Flying->Landing")
    parser.add_argument("--output_path", help="Path to the output file")
    parser.add_argument("--input_filename", help="Input graph file if path contains a directory.")
    parser.add_argument("--output_filename_prefix", help="A prefix to the name of the output graph file.")
    args = parser.parse_args()
    print(args)
    mode = args.mode_preprocess
    negative_samples = args.negative_samples == "True"
    edge_type = args.edge_type
    if os.path.isfile(args.path):
        path = os.path.dirname(args.path)
        input_fns = [os.path.basename(path)]
    else:
        path = args.path
        if args.input_filename:
            input_fns = [args.input_filename]
        else:
            input_fns = [x for x in os.listdir(path) if x.endswith(".xml")]

    output_path = args.output_path if args.output_path != None else path
    output_fns = ["{}_{}".format(args.output_filename_prefix, x) for x in input_fns]
    createPath(output_path)

    for i,input_fn in enumerate(input_fns):
        output_fn = output_fns[i]
        removedEdges_fn = ".".join(output_fn.split(".")[:-1])+".txt"

        file_removedEdges = open(output_path+"/"+removedEdges_fn, "w+")
        if (file_removedEdges == None):
            print("Incapable of creating and loading Removed Edges File")
            return

        if (negative_samples):
            file_negativeEdges = open(output_path+"/negative-"+removedEdges_fn, "w+")
            if (file_negativeEdges == None):
                print("Incapable of creating and loading Negative Edges File")
                return

        tree = ET.parse(path+"/"+input_fn)
        root = tree.getroot()
        print("Numero de arestas do grafo: {}".format(getNumberOfEdges(root)))

        if (mode == "Indirect"):
            removedEdgesIds, removedEdgesTypes = removeIndirectEdges(root, input_fn)
            for removedEdge in removedEdgesIds:
                file_removedEdges.write("{} {}\n".format(removedEdge[0], removedEdge[1]))
            tree.write(output_path+"/"+output_fn)

            if (negative_samples):
                negativeEdgesExamplesId = createNegativeExamples(root, input_fn, removedEdgesIds, removedEdgesTypes, 5)

                negativeEdgesSet = []
                if (len(negativeEdgesExamplesId) < len(removedEdgesIds)):
                    negativeEdgesSet = negativeEdgesExamplesId
                else:
                    while (len(negativeEdgesSet) < len(removedEdgesIds)):
                        c = choice(negativeEdgesExamplesId)
                        if (c not in negativeEdgesSet):
                            negativeEdgesSet.append(c)

                print("Conjunto de arestas negativas geradas aleatoriamente: {}".format(negativeEdgesSet))

                for negativeEdge in negativeEdgesSet:
                    file_negativeEdges.write("{} {}\n".format(negativeEdge[0], negativeEdge[1]))
        elif (mode == "EdgeType"):
            removedEdgesIds,_ = removeEdgesByType(root, input_fn, edge_type)
            for removedEdge in removedEdgesIds:
                file_removedEdges.write("{} {}\n".format(removedEdge[0], removedEdge[1]))
            tree.write(os.path.join(output_path, output_fn))

            if (negative_samples):
                negativeEdgesExamplesId,_,_,_ = createNegativeExamples(root, input_fn, removedEdgesIds, edge_type, 10)

                negativeEdgesSet = []
                if (len(negativeEdgesExamplesId) < len(removedEdgesIds)):
                    negativeEdgesSet = negativeEdgesExamplesId
                else:
                    while (len(negativeEdgesSet) < len(removedEdgesIds)):
                        c = choice(negativeEdgesExamplesId)
                        if (c not in negativeEdgesSet):
                            negativeEdgesSet.append(c)

                print("Conjunto de arestas negativas geradas aleatoriamente: {}".format(negativeEdgesSet))

                for negativeEdge in negativeEdgesSet:
                    file_negativeEdges.write("{} {}\n".format(negativeEdge[0], negativeEdge[1]))

        file_removedEdges.close()

if __name__=='__main__':
    main()
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
    all_possible_targetLabels = list(types.keys())
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

def removeIndirectEdges(root, origin):
    listOfSetEdges = []
    listOfIndirectEdgesIds = []
    dictOfLabelSequences = {}
    dictOfEdgesCount = {}
    elementsToRemove = []
    first_vertex = getVertexIntID(root.find("vertices").find("vertex"))
    for element in root:
        if (element.tag == "edges"):
            for edgeElement in element:
                sourceID = int(edgeElement.find("sourceID").text.split("_")[-1])
                targetID = int(edgeElement.find("targetID").text.split("_")[-1])
                if (abs(sourceID-targetID) > 1 and sourceID != first_vertex and targetID != first_vertex):
                    #print("{} {} {}".format(sourceID, targetID, abs(sourceID-targetID)))
                    listOfIndirectEdgesIds.append([sourceID, targetID])
                    elementsToRemove.append(edgeElement)
                    #element.remove(edgeElement)
                    sourceVertex = getVertexByID(root,"vertex_{}".format(sourceID))
                    targetVertex = getVertexByID(root,"vertex_{}".format(targetID))
                    targetLabel = getTextFromNode(targetVertex, "label")
                    sourceLabel = getTextFromNode(sourceVertex, "label")

                    listOfSetEdges.append(SetEdge(origin, sourceID, targetID, sourceLabel, targetLabel))

                    if (targetLabel not in dictOfLabelSequences):
                        dictOfLabelSequences[targetLabel] = [sourceLabel]
                    elif (sourceLabel not in dictOfLabelSequences[targetLabel]):
                        dictOfLabelSequences[targetLabel].append(sourceLabel)

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

    print("Numero de arestas removidas: {} \nConjunto de arestas:\n{}".format(len(listOfIndirectEdgesIds),listOfIndirectEdgesIds))
    print("Lista de Sequencias de labels de arestas removidas: {}".format(dictOfLabelSequences))
    print("Contagem de nós: {}".format(dictOfEdgesCount))
    return listOfIndirectEdgesIds, dictOfLabelSequences, listOfSetEdges, dictOfEdgesCount

def main():
    parser = ArgumentParser("Remove edges on Provenance data.")
    parser.add_argument("path", help="Path to files")
    parser.add_argument("input_filename", help="Input graph file.")
    parser.add_argument("output_filename", help="Output graph file.")
    parser.add_argument("removed_edges_filename", help="List of removed edges filename")
    parser.add_argument("mode_preprocess", help="None | Indirect")
    parser.add_argument("negative_samples", help="False | True")
    args = parser.parse_args()

    path = args.path
    input_fn = args.input_filename
    output_fn = args.output_filename
    removedEdges_fn = args.removed_edges_filename
    mode = args.mode_preprocess
    negative_samples = args.negative_samples == "True"

    file_removedEdges = open(path+"/"+removedEdges_fn, "w+")
    if (file_removedEdges == None):
        print("Incapable of creating and loading Removed Edges File")
        return

    if (negative_samples):
        file_negativeEdges = open(path+"/negative-"+removedEdges_fn, "w+")
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
        tree.write(path+"/"+output_fn)

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


    file_removedEdges.close()

if __name__=='__main__':
    main()
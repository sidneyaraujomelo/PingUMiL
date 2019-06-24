from config.util import *
from config.preprocess import *
from config.config import *
import edgeremover
from argparse import ArgumentParser

def createEdgeSetForClassifiers(filename, edgesetlists, partitions):
    toJson = []
    for class_set in edgesetlists:
        for c_part in partitions:
            for edgeset in class_set[c_part]:
                toJson.append({
                    "source": edgeset["edge"].id_source,
                    "target": edgeset["edge"].id_target,
                    "class": edgeset["class"]})
    createOutputJson(filename, toJson)

#Create folds for SetEdges
def createFolds(setEdgesList, typeEdgesCount, k):
    folds = {}
    for i in range(k):
        if i not in folds:
            folds[i] = []
    for label_source in typeEdgesCount.keys():
        sources = typeEdgesCount[label_source]
        for label_target in sources:
            c = sources[label_target]
            idx = 0
            while idx < c:
                for edge in setEdgesList:
                    if (edge.isType(label_source, label_target)):
                        folds[idx%k].append(edge)
                        idx = idx+1
                    if (idx >= c):
                        break
    return folds

def main():
    parser = ArgumentParser("Balance datasets of Provenance data.")
    parser.add_argument("path", help="Path to files")
    parser.add_argument("k", help="Number of folds")
    parser.add_argument("--output_folder", help="Output folder for edge sets.")
    parser.add_argument("--edge_type", help="Set if remove edges by edge type mode.")
    args = parser.parse_args()
    k = int(args.k)
    edge_type = args.edge_type

    print("XML files: ")
    graphRoots, input_filenames =  loadProvenanceXMLList(args.path, inputfiles)
    buildCategoricDictionaryForList(graphRoots)
    
    allRemovedEdges = []
    allRemovedEdgesTypes = {}
    allRemovedEdgesIds = []
    allRemovedEdgesCount = {}
    allNegativeExampleEdges = []
    allNegativeEdgesCount = {}

    for idx, root in enumerate(graphRoots):
        print("Number of edges: {}".format(getNumberOfEdges(root)))
        print("Removal of target edges:")
        if edge_type:
            removedEdgesIds, removedEdgesTypes, setEdges, removedEdgesCount = edgeremover.removeEdgesByType(root, idx, edge_type)
        else:
            removedEdgesIds, removedEdgesTypes, setEdges, removedEdgesCount = edgeremover.removeIndirectEdges(root, idx)
        tree = ET.ElementTree()
        tree._setroot(root)
        createPath(args.path+args.output_folder+"no_target_edge_graph/")
        tree.write(args.path+args.output_folder+"no_target_edge_graph/r-"+input_filenames[idx])
        allRemovedEdges = allRemovedEdges+setEdges
        allRemovedEdgesIds = allRemovedEdgesIds+removedEdgesIds
        allRemovedEdgesTypes = mergeEdgeLabelDictionaries(allRemovedEdgesTypes, removedEdgesTypes)
        allRemovedEdgesCount = sumEdgeCountDictionaries(allRemovedEdgesCount, removedEdgesCount)
        print("\n")

    for idx,root in enumerate(graphRoots):
        print("Creation of negative examples")
        if edge_type:
            negativeEdgesExamplesId, negativeEdgesTypes, setNegativeEdges, negativeEdgesCount = edgeremover.createNegativeExamples(root, idx, allRemovedEdgesIds, edge_type, 10)
        else:
            negativeEdgesExamplesId, negativeEdgesTypes, setNegativeEdges, negativeEdgesCount = edgeremover.createNegativeExamples(root, idx, allRemovedEdgesIds, allRemovedEdgesTypes, 10)
        allNegativeEdgesCount = sumEdgeCountDictionaries(allNegativeEdgesCount, negativeEdgesCount)
        allNegativeExampleEdges = allNegativeExampleEdges+setNegativeEdges
        print("\n")

    print("All types of indirect edges removed: {}".format(allRemovedEdgesTypes))
    print("Number of positive indirect edges removed: {}\n".format(len(allRemovedEdges)))
    print("Number of removed edges per type: {}\n".format(allRemovedEdgesCount))
    print("Number of negative indirect edges created: {}\n".format(len(allNegativeExampleEdges)))
    print("Number of created negative edges per type: {}\n".format(allNegativeEdgesCount))
    selectedNegativeExamples = []
    selectedPositiveExamples = []
    selectedExamplesCount = {}
    """skipEdges is used to prevent the algorithm to search edges already depleted from negative 
    edges pool"""
    skipEdges = {}
    for edge in allRemovedEdges:
        if (edge.label_source in skipEdges):
            if (edge.label_target in skipEdges[edge.label_source]):
                continue
        flag = 0
        for idx, n_edge in enumerate(allNegativeExampleEdges):
            if edge.compareType(n_edge):
                selectedNegativeExamples.append(n_edge)
                selectedPositiveExamples.append(edge)
                del allNegativeExampleEdges[idx]
                incrementEdgeTypeInDictionary(selectedExamplesCount, edge.label_source, edge.label_target)
                flag = 1
                break
        if flag == 0:
            if edge.label_source not in skipEdges:
                skipEdges[edge.label_source] = []
            elif edge.label_target not in skipEdges[edge.label_source]:
                skipEdges[edge.label_source].append(edge.label_target)

    print("Number of target edges > positive: {} and negative: {}  ".format(len(selectedPositiveExamples),len(selectedNegativeExamples)))
    """for i in range(len(selectedPositiveExamples)):
        print("{} {} {}".format(i, selectedPositiveExamples[i], selectedNegativeExamples[i]))"""
    print(selectedExamplesCount)

    print()
    print("No. of folds to create: {}".format(k))

    folds = createFolds(selectedPositiveExamples, selectedExamplesCount, k)
    folds_n = createFolds(selectedNegativeExamples, selectedExamplesCount, k)
    output_json_p = [[{"source": x.id_source, "target": x.id_target, "class":1} for x in folds[i]] for i in range(k)]
    output_json_n = [[{"source": x.id_source, "target": x.id_target, "class":0} for x in folds_n[i]] for i in range(k)]
    output_data_p = [[{"edge": x, "class":1} for x in folds[i]] for i in range(k)]
    output_data_n = [[{"edge": x, "class":0} for x in folds_n[i]] for i in range(k)]

    createPath(args.path+(args.output_folder if args.output_folder!=None else "")+"parts/")
    createPath(args.path+(args.output_folder if args.output_folder!=None else "")+"clf/")

    for i in range(k):
        print("Fold {} has {} positive examples and {} negative examples".format(i,len(folds[i]),len(folds_n[i])))
        output_file = args.path+(args.output_folder if args.output_folder!=None else "")+"parts/part-{}.json".format(i)
        createOutputJson(output_file, output_json_p[i]+output_json_n[i])

    print("Creating Fold Files")
    for i in range(k):

        print("Creating fold {} for Classifiers".format(i))
        output_fileprefix = args.path+(args.output_folder if args.output_folder!=None else "")+"clf/clffold-{}-train".format(i)
        createEdgeSetForClassifiers(output_fileprefix, [output_data_p,output_data_n], range(k-1))
        output_fileprefix = args.path+(args.output_folder if args.output_folder!=None else "")+"clf/clffold-{}-test".format(i)
        createEdgeSetForClassifiers(output_fileprefix, [output_data_p,output_data_n], [-1])

        print("Rotate Partitions to the right")
        output_data_p = output_data_p[-1:]+output_data_p[0:-1]
        output_data_n = output_data_n[-1:]+output_data_n[0:-1]

if __name__=='__main__':
    main()
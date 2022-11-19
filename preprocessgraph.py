from config.util import *
from config.config import *
import math
from argparse import ArgumentParser

def main(argv=None):
    parser = ArgumentParser("Alter the node ids of several graphs so that the ids don't overlap.")
    parser.add_argument("path", help="Path to files")
    parser.add_argument("--output_folder", help="Output folder.")
    args = parser.parse_args()

    print("Arquivos XML abertos: ")
    graphRoots, input_filenames =  loadProvenanceXMLList(args.path, inputfiles)
    print(input_filenames)
    c_node_offset = 0

    for i in range(len(graphRoots)):
        print("No of nodes in graph {}: {}".format(i, getNumberOfNodes(graphRoots[i])))

    for i in range(len(graphRoots)):
        vertexes = graphRoots[i].find("vertices")
        for node in vertexes:
            node_id = getVertexIntID(node)
            setVertexIntID(node, node_id+c_node_offset)
        edges = graphRoots[i].find("edges")
        for edge in edges:
            source_id = int(getTextFromNode(edge,"sourceID").split('_')[-1])
            target_id = int(getTextFromNode(edge,"targetID").split('_')[-1])
            setTextToNode(edge,"sourceID","vertex_"+str(source_id+c_node_offset))
            setTextToNode(edge,"targetID","vertex_"+str(target_id+c_node_offset))
        c_node_offset = c_node_offset + getNumberOfNodes(graphRoots[i])
        createPath(args.path+("../"+args.output_folder if args.output_folder!=None else ""))
        output_file = args.path+("../"+args.output_folder if args.output_folder!=None else "")+"j-{}".format(input_filenames[i])
        tree = ET.ElementTree()
        tree._setroot(graphRoots[i])
        tree.write(output_file)
    
if __name__=='__main__':
    main()
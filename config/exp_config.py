# INPUT PARAMETERS
# Only One or many provenance files
onlyOneFile = False
# Is it k-fold Generated?
kFoldGenerated = False
kFoldNumber = 10
# Extra edges are included?
includeExtraEdges = False
# Negative edges are included?
includeNegativeEdges = False

# Exclude last provenance file loaded from folds? (For tests in which accuracy is 100%)
leaveOneGraphOutOfTraining = False

# Provenance graph file name
inputfile = "output 08.xml"

# Provenance graph files name separated by ,
input_prefix = "/home/sidneymelo/Documentos/PingUMiL/Dataset/SmokeSquadron/no_target_edge_graph/"
#inputfiles = ["P1-S1.xml","P1-S2.xml", "P1-S3.xml", "P2-S1.xml", "P2-S2.xml", "P2-S3.xml", "P3-S1.xml", "P3-S2.xml", "P3-S3.xml", "P4-S1.xml"]
inputfiles = []

# Extra edge files
extraedgesfiles = ['e01.txt','e02.txt','e03.txt']

# Negative edge files
negativeedgefiles = []

# Prefix for all output files which will be generated
output_prefix = "prov"
output_folder = "ss10_test/"

# Ratio of training, test and validation sets
train_ratio = 0.7
test_ratio = 0.2
valid_ratio = 0.1

# Attributes for nodes regarding train\test\validation graphs
train_dict = {"test": False, "val": False}
test_dict = {"test": True, "val": False}
valid_dict = {"test": False, "val": True}

# Attributes for edges regarding train\test\validation graphs
train_edge_dict = {"test_removed": False, "train_removed": False}
test_edge_dict = {"test_removed": True, "train_removed": True}
valid_edge_dict = {"test_removed": False, "train_removed": True}

# Is the graph directed?
directed = False

# Attributes for GAT
train_nodes_per_class = 10
gat_test_ratio = 0.4
gat_valid_ratio = 0.2

# ATTRIBUTES SHOULD BE WRITTEN IN NODE JSON
attrib_written_in_node = False
import json

class ParseConfiguration():
    def __init__(self, parser_config_json):
        self.onlyOneFile = parser_config_json["onlyOneFile"]
        self.mergeGraphs = parser_config_json["mergeGraphs"]
        self.kFoldGenerated = parser_config_json["kFoldGenerated"]
        self.kFoldNumber = parser_config_json["kFoldNumber"]
        self.includeExtraEdges = parser_config_json["includeExtraEdges"]
        self.includeNegativeEdges = parser_config_json["includeNegativeEdges"]
        self.leaveOneGraphOutOfTraining = parser_config_json["leaveOneGraphOutOfTraining"]
        self.inputfile = parser_config_json["inputfile"]
        self.input_prefix = parser_config_json["input_prefix"]
        self.inputfiles = parser_config_json["inputfiles"]
        self.ignorefiles = parser_config_json["ignorefiles"]
        self.extraedgesfiles = parser_config_json["extraedgesfiles"]
        self.negativeedgefiles = parser_config_json["negativeedgefiles"]
        self.use_graph_name = parser_config_json["use_graph_name"]
        self.output_prefix = parser_config_json["output_prefix"]
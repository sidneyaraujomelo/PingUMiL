from pingumil.models.graphsage import GraphSAGE
from pingumil.models.gat import GAT
from pingumil.models.hgt import HGT
from pingumil.models.typeprojection import TypeProjection
from pingumil.models.linkpredictor import LinkPredictor
from pingumil.models.multiheadattention import MultiHeadAttention
from pingumil.models.multilabel_classifier import MultilabelClassifier

def load_model(config):
    if config["model"] == "graphsage":
        return GraphSAGE(
            config["in_channels"],
            config["hidden_channels"],
            config["out_channels"],
            config["dropout"])
    if config["model"] == "typeprojection":
        return TypeProjection(
            config["dim_types"],
            config["dim_output"])
    if config["model"] == "link_pred":
        return LinkPredictor(
            config["in_channels"],
            config["composition_function"]
            )
    if config["model"] == "mhattention":
        del config["model"]
        return MultiHeadAttention(**config)
    if config["model"] == "gat":
        del config["model"]
        return GAT(**config)
    if config["model"] == "hgt":
        del config["model"]
        return HGT(**config)
    if config["model"] == "multilabel_classification":
        del config["model"]
        return MultilabelClassifier(**config)

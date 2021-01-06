from pingumil.models.graphsage import GraphSAGE
from pingumil.models.typeprojection import TypeProjection
from pingumil.models.linkpredictor import LinkPredictor
from pingumil.models.multiheadattention import MultiHeadAttention

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
        return MultiHeadAttention(
            config["n_head"],
            config["d_model"],
            config["d_k"],
            config["d_v"]
        )
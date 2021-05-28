import torch

def get_edge_type_list(edge_index, edge_type_function):
    get_edge_type = lambda x,y: f"{edge_type_function(x)}->{edge_type_function(y)}"
    edge_type_list = [get_edge_type(x,y) for x,y in zip(
        edge_index[0].tolist(), edge_index[1].tolist()
    )]
    return edge_type_list
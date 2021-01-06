import glob
import json
import os
import torch
import numpy as np
from models.graphsage import GraphSAGE
from argparse import ArgumentParser
from networkx.readwrite import json_graph
from models import load_model
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import NeighborSampler

"""
PingUMiL is a framework for Deep Learning on Game Provenance Graphs.
"""

def run_experiment(model, data, exp_config):
    # Create NeighborSamplers for experiment 
    train_loader = NeighborSampler(data.edge_index, sizes=[2,2,2,2],
                                   node_idx=data.train, batch_size=512,
                                   shuffle=True)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                      sizes=[-1], batch_size=4096,
                                      shuffle=False)

    #Pass data to GPU
    device = torch.device(exp_config["device"])
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp_config["lr"],
                                 weight_decay=exp_config["weight_decay"])

    avg_train_losses = []
    avg_valid_losses = []

    #early_stopping = EarlyStopping(patience=3, verbose=True)

    test_accs = []
    for run in range(1, 2):
        #print('')
        #print(f'Run {run:02d}:')
        #print('')

        best_val_acc = final_test_acc = 0
        for epoch in range(1, exp_config["num_epochs"]):
            loss, acc = model.run_train(data, train_loader, optimizer, device)
            train_loss = loss
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

            """train_acc, val_acc, test_acc, val_loss = test(model_name)
            valid_loss = val_loss.item()
            
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            #print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            #        f'Test: {test_acc:.4f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
            
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping!")
                break

        test_accs.append(final_test_acc)

    test_acc = torch.tensor(test_accs)
    print('============================')
    print(f'{model_name}: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
    print('============================')"""

def main():
    """ Framework's main method.
    """
    argparser = ArgumentParser()
    argparser.add_argument("dataset_folder", type=str,
                           help="Path to the dataset")
    argparser.add_argument("dataset_prefix", type=str,
                           help="Prefix of dataset files")
    argparser.add_argument("model_config", type=str,
                           help="Path to model configuration json")
    argparser.add_argument("experiment_config", type=str,
                           help="Path to experiment configuration json")
    args = argparser.parse_args()

    #First, we load all the data in the dataset folder.
    graph_data = json.load(open(os.path.join(args.dataset_folder,
                                             f"{args.dataset_prefix}-G.json")))
    graph = json_graph.node_link_graph(graph_data)
    print(graph.nodes()[0])

    node_feats = np.load(os.path.join(args.dataset_folder,
                                      f"{args.dataset_prefix}-feats.npy"))
    print(node_feats[0])

    # Then, we load the model config and create the specified model
    model_config = json.load(open(args.model_config))
    model_config["in_channels"] = node_feats.shape[-1]
    model_config["out_channels"] = 2
    model = load_model(model_config)
    print(model)

    # Now, we transform graph data from NX to Pytorch Geometric's
    data = from_networkx(graph)
    data.train = torch.bitwise_not(torch.bitwise_xor(data.test, data.val))
    data.x = torch.from_numpy(node_feats).float()
    data.y = torch.randint(0,2,(data.x.shape[0],))
    print(data.y)

    # Now, we load the experiment config
    exp_config = json.load(open(args.experiment_config))

    run_experiment(model, data, exp_config)


if __name__ == "__main__":
    main()
import json
import os
from collections import Counter
import torch
from glob import glob
import numpy as np
from networkx.readwrite import json_graph
import torch
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, hamming_loss,
                             classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from pingumil.util.util import generate_class_weights
from pingumil.util.pytorchtools import EarlyStopping
import time
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-15

class GPPBaseExperiment():
    def __init__(
            self,
            dataset_folder="/raid/home/smelo/PingUMiL-pytorch/dataset/GPP/preprocessed_graphs_test",
            dataset_prefix="gpp",
            model_config="configs/gpp_sagemodel.json",
            experiment_tag="base",
            timestamp=None,
            standardization=True,
            epochs=1000,
            mlc_epochs=1000,
            patience=100,
            wandb=None,
            override_data=False):
        self.dataset_folder = dataset_folder
        self.dataset_prefix = dataset_prefix
        self.model_config = model_config
        self.experiment_tag = experiment_tag
        self.wandb = wandb
        self.override_data = override_data
        if timestamp:
            self.timestamp=timestamp
        else:
            self.timestamp = time.time()
        if not self.wandb:
            self.run_name = self.timestamp
        else:
            self.run_name = self.wandb.run.name
        self.standardization = standardization
        self.epochs = epochs
        self.mlc_epochs = mlc_epochs
        self.patience = patience
        self.log_folder = f"experiment_log/{self.dataset_prefix}_{self.experiment_tag}/{self.run_name}"
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        self.output_file = f"{self.log_folder}/{self.run_name}.txt"
        self.ground_truth = pd.read_csv("/raid/home/smelo/PingUMiL-pytorch/dataset/GPP/final_results.csv", index_col=False)
        columns = self.ground_truth.columns
        columns = ["Player"]+columns.to_list()[1:]
        self.ground_truth.columns = columns
        self.log(f"Experiment {self.timestamp}")

    def log(self, message, mode="a"):
        with open(self.output_file, mode) as fp:
            fp.write(message+"\n")
    
    def read_data(self):
        graph_folders = glob(os.path.join(self.dataset_folder,"*"))
        dataset = {x: {} for x in graph_folders if not os.path.isfile(x)}
        for graph_folder in graph_folders:
            if os.path.isfile(graph_folder):
                continue
            print(graph_folder)
            graph_json_path = glob(os.path.join(graph_folder, f"{self.dataset_prefix}*-G.json"))[0]
            graph_json_file = os.path.basename(graph_json_path)
            graph_id = graph_json_file[len(self.dataset_prefix):-7]
            if (not self.override_data and 
                os.path.exists(os.path.join(self.dataset_folder, graph_folder,
                                            f"{self.dataset_prefix}{graph_id}-G.data"))):
                data = torch.load(os.path.join(self.dataset_folder, graph_folder,
                                            f"{self.dataset_prefix}{graph_id}-G.data"))
            else:
                #First, we load all the data in the dataset folder.
                graph_json = json.load(open(os.path.join(self.dataset_folder, graph_folder,
                                                        f"{self.dataset_prefix}{graph_id}-G.json")))
                graph = json_graph.node_link_graph(graph_json)
                #Create data object for pytorch geometric (takes a long time)
                data = from_networkx(graph)
                torch.save(data, os.path.join(self.dataset_folder, graph_folder,
                                            f"{self.dataset_prefix}{graph_id}-G.data"))
                #print(data)
            dataset[graph_folder]["graph"] = data

            #Load attribute set list that describes each set
            atbsets_list = json.load(open(os.path.join(self.dataset_folder, graph_folder,
                                                    f"{self.dataset_prefix}{graph_id}-atbset_list.json")))
            #print(atbsets_list)
            dataset[graph_folder]["atbset_list"] = atbsets_list

            #Now, we load the attribute set map files
            node_maps = []
            node_maps_files = sorted(
                [x for x in glob(os.path.join(self.dataset_folder, graph_folder, f"*{graph_id}*-map.json"))])
            node_maps = [json.load(open(x)) for x in node_maps_files]
            dataset[graph_folder]["node_maps"] = node_maps

            #Now, we load the attribute set feats files
            node_feats = []
            node_feats_files = sorted(
                [x for x in glob(os.path.join(self.dataset_folder, graph_folder, f"*{graph_id}*-feats.npy"))])
            node_feats = [torch.from_numpy(np.load(x)).float() for x in node_feats_files]
            dataset[graph_folder]["node_feats"] = node_feats

            #Now, we load the classes from the class maps
            class_maps_file = glob(os.path.join(self.dataset_folder, graph_folder, f"*{graph_id}*-class_map.json"))[0]
            class_map = json.load(open(class_maps_file))
            dataset[graph_folder]["class_map"] = class_map

            #Check if everything is sound
            assert len(node_feats) == len(node_maps)

            for k in range(len(node_feats)):
                assert len(node_maps[k])==node_feats[k].size()[0]
        return dataset

    def standardize_all(self, dataset):
        all_data = torch.cat([v["node_feats"] for k,v in dataset.items()])
        scaler = StandardScaler()
        stdzed_data = torch.from_numpy(scaler.fit_transform(all_data)).type(torch.FloatTensor)
        k = 0
        for graph_folder, graph_data in dataset.items():
            stdzed_graph_data = stdzed_data[k:k+graph_data["node_feats"].shape[0],:]
            dataset[graph_folder]["node_feats"] = stdzed_graph_data
            k = k + graph_data["node_feats"].shape[0]
        stdzed_confirm_data = torch.cat([v["node_feats"] for k,v in dataset.items()])
        assert stdzed_confirm_data.shape == all_data.shape
        return dataset

    def get_early_stopping(self, verbose, prefix, delta=0, patience=None):
        if patience is None:
            patience = self.patience
        return EarlyStopping(patience=patience,
                             verbose=verbose,
                             delta=delta,
                             path=f"{self.log_folder}/{prefix}_{self.timestamp}.pt")
    
    def multilabel_classification_step(self, device, model, x, y, graph_ids, wandb, suffix=""):
        self.log(f"Multilabel Classification Model: {model}_{suffix}")
        average_metrics = { k : [] for k in ["train_loss", "p", "r", "f1", "test_loss", "hamming"] }
        
        x = x.to(device)
        y = y.to(device)
        
        best_predicts = {}
        output_dict = {}
        output_dict["graph"] = []
        for k in range(y.size(1)):
            output_dict[f"class_{k}_pred"] = []
        for k in range(y.size(1)):
            output_dict[f"class_{k}_true"] = []
        
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(x):
            early_stopping = self.get_early_stopping(patience=self.patience,
                                                 verbose=True,
                                                 prefix="predictor")
            model.reset_parameters()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=wandb.config.lr_clf)
            test_graph_id = graph_ids[test_index[0]]
            best_metrics = {
                "train_loss": np.inf,
                "p": 0,
                "r": 0,
                "hamming": 0
            }
            for _ in range(1, self.mlc_epochs):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                model.train()
                optimizer.zero_grad()
                
                y_hat = model(x_train, sigmoid=False)
                
                y_train_weight = torch.Tensor(generate_class_weights(y_train)).to(device)
                
                train_loss = F.binary_cross_entropy_with_logits(y_hat, y_train,
                                                                pos_weight=y_train_weight)
                y_pred_train = torch.round(torch.sigmoid(y_hat))
                
                train_loss.backward()
                optimizer.step()
                
                model.eval()
                y_hat_test = model(x_test, sigmoid=False)
                test_loss = F.binary_cross_entropy_with_logits(y_hat_test, y_test)
                y_pred_test = torch.round(torch.sigmoid(y_hat_test))
                
                #print(f"{epoch}: Train loss: {train_loss.item()}")
                #print(f"{epoch}: Test loss: {test_loss.item()}")
                #Detaching all tensors
                np_y_test = y_test.cpu().detach().numpy()
                np_y_pred_test = y_pred_test.cpu().detach().numpy()
                np_y_train = y_train.cpu().detach().numpy()
                np_y_pred_train = y_pred_train.cpu().detach().numpy()
                
                p = precision_score(np_y_pred_test,
                                    np_y_test,
                                    average="weighted")
                r = recall_score(np_y_pred_test,
                                 np_y_test,
                                 average="weighted")
                f1 = f1_score(np_y_pred_test,
                              np_y_test,
                              average="weighted")
                h = hamming_loss(np_y_pred_test,
                                 np_y_test)
                
                train_acc = accuracy_score(np_y_pred_train,
                                           np_y_train)
                wandb.log({
                    f"train mlc loss_{suffix}" : train_loss.item(),
                    f"test mlc loss_{suffix}": test_loss.item(),
                    f"train acc_{suffix}": train_acc,
                    f"weighted_p_{suffix}" : p,
                    f"weighted_r_{suffix}" : r,
                    f"weighted_f1_{suffix}" : f1,
                    f"hamming_loss_{suffix}" : h
                })
                                
                if (train_loss.cpu().item() < best_metrics["train_loss"]):
                    best_metrics["train_loss"] = train_loss.cpu().item()
                    best_metrics["r"] = r
                    best_metrics["p"] = p
                    best_metrics["f1"] = f1
                    best_metrics["hamming"] = h
                    best_metrics["test_loss"] = test_loss.cpu().item()
                    best_predicts[test_graph_id] = (torch.flatten(y_pred_test.cpu().detach()).tolist(),
                                                    torch.flatten(y_test.cpu().detach()).tolist())

                early_stopping(train_loss, model)
                if early_stopping.early_stop:
                    print("Early Stopping!")
                    break
                
            for k,v in best_metrics.items():
                average_metrics[k].append(v)
        for k,v in average_metrics.items():
            self.log(f"{k}_avg: {torch.mean(torch.Tensor(v)).item()}")
            self.log(f"{k}_var {torch.var(torch.Tensor(v)).item()}")
            wandb.log({
                f"{k}_avg_{suffix}" : torch.mean(torch.Tensor(v)).item(),
                f"{k}_var_{suffix}" : torch.var(torch.Tensor(v)).item(),
            })
        for k,v in best_predicts.items():
            self.log(f"{k}_{suffix}->{v[0]}/{v[1]}")
            output_dict["graph"].append(os.path.basename(k))
            for i, v_pred in enumerate(v[0]):
                output_dict[f"class_{i}_pred"].append(v_pred)
            for i, v_pred in enumerate(v[1]):
                output_dict[f"class_{i}_true"].append(v_pred)
        output_df = pd.DataFrame.from_dict(output_dict)
        self.log(os.path.join(self.log_folder, f"results_{suffix}.csv"))
        results_path = os.path.join(self.log_folder, f"results_{suffix}.csv")
        output_df.to_csv(results_path, index=False)
        self.finish()
        
    def get_repeated_mode(self, lst):
        if (len(lst)) == 0:
            return ""
        data = Counter(lst)
        most_common = data.most_common()
        result = [most_common[0]]
        del most_common[0]
        i = 0
        while len(most_common) > 0 and i < len(most_common) and result[-1][1] == most_common[i][1]:
            result.append(most_common[i])
            i = i+1
        string_result = [x[0] for x in result]
        if len(string_result) > 1:
            return max(string_result, key=len)
        else:
            return string_result[0]
    
    def get_quality_results(self):
        
        self.labels = ["Achiever","Killer","Socializer","Explorer","Casual","Hardcore"]
        columns = ["graph"] + [f"{x}_pred" for x in self.labels] + [f"{x}_true" for x in self.labels]
        
        bartle_model_results = pd.read_csv(os.path.join(self.log_folder, f"results_bartle.csv"),
                                           index_col=False)
        bartle_model_results.columns = columns[:5]+columns[7:11]

        dedica_model_results = pd.read_csv(os.path.join(self.log_folder, f"results_dedication.csv"),
                                           index_col=False)
        dedica_model_results.columns = columns[:1]+columns[5:7]+columns[11:13]

        model_results = bartle_model_results.merge(dedica_model_results, on="graph")
        model_results = model_results[columns]
                
        # Create a dictionary that maps players to their respective graphs
        player2graphs = {}
        for _, row in self.ground_truth.iterrows():
            player = row["Player"]
            if player not in player2graphs:
                player2graphs[player] = []
            for i in range(1,11):
                graph = row[f"Partida_{i}"]
                if graph is not np.nan:
                    player2graphs[player].append(graph)

        # Create a dictionary that maps players to their predicted classes. Since a player
        # might have more than 1 graph, we set their predicted class as the most frequent
        # predicted class among a player's graphs.
        player2class = {}
        f = lambda lst: max(set(lst), key=lst.count)
        for player, graphs in player2graphs.items():
            if player not in player2class:
                player2class[player] = {
                    "Bartle": [],
                    "Dedication": []
                }
            for graph in graphs:
                result_row = model_results[model_results["graph"] == graph[:-4]]
                if result_row.empty:
                    continue
                bartle_preds = [label for label in self.labels[:4] if result_row[f"{label}_pred"].iloc[0] > 0]
                dedica_preds = [label for label in self.labels[4:] if result_row[f"{label}_pred"].iloc[0] > 0]
                player2class[player]["Bartle"].append("/".join(bartle_preds))
                player2class[player]["Dedication"].append("/".join(dedica_preds))
            player2class[player]["Bartle"] = self.get_repeated_mode(player2class[player]["Bartle"])
            player2class[player]["Dedication"] = self.get_repeated_mode(player2class[player]["Dedication"])

        # Create dataframe with predictions            
        model_pred_df = pd.DataFrame.from_dict(player2class, orient='index').reset_index().rename({
            "index":"Player"}, axis=1)
        model_pred_df = model_pred_df[model_pred_df['Bartle'] != '']
        model_pred_df.rename(columns={
            "Bartle" : "PING Bartle",
            "Dedication" : "PING Dedication"
        }, inplace=True)

        # Merge ground truth and prediction dataframes
        ground_truth_redux = self.ground_truth.iloc[:,:5]
        merged_df = ground_truth_redux.merge(model_pred_df, on="Player")
        merged_df = merged_df[merged_df["Bicalho's Classification"]!="Not available"]
        merged_df["Bicalho Bartle"] = merged_df["Bicalho\'s Classification"].apply(lambda x: x.split(" ")[1])
        merged_df["Bicalho Dedication"] = merged_df["Bicalho\'s Classification"].apply(lambda x: x.split(" ")[0])
        merged_df.drop(columns={
            "Total de Partidas",
            "Bicalho\'s Classification"
        }, inplace=True)
        
        # Apply multilabel binarization on classes columns.
        bartle_mlb = MultiLabelBinarizer()
        bartle_mlb.fit([{"Achiever","Killer"}])
        dedica_mlb = MultiLabelBinarizer()
        dedica_mlb.fit([{"Hardcore", "Casual"}])
        bartle_gt_encoded = bartle_mlb.transform([x.split("/") for x in list(merged_df["Bartle Classification"])])
        dedica_gt_encoded = dedica_mlb.transform([x.split("/") for x in list(merged_df["Gamer Dedication"])])
        bartle_pg_encoded = bartle_mlb.transform([x.split("/") for x in list(merged_df["PING Bartle"])])
        dedica_pg_encoded = dedica_mlb.transform([x.split("/") for x in list(merged_df["PING Dedication"])])
        bartle_bc_encoded = bartle_mlb.transform([x.split("/") for x in list(merged_df["Bicalho Bartle"])])
        dedica_bc_encoded = dedica_mlb.transform([x.split("/") for x in list(merged_df["Bicalho Dedication"])])
        
        #Log Results
        self.log("Bartle Classification - Bicalho's Classification")
        self.log(classification_report(bartle_gt_encoded, bartle_bc_encoded, target_names=bartle_mlb.classes_))
        self.log("Bartle Classification - PINGUMIL Classification")
        self.log(classification_report(bartle_gt_encoded, bartle_pg_encoded, target_names=bartle_mlb.classes_))
        self.log("Gamer Dedication - Bicalho's Classification")
        self.log(classification_report(dedica_gt_encoded, dedica_bc_encoded, target_names=dedica_mlb.classes_))
        self.log("Gamer Dedication - PINGUMIL Classification")
        self.log(classification_report(dedica_gt_encoded, dedica_pg_encoded, target_names=dedica_mlb.classes_))

    
    def finish(self):
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.timestamp))
        self.log(f"Experiment time: {elapsed}")

if __name__ ==  "__main__":
    k = GPPBaseExperiment()
    dataset = k.read_data()
    print(dataset)
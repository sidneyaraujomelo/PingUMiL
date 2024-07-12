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
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from pingumil.util.util import generate_class_weights
from pingumil.util.pytorchtools import EarlyStopping
import time
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-15

class SSDeathPredBaseExperiment():
    def __init__(
            self,
            dataset_folder="/raid/home/smelo/PingUMiL-pytorch/dataset/SmokeSquadron/ss_hitprediction/preprocessed_graphs",
            dataset_prefix="hitpred",
            model_config="configs/sswinpred_sagemodel.json",
            experiment_tag="base",
            timestamp=None,
            standardization=True,
            epochs=1000,
            cls_epochs=1000,
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
        self.cls_epochs = cls_epochs
        self.patience = patience
        self.log_folder = f"experiment_log/deathpred_{self.experiment_tag}/{self.run_name}"
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        self.output_file = f"{self.log_folder}/{self.run_name}.txt"
        self.train_df = pd.read_csv(
            "/raid/home/smelo/PingUMiL-pytorch/dataset/SmokeSquadron/ss_deathprediction/deathprediction_dataset_5s_train.csv",
            index_col=False)
        self.test_df = pd.read_csv(
            "/raid/home/smelo/PingUMiL-pytorch/dataset/SmokeSquadron/ss_deathprediction/deathprediction_dataset_5s_test.csv",
            index_col=False)
        self.log(f"Experiment {self.timestamp}")
        
    def reset(self, wandb):
        self.wandb = wandb
        if not self.wandb:
            self.run_name = self.timestamp
        else:
            self.run_name = self.wandb.run.name
        self.log_folder = f"experiment_log/deathpred_{self.experiment_tag}/{self.run_name}"
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        self.output_file = f"{self.log_folder}/{self.run_name}.txt"
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
            #print(graph_folder)
            #print(self.dataset_prefix)
            #print(os.path.join(graph_folder, f"{self.dataset_prefix}*-G.json"))
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
            
            #Now, we load the ids from the id maps
            id_maps_file = glob(os.path.join(self.dataset_folder, graph_folder, f"*{graph_id}*-id_map.json"))[0]
            id_map = json.load(open(id_maps_file))
            dataset[graph_folder]["id_map"] = id_map

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
    
    def finish(self):
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.timestamp))
        self.log(f"Experiment time: {elapsed}")
        
    def create_output_dict(self):
        output_dict = {
            "parameters": [],
            "fold" : []
        }
        return output_dict

    def calculate_metrics(self, y, y_hat, average="macro"):
        acc = accuracy_score(y, y_hat)
        prec = precision_score(y, y_hat, average=average)
        rec = recall_score(y, y_hat, average=average)
        f1 = f1_score(y, y_hat, average=average)
        return acc, prec, rec, f1

    def add_metrics_to_output_dict(self, output_dict, param, fold, sample_sets):
        output_dict["parameters"].append(json.dumps(param))
        output_dict["fold"].append(fold)
        for sample_set in sample_sets:
            y, y_hat, set_name = sample_set[0], sample_set[1], sample_set[2]
            acc, prec, rec, f1 = self.calculate_metrics(y, y_hat)
            if f"accuracy_{set_name}" not in output_dict:
                output_dict[f"accuracy_{set_name}"] = []
            if f"precision_{set_name}" not in output_dict:
                output_dict[f"precision_{set_name}"] = []
            if f"recall_{set_name}" not in output_dict:
                output_dict[f"recall_{set_name}"] = []
            if f"f1_{set_name}" not in output_dict:
                output_dict[f"f1_{set_name}"] = []
            output_dict[f"accuracy_{set_name}"].append(acc)
            output_dict[f"precision_{set_name}"].append(prec)
            output_dict[f"recall_{set_name}"].append(rec)
            output_dict[f"f1_{set_name}"].append(f1)
        return output_dict
    
    def split_train_df(self, n_splits=9):
        gss_train_val = GroupShuffleSplit(n_splits=n_splits, train_size=.7, random_state=43)
        return gss_train_val.split(self.train_df.index, self.train_df["class"], self.train_df["source"])
          
    def classification_step(self, device, model, X_trainval, y_trainval, groups_trainval,
                            X_test, y_test, X_test_idx, groups_test, wandb):
        self.log(f"Classification Model: {model}")
        experiment_control_dict = self.create_output_dict()
        output_dict = self.create_output_dict()
        
        gss_train_val = GroupShuffleSplit(n_splits=9, train_size=.7, random_state=43)
        
        for fold, (train_index, val_index) in enumerate(gss_train_val.split(X_trainval, y_trainval, groups_trainval)):
            early_stopping = self.get_early_stopping(patience=self.patience,
                                                    verbose=True,
                                                    prefix="predictor")
            model.reset_parameters()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=wandb.config.lr_clf, weight_decay=wandb.config.weight_decay_clf)
            
            X_train, y_train = X_trainval[train_index], y_trainval[train_index]
            X_val, y_val = X_trainval[val_index], y_trainval[val_index]
            
            X_train = X_train.to(device)
            X_val = X_val.to(device)
            y_train = y_train.view(-1, 1).to(device)
            y_val = y_val.view(-1, 1).to(device)
            
            for epoch in range(1, self.cls_epochs):
                model.train()
                optimizer.zero_grad()
                
                y_hat = model(X_train, sigmoid=False)
                
                #y_train_weight = torch.Tensor(generate_class_weights(y_train)).to(device)
                
                #train_loss = F.binary_cross_entropy_with_logits(y_hat, y_train,
                #                                                pos_weight=y_train_weight)
                train_loss = F.binary_cross_entropy_with_logits(y_hat, y_train)
                y_pred_train = torch.round(torch.sigmoid(y_hat))
                
                train_loss.backward()
                optimizer.step()
                
                model.eval()
                y_hat_val = model(X_val, sigmoid=False)
                val_loss = F.binary_cross_entropy_with_logits(y_hat_val, y_val)
                y_pred_val = torch.round(torch.sigmoid(y_hat_val))
                
                #Detaching all tensors
                np_y_val = y_val.cpu().detach().numpy()
                np_y_pred_val = y_pred_val.cpu().detach().numpy()
                np_y_train = y_train.cpu().detach().numpy()
                np_y_pred_train = y_pred_train.cpu().detach().numpy()
                
                print(fold, epoch, np_y_val.shape, np_y_pred_val.shape)
                
                acc_train, _, _, f1_val = self.calculate_metrics(np_y_train, np_y_pred_train)
                acc_val, p_val, r_val, f1_val = self.calculate_metrics(np_y_val, np_y_pred_val)
                if f1_val is np.NaN:
                    f1_val = 0
                experiment_control_dict = self.add_metrics_to_output_dict(
                    experiment_control_dict, "NONE", fold,  
                    [(np_y_train, np_y_pred_train, "train"), (np_y_val, np_y_pred_val, "val")])
                
                wandb.log({
                    f"train cls loss" : train_loss.item(),
                    f"val cls loss": val_loss.item(),
                    f"train acc" : acc_train,
                    f"val acc": acc_val,
                    f"val_p" : p_val,
                    f"val_r" : r_val,
                    f"val_f1" : f1_val
                })

                early_stopping(train_loss, model)
                if early_stopping.early_stop:
                    print("Early Stopping!")
                    break
        # Now, we train our final model
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=wandb.config.lr_clf, weight_decay=wandb.config.weight_decay)
        X_trainval = X_trainval.to(device)
        y_trainval = y_trainval.view(-1, 1).to(device)
        for _ in range(1, self.cls_epochs):
            model.train()
            optimizer.zero_grad()
            
            y_hat = model(X_trainval, sigmoid=False)
            
            train_loss = F.binary_cross_entropy_with_logits(y_hat, y_trainval)
            y_pred_train = torch.round(torch.sigmoid(y_hat))
            
            train_loss.backward()
            optimizer.step()
            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping!")
                break
        # Finally, we run the final model on the test set
        model.eval()
        X_test = X_test.to(device)
        y_test = y_test.view(-1, 1).to(device)
        y_test_hat = model(X_test, sigmoid=False)
        y_test_hat = torch.round(torch.sigmoid(y_test_hat))
        np_y_test = y_test.cpu().detach().numpy()
        np_y_pred_test = y_test_hat.cpu().detach().numpy()
        acc_test, p_test, r_test, f1_test = self.calculate_metrics(np_y_test, np_y_pred_test)
        output_dict = self.add_metrics_to_output_dict(
            output_dict, "NONE", 0, [(np_y_test, np_y_pred_test, "test")])
        pred_dict = self.create_prediction_dict(X_test_idx, groups_test, y_test, y_test_hat)
        wandb.log({
            "p_test": p_test,
            "r_test" : r_test,
            "f1_test" : f1_test,
            "test_acc" : acc_test
        })
        
        self.save_experiment_dicts(experiment_control_dict, output_dict, pred_dict)
        self.finish()
            
    def save_experiment_dicts(self, experimental_control_dict=None, output_dict=None, pred_dict=None):
        if experimental_control_dict is not None:
            experiment_df = pd.DataFrame.from_dict(experimental_control_dict)
            experiment_df.to_csv(f"{self.log_folder}/dp_pingumil_trainval.csv")
            print(experiment_df.describe())

        if output_dict is not None:
            output_df = pd.DataFrame.from_dict(output_dict)
            output_df.to_csv(f"{self.log_folder}/dp_pingumil_test.csv")
            print(output_df.describe())
        
        if pred_dict is not None:
            class_pred_df = pd.DataFrame.from_dict(pred_dict)
            class_pred_df.to_csv(f"{self.log_folder}/dp_pingumil_test_preds.csv")
            print(class_pred_df.describe())
        
    def create_prediction_dict(self, index, source, y, y_hat):
        pred_dict = {
            "index" : index,
            "source": source,
            "y": y,
            "y_hat" : y_hat
        }
        return pred_dict
        
    def classification_step_tvt(self, device, model, x, y, graph_ids, current_timestamp, wandb):
        self.log(f"Classification Model: {model}")
        average_metrics = { k : [] for k in ["train_loss", "p", "r", "f1", "val_loss", "acc"] }
        
        output_test_dict = {
            "fold" : [],
            "timestamp" : [],
            "accuracy_test" : [],
            "precision_test" : [],
            "recall_test" : [],
            "f1_test" : [],
        }
        
        kf_train_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for test_fold, (train_val_index, test_index) in enumerate(kf_train_val.split(x, y)):
            x_trainval, y_trainval = x[train_val_index], y[train_val_index]
            x_test, y_test = x[test_index], y[test_index]
        
            skf = StratifiedKFold(n_splits=9)
            best_model = None
            best_metrics = {
                "train_loss": np.inf,
                "p": 0,
                "r": 0,
                "acc": 0,
                "f1": 0
            }
            for train_index, val_index in skf.split(x_trainval, y_trainval):
                early_stopping = self.get_early_stopping(patience=self.patience,
                                                    verbose=True,
                                                    prefix="predictor")
                model.reset_parameters()
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=wandb.config.lr_clf, weight_decay=wandb.config.weight_decay)

                for _ in range(1, self.cls_epochs):
                    x_train, x_val = x[train_index], x[val_index]
                    y_train, y_val = y[train_index], y[val_index]
                    
                    x_train = x_train.to(device)
                    x_val = x_val.to(device)
                    y_train = y_train.view(-1, 1).to(device)
                    y_val = y_val.view(-1, 1).to(device)
                    
                    model.train()
                    optimizer.zero_grad()
                    
                    y_hat = model(x_train, sigmoid=False)
                    
                    #y_train_weight = torch.Tensor(generate_class_weights(y_train)).to(device)
                    
                    #train_loss = F.binary_cross_entropy_with_logits(y_hat, y_train,
                    #                                                pos_weight=y_train_weight)
                    train_loss = F.binary_cross_entropy_with_logits(y_hat, y_train)
                    y_pred_train = torch.round(torch.sigmoid(y_hat))
                    
                    train_loss.backward()
                    optimizer.step()
                    
                    model.eval()
                    y_hat_val = model(x_val, sigmoid=False)
                    val_loss = F.binary_cross_entropy_with_logits(y_hat_val, y_val)
                    y_pred_val = torch.round(torch.sigmoid(y_hat_val))
                    
                    #print(f"{epoch}: Train loss: {train_loss.item()}")
                    #print(f"{epoch}: Test loss: {val_loss.item()}")
                    #Detaching all tensors
                    np_y_val = y_val.cpu().detach().numpy()
                    np_y_pred_val = y_pred_val.cpu().detach().numpy()
                    np_y_train = y_train.cpu().detach().numpy()
                    np_y_pred_train = y_pred_train.cpu().detach().numpy()
                    
                    val_p = precision_score(np_y_pred_val,
                                        np_y_val,
                                        average="macro")
                    val_r = recall_score(np_y_pred_val,
                                    np_y_val,
                                    average="macro")
                    val_f1 = f1_score(np_y_pred_val,
                                np_y_val,
                                average="macro")
                    
                    train_acc = accuracy_score(np_y_pred_train,
                                            np_y_train)
                    val_acc = accuracy_score(np_y_pred_val,
                                            np_y_val)
                    wandb.log({
                        f"train cls loss" : train_loss.item(),
                        f"val cls loss": val_loss.item(),
                        f"train acc" : train_acc,
                        f"val acc": val_acc,
                        f"val_p" : val_p,
                        f"val_r" : val_r,
                        f"val_f1" : val_f1
                    })
                                    
                    if (val_f1 > best_metrics["f1"]):
                        best_metrics["train_loss"] = train_loss.cpu().item()
                        best_metrics["r"] = val_r
                        best_metrics["p"] = val_p
                        best_metrics["f1"] = val_f1
                        best_metrics["val_loss"] = val_loss.cpu().item()
                        best_metrics["acc"] = val_acc
                        best_model = model.state_dict()

                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        print("Early Stopping!")
                        break
            model.load_state_dict(best_model)
            model.eval()
            
            x_test = x_test.to(device)
            y_test = y_test.view(-1, 1).to(device)
                    
            y_hat_test = model(x_test, sigmoid=False)
            test_loss = F.binary_cross_entropy_with_logits(y_hat_test, y_test)
            y_pred_test = torch.round(torch.sigmoid(y_hat_test))
            np_y_test = y_test.cpu().detach().numpy()
            np_y_pred_test = y_pred_test.cpu().detach().numpy()
                    
            p = precision_score(np_y_pred_test,
                                np_y_test,
                                average="macro")
            r = recall_score(np_y_pred_test,
                                np_y_test,
                                average="macro")
            f1 = f1_score(np_y_pred_test,
                            np_y_test,
                            average="macro") 
            test_acc = accuracy_score(np_y_pred_test,
                                        np_y_test)
            
            output_test_dict["fold"].append(test_fold)
            output_test_dict["timestamp"].append(current_timestamp)
            output_test_dict["accuracy_test"].append(test_acc)
            output_test_dict["precision_test"].append(p)
            output_test_dict["recall_test"].append(r)
            output_test_dict["f1_test"].append(f1)
            wandb.log({
                "p_test": p,
                "r_test" : r,
                "f1_test" : f1,
                "test_acc" : test_acc
            })
        self.log(f"timestamp: {current_timestamp}")
        output_df = pd.DataFrame.from_dict(output_test_dict)
        results_path = os.path.join(self.log_folder, f"results.csv")
        output_df.to_csv(results_path, index=False)
        self.finish()
        
if __name__ ==  "__main__":
    k = SSDeathPredBaseExperiment()
    dataset = k.read_data()
    print(len(dataset))
from pyts.multivariate.classification import MultivariateClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import ParameterGrid, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score)
from skfeature.function.statistical_based import CFS
from statistics import mean 
import json
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def calculate_transition_matrices(X, n_states, class_labels, match_column, class_column):
    # Number of classes in Y
    n_classes = 2
    # Initialize dictionary to store transition matrices for each column
    transition_matrices = {}

    # For each column in X
    for col in X.columns[:-2]:
        #print(col)
        # Initialize transition matrix for the current column
        transition_matrix_col = np.zeros((n_classes, n_states, n_states))
        
        for match in X.iloc[:,-2].unique():   
            # Filter time series data for the current class
            #print(class_labels[c])
            X_c = X.loc[X[match_column]==match]
            #print(class_labels)
            c = class_labels.index(X_c[class_column].values[0])
            X_c = X_c[col]
            # Count the transitions to other states
            for t in range(1, len(X_c)):
                prev_state = X_c.iloc[t - 1]
                current_state = X_c.iloc[t]
                transition_matrix_col[c, int(prev_state), int(current_state)] += 1
        # Normalize the transition matrices
        
        transition_matrix_col /= np.sum(transition_matrix_col, axis=2, keepdims=True)
        
        # Replace NaN values with 1/n_states (this happens when a state does not appear in the data)
        transition_matrix_col = np.nan_to_num(transition_matrix_col, nan=1.0/n_states)
        #print(transition_matrix_col)
        # Store the transition matrix for the current column in the dictionary
        transition_matrices[col] = transition_matrix_col

    return transition_matrices

def calculate_posterior_probabilities(X, transition_matrices, labels):
    # Number of classes in Y
    n_classes = len(labels)
    
    y_hat = []
    
    for X_sample in X:
        prior_Y = [1/n_classes]*n_classes
    
        # Initialize dictionary to store posterior probabilities for each class in Y
        posterior_probs = {}
    
        # For each class in Y
        for c in range(n_classes):
            # Initialize posterior probability for the current class
            posterior_prob_c = prior_Y[c]
            
            # For each column in X
            for col, transition_matrix_col in transition_matrices.items():
                # Get the transition matrix for the current column and class
                transition_matrix_c = transition_matrix_col[c]
                
                # Get the states for the current column in X_new
                states = X_sample[col].values
                
                # Update the posterior probability using the transition probabilities for the states
                for t in range(1, len(states)):
                    prev_state = states[t - 1]
                    current_state = states[t]
                    posterior_prob_c *= transition_matrix_c[prev_state, current_state]
            
            # Store the posterior probability for the current class in the dictionary
            posterior_probs[c] = posterior_prob_c
    
        # Normalize the posterior probabilities
        sum_posterior_probs = sum(posterior_probs.values())
        #print(sum_posterior_probs)
        for c in posterior_probs:
            posterior_probs[c] /= sum_posterior_probs
        label_index = np.argmax(list(posterior_probs.values()))
        y_hat.append(labels[label_index])
    return y_hat

def create_output_dict():
    output_dict = {
        "parameters": [],
        "fold" : []
    }
    return output_dict

def create_prediction_dict():
    pred_dict = {
        "parameters": [],
        "index" : [],
        "source" : [],
        "y" : [],
        "y_hat" : []
    }
    return pred_dict

def create_prediction_dict(param, index, source, y, y_hat):
    pred_dict = {
        "parameters": [param]*len(index),
        "index" : index,
        "source": source,
        "y": y,
        "y_hat" : y_hat
    }
    return pred_dict

def calculate_metrics(y, y_hat, average="macro"):
    acc = accuracy_score(y, y_hat)
    prec = precision_score(y, y_hat, average=average)
    rec = recall_score(y, y_hat, average=average)
    f1 = f1_score(y, y_hat, average=average)
    return acc, prec, rec, f1

def add_metrics_to_output_dict(output_dict, param, fold, sample_sets):
    output_dict["parameters"].append(json.dumps(param))
    output_dict["fold"].append(fold)
    for sample_set in sample_sets:
        y, y_hat, set_name = sample_set[0], sample_set[1], sample_set[2]
        acc, prec, rec, f1 = calculate_metrics(y, y_hat)
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

def get_model(model_name, param):
    if model_name == "lr":
        lr_ridge = RidgeClassifier(**param)
        clf = MultivariateClassifier(lr_ridge)
        return clf
    if model_name == "rf":
        rf = RandomForestClassifier(**param)
        clf = MultivariateClassifier(rf)
        return clf
    if model_name == "lgb":
        base_clf = LGBMClassifier(**param, verbose=-1)
        clf = MultivariateClassifier(base_clf)
        return clf

# Function to filter out the first five rows of each group
def filter_first_five(group):
    return group.iloc[5:]

def train_test_routine(model_name, trainval_df, test_df, parameters, data_columns, experiment_control_dict,
                        output_dict, class_labels=None, match_column=None, class_column=None):
    
    gss_train_val = GroupShuffleSplit(n_splits=9, train_size=.7, random_state=43)
    best_param = None
    best_avg_f1 = 0
    for param in list(ParameterGrid(parameters)):
        datapoints_idx = trainval_df.index
        groups = trainval_df.loc[datapoints_idx, match_column].values
        classes = trainval_df.loc[datapoints_idx, class_column].values
        f1_val_toavg = []
        for fold, (train_index, val_index) in enumerate(gss_train_val.split(datapoints_idx, classes, groups)):
            if len(val_index) == 0:
                continue
            train_df = trainval_df.loc[datapoints_idx[train_index]]
            val_df = trainval_df.loc[datapoints_idx[val_index]]
            
            X_train, y_train, _ = get_data(model_name, train_df, data_columns, match_column, class_column)
            X_val, y_val, _ = get_data(model_name, val_df, data_columns, match_column, class_column)
        
            if model_name == "asm":
                X_train_df = pd.concat(X_train, axis=0)    
                clf = calculate_transition_matrices(X_train_df, 24, class_labels, match_column, class_column)
                y_train_hat = calculate_posterior_probabilities(X_train, clf, class_labels)
                y_val_hat = calculate_posterior_probabilities(X_val, clf, class_labels)
            else:
                print(fold, param)
                clf = get_model(model_name, param)
                clf.fit(X_train, y_train)
                y_train_hat = clf.predict(X_train)
                y_val_hat = clf.predict(X_val)
            _, _, _, f1_val = calculate_metrics(y_val, y_val_hat)
            if f1_val is np.NaN:
                f1_val = 0
            experiment_control_dict = add_metrics_to_output_dict(
                experiment_control_dict, param, fold,  
                [(y_train, y_train_hat, "train"), (y_val, y_val_hat, "val")])
            f1_val_toavg.append(f1_val)
        if mean(f1_val_toavg) > best_avg_f1:
            best_avg_f1 = mean(f1_val_toavg)
            best_param = param
    
    X_test, y_test, X_idx = get_data(model_name, test_df, data_columns, match_column, class_column)
    y_test_hat = None
    X_trainval, y_trainval, _ = get_data(model_name, trainval_df, data_columns, match_column, class_column)
    if model_name == "asm":
        X_trainval_df = pd.concat(X_trainval, axis=0)
        best_clf = calculate_transition_matrices(X_trainval_df, 24, class_labels, match_column, class_column)
        y_test_hat = calculate_posterior_probabilities(X_test, best_clf, class_labels)
    else:
        best_clf = get_model(model_name, best_param)
        best_clf.fit(X_trainval, y_trainval)
        y_test_hat = best_clf.predict(X_test)
    output_dict = add_metrics_to_output_dict(
        output_dict, param, "test",
        [(y_test, y_test_hat, "test")]
    )
    pred_dict = create_prediction_dict(best_param, X_idx, test_df.loc[X_idx,match_column], y_test, y_test_hat)
    return experiment_control_dict, output_dict, pred_dict

def get_data(model_name, df, data_columns, match_column, class_column):
    ts_index = df.groupby(match_column).apply(filter_first_five).index.get_level_values(1)
    X = []
    y = []
    for i in range(len(ts_index)):
        new_x = None
        if model_name == "asm":
            new_x = df.loc[ts_index[i]-4:ts_index[i], data_columns+[match_column, class_column]]
        else:
            new_x = df.loc[ts_index[i]-4:ts_index[i], data_columns]
        if (len(new_x)) == 5:
            X.append(new_x)      
            y.append(df.loc[ts_index[i], class_column])
    return X, y, ts_index

def run(model_name, add_cluster=False, num_clusters=5, base_clustering="kmeans", run_csf=True, parameters = None):
    output_base_filename = f"dp_{model_name}_sk10f_lr_csf" if run_csf else f"{model_name}_sk10f_lr"
    print(f"Execution: {output_base_filename}")
    if add_cluster:
        print(f"Clustering: {base_clustering} - k={num_clusters}")
        cluster_file = f"/raid/home/smelo/PingUMiL-pytorch/player_profiles_winpred/player_profiles_sswinpredllr_{base_clustering}{num_clusters}.csv"
        cluster_df = pd.read_csv(cluster_file, index_col=0)
        cluster_df["graph"] = cluster_df["graph"].apply(lambda x : f"{x}.xml")

    match_column = "source"
    timestamp_column = "timestamp"
    class_columns = ["willdieplayer01", "willdieplayer02"]
    ignore_columns = ["node_idxplayer01", "node_idxplayer02"]
    non_data_columns = [match_column, timestamp_column]+class_columns+ignore_columns
    
    train_df = pd.read_csv("/raid/home/smelo/PingUMiL-pytorch/dataset/SmokeSquadron/ss_deathprediction/deathprediction_dataset_5s_train.csv", index_col=0)
    test_df = pd.read_csv("/raid/home/smelo/PingUMiL-pytorch/dataset/SmokeSquadron/ss_deathprediction/deathprediction_dataset_5s_test.csv", index_col=0)
    if add_cluster:
        train_df = train_df.merge(cluster_df, how="inner", left_on=['source','timestamp'], right_on=['graph','timestamp'])
        train_df = train_df.drop(columns=["graph"])
        test_df = test_df.merge(cluster_df, how="inner", left_on=['source','timestamp'], right_on=['graph','timestamp'])
        test_df = test_df.drop(columns=["graph"])
    
    data_columns = [x for x in train_df.columns if x not in non_data_columns]
    
    for class_column in class_columns:
        if model_name == "asm":
            labels = list(range(24))
            all_data_df = pd.concat([train_df,test_df])
            for data_col in data_columns:
                _, bins = pd.cut(all_data_df[data_col], bins=24, labels=labels, retbins=True)
                train_df[data_col] = pd.cut(train_df[data_col], bins, labels=labels)
                test_df[data_col] = pd.cut(test_df[data_col], bins, labels=labels)
            if add_cluster:
                data_columns = [x for x in data_columns if 'dif' in x]+['player01_cluster','player02_cluster']
            else:
                data_columns = [x for x in data_columns if 'dif' in x]
        elif run_csf:
            try:
                features = CFS.cfs(train_df.loc[:,data_columns].to_numpy(), train_df.loc[:,class_column])
                data_columns = [data_columns[feature] for feature in features]
            except Exception as e:
                print(e)
            
        
        class_labels = list(train_df[class_column].unique())
    
        experiment_control_dict = create_output_dict()
        output_dict = create_output_dict()
        experiment_control_dict, output_dict, pred_dict = train_test_routine(model_name, train_df, test_df,
                                                                parameters, data_columns, experiment_control_dict,
                                                                output_dict, class_labels, match_column, class_column)
            
        experiment_df = pd.DataFrame.from_dict(experiment_control_dict)
        if add_cluster:
            experiment_df.to_csv(f"{output_base_filename}_{base_clustering}k{num_clusters}_{class_column}_trainval.csv")
        else:
            experiment_df.to_csv(f"{output_base_filename}_{class_column}_trainval.csv")
        print(experiment_df.describe())

        output_df = pd.DataFrame.from_dict(output_dict)
        if add_cluster:
            output_df.to_csv(f"{output_base_filename}_{base_clustering}k{num_clusters}_{class_column}_test.csv")
        else:
            output_df.to_csv(f"{output_base_filename}_{class_column}_test.csv")
        print(output_df.describe())
        
        class_pred_df = pd.DataFrame.from_dict(pred_dict)
        if add_cluster:
            class_pred_df.to_csv(f"{output_base_filename}_{base_clustering}k{num_clusters}_{class_column}_test_preds.csv")
        else:
            class_pred_df.to_csv(f"{output_base_filename}_{class_column}_test_preds.csv")
    
if __name__ == "__main__":
    param_dict = {
        "lr" : {'alpha': [1, 0.1, 0.01]},
        "rf" : {'n_estimators': [10, 50]},
        "lgb" : {'n_estimators': [10, 50], 'num_leaves': [5, 10], 'objective': ['binary']},
        "asm" : {"ASM" : ["ASM"]}
    }
    '''param_dict = {
        "asm" : {"ASM" : "ASM"}
    }'''
    for model_name in ["rf", "lgb", "lr", "asm"]:
        for run_csf in [True, False]:
            if model_name == "asm" and run_csf == True:
                continue
            params = param_dict[model_name]
            run(model_name, add_cluster=False, parameters=params, run_csf=run_csf)
            for base_clustering in ["kmeans", "spectralg55k"]:
                for i in [5, 10, 20]:
                    run(model_name, add_cluster=True, num_clusters=i, base_clustering=base_clustering, parameters=params, run_csf=run_csf)
from pyts.multivariate.classification import MultivariateClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid, StratifiedKFold
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
        "fold" : [],
        "timestamp" : [] 
    }
    return output_dict

def calculate_metrics(y, y_hat, average="macro"):
    acc = accuracy_score(y, y_hat)
    prec = precision_score(y, y_hat, average=average)
    rec = recall_score(y, y_hat, average=average)
    f1 = f1_score(y, y_hat, average=average)
    return acc, prec, rec, f1

def add_metrics_to_output_dict(output_dict, param, fold, timestamp, sample_sets):
    output_dict["parameters"].append(json.dumps(param))
    output_dict["fold"].append(fold)
    output_dict["timestamp"].append(timestamp)
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
    
def train_test_routine(model_name, X, y, timestamp, parameters, experiment_control_dict, 
                       output_dict, class_labels=None, match_column=None, class_column=None):
    if len(X) <= 10:
        return experiment_control_dict, output_dict
    kf_train_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    if y.count('player01') <= 10 or y.count('player02') <= 10:
        return experiment_control_dict, output_dict
    for test_fold, (train_val_index, test_index) in enumerate(kf_train_val.split(X, y)):
        x_trainval, y_trainval = [X[i] for i in train_val_index], [y[i] for i in train_val_index]
        x_test, y_test = [X[i] for i in test_index], [y[i] for i in test_index]
        
        skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
        if y_trainval.count('player01') <= 9 or y_trainval.count('player02') <= 9:
            return experiment_control_dict, output_dict
        for param in list(ParameterGrid(parameters)):
            best_f1_val = 0
            best_clf = None
            for fold, (train_index, val_index) in enumerate(skf.split(x_trainval, y_trainval)):
                X_train, y_train = [X[i] for i in train_index], [y[i] for i in train_index]
                X_val, y_val = [X[i] for i in val_index], [y[i] for i in val_index]
                if model_name == "asm":
                    X_train_df = pd.concat(X_train, axis=0)    
                    clf = calculate_transition_matrices(X_train_df, 24, class_labels, match_column, class_column)
                    y_train_hat = calculate_posterior_probabilities(X_train, clf, class_labels)
                    y_val_hat = calculate_posterior_probabilities(X_val, clf, class_labels)
                else:
                    clf = get_model(model_name, param)
                    clf.fit(X_train, y_train)
                    y_train_hat = clf.predict(X_train)
                    y_val_hat = clf.predict(X_val)
                _, _, _, f1_val = calculate_metrics(y_val, y_val_hat)
                experiment_control_dict = add_metrics_to_output_dict(
                    experiment_control_dict, param, fold, timestamp, 
                    [(y_train, y_train_hat, "train"), (y_val, y_val_hat, "val")])
                if f1_val > best_f1_val:
                    best_f1_val = f1_val
                    best_clf = clf
            y_test_hat = None
            if model_name == "asm":
                X_trainval_df = pd.concat(x_trainval, axis=0)
                best_clf = calculate_transition_matrices(X_trainval_df, 24, class_labels, match_column, class_column)
                y_test_hat = calculate_posterior_probabilities(x_test, best_clf, class_labels)
            else:
                best_clf.fit(x_trainval, y_trainval)
                y_test_hat = best_clf.predict(x_test)
            output_dict = add_metrics_to_output_dict(
                output_dict, param, test_fold, timestamp,
                [(y_test, y_test_hat, "test")]
            )
    return experiment_control_dict, output_dict

def get_data(model_name, df, timestamp, data_columns, match_column, class_column):
    timestamp_subset = df[df['timestamp'] == timestamp]
    ts_index = [x for x in timestamp_subset.index]
    X = []
    y = []
    for i in range(len(timestamp_subset)):
        if model_name == "asm":
            X.append(df.loc[ts_index[i]-4:ts_index[i], data_columns+[match_column, class_column]])
        else:
            X.append(df.loc[ts_index[i]-4:ts_index[i], data_columns])
        y.append(df.loc[ts_index[i], class_column])
    return X, y

def run(model_name, add_cluster=False, num_clusters=5, base_clustering="kmeans", run_csf=True, parameters = None):
    output_base_filename = f"{model_name}_sk10f_lr_csf" if run_csf else f"{model_name}_sk10f_lr"
    print(f"Execution: {output_base_filename}")
    if add_cluster:
        print(f"Clustering: {base_clustering} - k={num_clusters}")
        cluster_file = f"player_profiles_sswinpred_{base_clustering}k{num_clusters}.csv"
        cluster_df = pd.read_csv(cluster_file, index_col=0)
        cluster_df["graph"] = cluster_df["graph"].apply(lambda x : f"{x}.xml")

    match_column = "source"
    timestamp_column = "timestamp"
    class_column = "winner"
    non_data_columns = [match_column, timestamp_column, class_column]

    df = pd.read_csv("dataset/SmokeSquadron/ss_winprediction/lpmp_dataset_5s.csv").drop(columns=["Unnamed: 0"])
    if add_cluster:
        df = df.merge(cluster_df, how="inner", left_on=['source','timestamp'], right_on=['graph','timestamp'])
        df = df.drop(columns=["graph"])
    
    data_columns = [x for x in df.columns if x not in non_data_columns]
    if model_name == "asm":
        labels = list(range(24))
        for data_col in data_columns:
            df[data_col] = pd.cut(df[data_col], bins=24, labels=labels)
        if add_cluster:
            data_columns = [x for x in data_columns if 'dif' in x]+['player01_cluster','player02_cluster']
        else:
            data_columns = [x for x in data_columns if 'dif' in x]
    elif run_csf:
        features = CFS.cfs(df.loc[:,data_columns].to_numpy(), df.loc[:,class_column])
        data_columns = [data_columns[feature] for feature in features]
    class_labels = list(df[class_column].unique())
    
    data_index = []
    y = []
    # Get the unique sources in the DataFrame
    unique_sources = df[match_column].unique()

    # Iterate over each source
    for source in unique_sources:
        source_subset = df[df[match_column] == source]
        data_index.append(list(source_subset.index[4:]))
        y.append(source_subset.loc[source_subset.index[0],class_column])
    assert len(data_index) == len(y)
    
    experiment_control_dict = create_output_dict()
    output_dict = create_output_dict()
    unique_timestamps = df['timestamp'].unique()[4:]
    for timestamp in unique_timestamps:
        X, y = get_data(model_name, df, timestamp, data_columns, match_column, class_column)
        experiment_control_dict, output_dict = train_test_routine(model_name, X, y, timestamp, 
                                                                  parameters, experiment_control_dict,
                                                                  output_dict, class_labels, match_column, class_column)
        
    experiment_df = pd.DataFrame.from_dict(experiment_control_dict)
    if add_cluster:
        experiment_df.to_csv(f"{output_base_filename}_{base_clustering}k{num_clusters}_trainval.csv")
    else:
        experiment_df.to_csv(f"{output_base_filename}_trainval.csv")
    print(experiment_df.describe())

    output_df = pd.DataFrame.from_dict(output_dict)
    if add_cluster:
        output_df.to_csv(f"{output_base_filename}_{base_clustering}k{num_clusters}_test.csv")
    else:
        output_df.to_csv(f"{output_base_filename}_test.csv")
    print(output_df.describe())
    
if __name__ == "__main__":
    '''param_dict = {
        "lr" : {'alpha': [1, 0.1, 0.01]},
        "rf" : {'n_estimators': [10, 50]},
        "lgb" : {'n_estimators': [10, 50], 'num_leaves': [5, 10], 'objective': ['binary']},
        "asm" : {"ASM" : "ASM"}
    }'''
    param_dict = {
        "asm" : {"ASM" : "ASM"}
    }
    #for model_name in ["rf", "lgb", "lr"]:
    for model_name in ["asm"]:
        run_csf = True
        if model_name == "asm":
            run_csf = False
        params = param_dict[model_name]
        run(model_name, add_cluster=False, parameters=params, run_csf=run_csf)
        #run(model_name, add_cluster=True, num_clusters=5, base_clustering="kmeans", parameters=params, run_csf=run_csf)
        #for i in [5, 10, 20]:
        #    run(model_name, add_cluster=True, num_clusters=i, base_clustering="spectralg55", parameters=params, run_csf=run_csf)
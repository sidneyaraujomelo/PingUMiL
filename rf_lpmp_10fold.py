from pyts.multivariate.classification import MultivariateClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score)
from skfeature.function.statistical_based import CFS
import json
import warnings
warnings.filterwarnings('ignore')

ADD_CLUSTER = True
if ADD_CLUSTER:
    num_clusters = 5
    base_clustering = "kmeans"
    cluster_file = f"player_profiles_sswinpred_{base_clustering}k{num_clusters}.csv"
    cluster_df = pd.read_csv(cluster_file, index_col=0)
    cluster_df["graph"] = cluster_df["graph"].apply(lambda x : f"{x}.xml")

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

match_column = "source"
timestamp_column = "timestamp"
class_column = "winner"
non_data_columns = [match_column, timestamp_column, class_column]

df = pd.read_csv("dataset/SmokeSquadron/ss_winprediction/lpmp_dataset_5s.csv").drop(columns=["Unnamed: 0"])
if ADD_CLUSTER:
    df = df.merge(cluster_df, how="inner", left_on=['source','timestamp'], right_on=['graph','timestamp'])
    df = df.drop(columns=["graph"])
data_columns = [x for x in df.columns if x not in non_data_columns]

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

parameters = {'n_estimators': [10, 50]}

experiment_control_dict = create_output_dict()
output_dict = create_output_dict()
unique_timestamps = df['timestamp'].unique()[4:]
for timestamp in unique_timestamps:
    timestamp_subset = df[df['timestamp'] == timestamp]
    ts_index = [x for x in timestamp_subset.index]
    X = []
    y = []
    for i in range(len(timestamp_subset)):
        X.append(df.loc[ts_index[i]-4:ts_index[i], data_columns])
        y.append(df.loc[ts_index[i], class_column])
    #print(len(X), len(y))
    if len(X) <= 10:
        continue
    kf_train_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    if y.count('player01') <= 10 or y.count('player02') <= 10:
            continue
    for test_fold, (train_val_index, test_index) in enumerate(kf_train_val.split(X, y)):
        x_trainval, y_trainval = [X[i] for i in train_val_index], [y[i] for i in train_val_index]
        x_test, y_test = [X[i] for i in test_index], [y[i] for i in test_index]
        
        skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
        best_f1_val = 0
        best_clf = None
        if y_trainval.count('player01') <= 9 or y_trainval.count('player02') <= 9:
            continue
        for fold, (train_index, val_index) in enumerate(skf.split(x_trainval, y_trainval)):
            X_train, y_train = [X[i] for i in train_index], [y[i] for i in train_index]
            X_val, y_val = [X[i] for i in val_index], [y[i] for i in val_index]
            for param in list(ParameterGrid(parameters)):
                lr_ridge = RandomForestClassifier(**param)
                clf = MultivariateClassifier(lr_ridge)
                clf.fit(X_train, y_train)
                y_train_hat = clf.predict(X_train)
                y_val_hat = clf.predict(X_val)
                acc_train, prec_train, rec_train, f1_train = calculate_metrics(y_train, y_train_hat)
                acc_val, prec_val, rec_val, f1_val = calculate_metrics(y_val, y_val_hat)
                experiment_control_dict = add_metrics_to_output_dict(
                    experiment_control_dict, param, fold, timestamp, 
                    [(y_train, y_train_hat, "train"), (y_val, y_val_hat, "val")])
                if f1_val > best_f1_val:
                    best_f1_val = f1_val
                    best_clf = clf
                    best_params = param
        y_test_hat = best_clf.predict(x_test)
        acc_test, prec_test, rec_test, f1_test = calculate_metrics(y_test, y_test_hat)
        output_dict = add_metrics_to_output_dict(
            output_dict, best_params, test_fold, timestamp,
            [(y_test, y_test_hat, "test")]
        )
    
experiment_df = pd.DataFrame.from_dict(experiment_control_dict)
output_base_filename = "rf_sk10f"
if ADD_CLUSTER:
    experiment_df.to_csv(f"{output_base_filename}_{base_clustering}k{num_clusters}_trainval.csv")
else:
    experiment_df.to_csv(f"{output_base_filename}_trainval.csv")
print(experiment_df.describe())

output_df = pd.DataFrame.from_dict(output_dict)
if ADD_CLUSTER:
    output_df.to_csv(f"{output_base_filename}_{base_clustering}k{num_clusters}_test.csv")
else:
    output_df.to_csv(f"{output_base_filename}_test.csv")
print(output_df.describe())

features = CFS.cfs(df.loc[:,data_columns].to_numpy(), df.loc[:,class_column])
print(features)

csf_data_columns = [data_columns[feature] for feature in features]
print(csf_data_columns)

experiment_control_dict = create_output_dict()
output_dict = create_output_dict()
unique_timestamps = df['timestamp'].unique()[4:]
for timestamp in unique_timestamps:
    timestamp_subset = df[df['timestamp'] == timestamp]
    ts_index = [x for x in timestamp_subset.index]
    X = []
    y = []
    for i in range(len(timestamp_subset)):
        X.append(df.loc[ts_index[i]-4:ts_index[i], csf_data_columns])
        y.append(df.loc[ts_index[i], class_column])
    #print(len(X), len(y))
    if len(X) <= 10:
        continue
    kf_train_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    if y.count('player01') <= 10 or y.count('player02') <= 10:
        continue
    for test_fold, (train_val_index, test_index) in enumerate(kf_train_val.split(X, y)):
        x_trainval, y_trainval = [X[i] for i in train_val_index], [y[i] for i in train_val_index]
        x_test, y_test = [X[i] for i in test_index], [y[i] for i in test_index]
        
        skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
        best_f1_val = 0
        best_clf = None
        if y_trainval.count('player01') <= 9 or y_trainval.count('player02') <= 9:
            continue
        for fold, (train_index, val_index) in enumerate(skf.split(x_trainval, y_trainval)):
            X_train, y_train = [X[i] for i in train_index], [y[i] for i in train_index]
            X_val, y_val = [X[i] for i in val_index], [y[i] for i in val_index]
            for param in list(ParameterGrid(parameters)):
                base_clf = RandomForestClassifier(**param)
                clf = MultivariateClassifier(base_clf)
                clf.fit(X_train, y_train)
                y_train_hat = clf.predict(X_train)
                y_val_hat = clf.predict(X_val)
                acc_train, prec_train, rec_train, f1_train = calculate_metrics(y_train, y_train_hat)
                acc_val, prec_val, rec_val, f1_val = calculate_metrics(y_val, y_val_hat)
                experiment_control_dict = add_metrics_to_output_dict(
                    experiment_control_dict, param, fold, timestamp, 
                    [(y_train, y_train_hat, "train"), (y_val, y_val_hat, "val")])
                if f1_val > best_f1_val:
                    best_f1_val = f1_val
                    best_clf = clf
                    best_params = param
        y_test_hat = best_clf.predict(x_test)
        acc_test, prec_test, rec_test, f1_test = calculate_metrics(y_test, y_test_hat)
        output_dict = add_metrics_to_output_dict(
            output_dict, best_params, test_fold, timestamp,
            [(y_test, y_test_hat, "test")]
        )
    
experiment_df = pd.DataFrame.from_dict(experiment_control_dict)
if ADD_CLUSTER:
    experiment_df.to_csv(f"{output_base_filename}_csf_{base_clustering}k{num_clusters}_trainval.csv")
else:
    experiment_df.to_csv(f"{output_base_filename}_csf_trainval.csv")
print(experiment_df.describe())

output_df = pd.DataFrame.from_dict(output_dict)
if ADD_CLUSTER:
    output_df.to_csv(f"{output_base_filename}_csf_{base_clustering}k{num_clusters}_test.csv")
else:
    output_df.to_csv(f"{output_base_filename}_csf_test.csv")
print(output_df.describe())
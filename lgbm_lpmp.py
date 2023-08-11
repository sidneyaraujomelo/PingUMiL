from pyts.classification import BOSSVS
from pyts.multivariate.classification import MultivariateClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score)
from skfeature.function.statistical_based import CFS
import json

def create_output_dict():
    output_dict = {
        "parameters": [],
        "fold" : [],
        "timestamp" : [],
        "accuracy_train" : [],
        "precision_train" : [],
        "recall_train" : [],
        "f1_train": [],
        "accuracy_test" : [],
        "precision_test" : [],
        "recall_test" : [],
        "f1_test": []    
    }
    return output_dict

def add_metrics_to_output_dict(output_dict, param, fold, timestamp, y_train, y_train_hat, y_test, y_test_hat):
    acc_train = accuracy_score(y_train, y_train_hat)
    prec_train = precision_score(y_train, y_train_hat, average="macro")
    rec_train = recall_score(y_train, y_train_hat, average="macro")
    f1_train = f1_score(y_train, y_train_hat, average="macro")
    output_dict["parameters"].append(json.dumps(param))
    output_dict["fold"].append(fold)
    output_dict["timestamp"].append(timestamp)
    output_dict["accuracy_train"].append(acc_train)
    output_dict["precision_train"].append(prec_train)
    output_dict["recall_train"].append(rec_train)
    output_dict["f1_train"].append(f1_train)
    acc_test = accuracy_score(y_test, y_test_hat)
    prec_test = precision_score(y_test, y_test_hat, average="macro")
    rec_test = recall_score(y_test, y_test_hat, average="macro")
    f1_test = f1_score(y_test, y_test_hat, average="macro")
    output_dict["accuracy_test"].append(acc_test)
    output_dict["precision_test"].append(prec_test)
    output_dict["recall_test"].append(rec_test)
    output_dict["f1_test"].append(f1_test)
    return output_dict

match_column = "source"
timestamp_column = "timestamp"
class_column = "winner"
non_data_columns = [match_column, timestamp_column, class_column]

df = pd.read_csv("dataset/SmokeSquadron/ss_winprediction/lpmp_dataset_5s.csv").drop(columns=["Unnamed: 0"])
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

parameters = {'n_estimators': [10, 50, 100], 'num_leaves': [5, 10, 20], 'objective': ['binary']}

sgkf = StratifiedGroupKFold(n_splits=3)
sgkf.get_n_splits(data_index, y)
output_dict = create_output_dict()
for fold, (train_index, test_index) in enumerate(sgkf.split(data_index, y, unique_sources)):
    print(f"fold {fold}")
    index_train = [x for idx in train_index for x in data_index[idx]]
    index_test = [x for idx in test_index for x in data_index[idx]]
    unique_timestamps = df['timestamp'].unique()[4:]
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for timestamp in unique_timestamps:
        #print(f"timestamp {unique_timestamp}")
        timestamp_subset = df[df['timestamp']==timestamp]
        ts_index_train = [x for x in timestamp_subset.index if x in list(index_train)]
        ts_index_test = [x for x in timestamp_subset.index if x in list(index_test)]
        for i in range(len(ts_index_train)):
            X_train.append(df.loc[ts_index_train[i]-4:ts_index_train[i],data_columns])
            y_train.append(df.loc[ts_index_train[i], class_column])
        for i in range(len(ts_index_test)):
            X_test.append(df.loc[ts_index_test[i]-4:ts_index_test[i], data_columns])
            y_test.append(df.loc[ts_index_test[i], class_column])
        #print(X_train, y_train)
        for param in list(ParameterGrid(parameters)):
            #print(param)
            base_clf = LGBMClassifier(**param)
            clf = MultivariateClassifier(base_clf)
            clf.fit(X_train, y_train)
            y_train_hat = clf.predict(X_train)
            y_test_hat = clf.predict(X_test)
            output_dict = add_metrics_to_output_dict(output_dict, param, fold, timestamp, y_train, y_train_hat, y_test, y_test_hat)
            
output_df = pd.DataFrame.from_dict(output_dict)
output_df.to_csv("lgb_sg3f.csv")
print(output_df.describe())

features = CFS.cfs(df.loc[:,data_columns].to_numpy(), df.loc[:,class_column])
print(features)

csf_data_columns = [data_columns[feature] for feature in features]
print(csf_data_columns)

sgkf = StratifiedGroupKFold(n_splits=3)
sgkf.get_n_splits(data_index, y)
output_dict = create_output_dict()
for fold, (train_index, test_index) in enumerate(sgkf.split(data_index, y, unique_sources)):
    print(f"fold {fold}")
    index_train = [x for idx in train_index for x in data_index[idx]]
    index_test = [x for idx in test_index for x in data_index[idx]]
    unique_timestamps = df['timestamp'].unique()[4:]
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for timestamp in unique_timestamps:
        #print(f"timestamp {unique_timestamp}")
        timestamp_subset = df[df['timestamp']==timestamp]
        ts_index_train = [x for x in timestamp_subset.index if x in list(index_train)]
        ts_index_test = [x for x in timestamp_subset.index if x in list(index_test)]
        for i in range(len(ts_index_train)):
            X_train.append(df.loc[ts_index_train[i]-4:ts_index_train[i],csf_data_columns])
            y_train.append(df.loc[ts_index_train[i], class_column])
        for i in range(len(ts_index_test)):
            X_test.append(df.loc[ts_index_test[i]-4:ts_index_test[i], csf_data_columns])
            y_test.append(df.loc[ts_index_test[i], class_column])
        for param in list(ParameterGrid(parameters)):
            base_clf = LGBMClassifier(**param)
            clf = MultivariateClassifier(base_clf)
            clf.fit(X_train, y_train)
            y_train_hat = clf.predict(X_train)
            y_test_hat = clf.predict(X_test)
            output_dict = add_metrics_to_output_dict(output_dict, param, fold, timestamp, y_train, y_train_hat, y_test, y_test_hat)

output_df = pd.DataFrame.from_dict(output_dict)
output_df.to_csv("lgb_sg3f_csf.csv")
print(output_df.describe())
import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

task = "dp"
teams_labels = ["Player01", "Player02"]
match_column = "Source"

if task == "hp":
    target_column = "class"
    ignore_columns = ["Death", "Source", "source_id", "target_id"]
    input_path = "dataset/SmokeSquadron/ss_hitprediction2/base_dataset_hp2_5s.csv"
    output_file = f"hitprediction2_dataset_5s.csv"
    apply_sgkf = True
    apply_gss = False
elif task == "dp":
    target_column = ["WillDiePlayer01", "WillDiePlayer02"]
    ignore_columns = ["Source", "node_idxPlayer01", "node_idxPlayer02", "WillDiePlayer01", "WillDiePlayer02"]
    input_path = "dataset/SmokeSquadron/ss_deathprediction/death_prediction_new_base.csv"
    output_file = f"deathprediction_dataset_5s.csv"
    apply_sgkf = False
    apply_gss = True
output_path = os.path.join(os.path.dirname(input_path), output_file)    

df = pd.read_csv(input_path)
df_process = pd.read_csv(input_path)

ignored_col = lambda x: any([(igcol in x) for igcol in ignore_columns])

for column in [col for col in df.columns if "Player01" in col and not ignored_col(col)]:
    df_process[column.replace("Player01", "Diff")] = 0
    df_process[column+"Gradient"] = 0
    df_process[column.replace("Player01", "Player02")+"Gradient"] = 0

# Get the unique sources in the DataFrame
unique_sources = df[match_column].unique()

# Iterate over each source
for source in unique_sources:
    # Get the subset of rows with the current source
    source_subset = df[df[match_column] == source]
    
    # Iterate over the columns
    for column in source_subset.columns:
        # Check if the column has "Player01"
        if "Player01" in column and not ignored_col(column):
            # Get the corresponding Player01 and Player02 column names
            player01_col = column
            player02_col = column.replace("Player01", "Player02")
            
            # Create a new column name for the difference
            diff_col = column.replace("Player01", "Diff")
            
            # Calculate the difference between Player01 and Player02 columns
            source_subset[diff_col] = source_subset[player01_col] - source_subset[player02_col]
            
            # Create a new column name for the previous difference
            player01_grad_col = player01_col+"Gradient"
            player02_grad_col = player02_col+"Gradient"
            
            # Calculate the difference between the current row value and the previous rows
            source_subset[player01_grad_col] = source_subset[player01_col].diff()
            source_subset[player02_grad_col] = source_subset[player02_col].diff()
    df_process.loc[source_subset.index, source_subset.columns] = source_subset

df_process = df_process.drop(columns=[col for col in df_process.columns if "Unnamed" in col])
df_process.sort_index(axis=1, inplace=True)
df_process.fillna(0, inplace=True)

if apply_sgkf:
    datapoints_idx = [x for x in range(len(df_process)) if x%5==0]
    groups = df_process.loc[datapoints_idx, match_column]
    classes = df_process.loc[datapoints_idx, target_column]
    
    df_process.columns = df_process.columns.str.lower()

    sgkf = StratifiedGroupKFold(n_splits=5, random_state=43, shuffle=True)
    i_train_index, i_test_index = next(sgkf.split(datapoints_idx, classes, groups))
    train_index = [datapoints_idx[x] for x in i_train_index]
    test_index = [datapoints_idx[x] for x in i_test_index]
    
    test_output_path = os.path.join(os.path.dirname(input_path), output_file.replace('.','_test.'))
    df_process.loc[test_index].to_csv(test_output_path)
    
    train_output_path = os.path.join(os.path.dirname(input_path), output_file.replace('.','_train.'))
    df_process.loc[train_index].to_csv(train_output_path)
elif apply_gss:
    datapoints_idx = [x for x in range(len(df_process)) if x>4]
    groups = df_process.loc[datapoints_idx, match_column]
    classes = df_process.loc[datapoints_idx, target_column]
    
    df_process.columns = df_process.columns.str.lower()
    gss = GroupShuffleSplit(n_splits=10, train_size=.7, random_state=43)
    i_train_index, i_test_index = next(gss.split(datapoints_idx, classes, groups))
    train_index = [datapoints_idx[x] for x in i_train_index]
    test_index = [datapoints_idx[x] for x in i_test_index]
    
    test_output_path = os.path.join(os.path.dirname(input_path), output_file.replace('.','_test.'))
    df_process.loc[test_index].to_csv(test_output_path)
    
    train_output_path = os.path.join(os.path.dirname(input_path), output_file.replace('.','_train.'))
    df_process.loc[train_index].to_csv(train_output_path)
else:
    df_process.columns = df_process.columns.str.lower()
    df_process.round({'timestamp':0})
    df_process.to_csv(output_path)

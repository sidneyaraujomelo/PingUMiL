import os
import pandas as pd

teams_labels = ["Player01", "Player02"]
target_column = "winner"
match_column = "Source"
ignore_columns = ["Death", "Source"]
input_path = "dataset/SmokeSquadron/ss_winprediction/base_dataset_5s.csv"
output_file = f"lpmp_dataset_5s.csv"
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
df_process.columns = df_process.columns.str.lower()
df_process.sort_index(axis=1, inplace=True)
df_process.fillna(0, inplace=True)
df_process.round({'timestamp':0})
df_process.to_csv(output_path)

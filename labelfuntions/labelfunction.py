import pandas as pd

def category2mlb(classes, y):
    output = [0]*len(classes)
    for y_l in y:
        output[classes.index(y_l)] = 1
    return output

def create_label_function(labels, label_df_path, identifier_col, class_col):
    label_df = pd.read_csv(label_df_path)
    category_function = lambda x: category2mlb(labels, x)
    identifier2label = {}
    for _, row in label_df.iterrows():
        identifier = row[identifier_col]
        row_label = row[class_col]
        identifier2label[identifier] = category_function([row_label])
    label_func = lambda x: identifier2label[x]
    return label_func

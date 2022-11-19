import pandas as pd
"""
LIST OF THE ATTRIBUTES YOU WANT TO EXTRACT FROM THE ATTRIBUTE
NODE OF EACH VERTEX ON A PROVENANCE GRAPH XML. ONE OF THESE ATTRIBUTES MIGHT BE
OUR LABEL!
"""

""" GAME PROVENANCE PROFILE """
attrib_name_list = ['HP','Power','ObjectName','ObjectTag','ObjectPosition_X','ObjectPosition_Y',
'ObjectPosition_Z','Increase Wealth','Increase Level']
attrib_type_list = ['numeric','numeric','categoric','categoric','numeric','numeric','numeric','numeric','numeric']
attrib_default_value_list = [0,0,'inexistent','inexistent',0,0,0,0,0]

tag_name_list = ['label','type']
tag_type_list = ['categoric','categoric']
tag_default_value_list = ['inexistent','inexistent']

label_attrib_name = 'function'
label_attrib_type = 'categoric'
""" Label Conditions is a dictionary that determines conditions for a node to
become a label node. For example, if we want test and validations sets to contain
only nodes with given attributes or tags values."""
label_conditions = {'type':'Agent', 'label':'Player'}
""" Label Data. This part of the code is responsible for reading a csv that maps players to their classes
and build a dictionary with such data"""
label_csv_path = "/raid/home/smelo/PingUMiL-pytorch/dataset/GPP/final_results.csv"
file2label = {}
label_df = pd.read_csv(label_csv_path)
all_bartle_classes = ["Achiever", "Killer", "Socializer", "Explorer"]
all_dedication_classes = ["Casual", "Hardcore"]
def category2mlb(classes, y):
    print(y)
    output = [0]*len(classes)
    for y_l in y:
        output[classes.index(y_l)] = 1
    return output
bartle2mlb = lambda x: category2mlb(all_bartle_classes, x)
dedica2mlb = lambda x: category2mlb(all_dedication_classes, x)
for _, row in label_df.iterrows():
    match_files = row[[x for x in row.index if "Partida_" in x]]
    for match_file in match_files:
        if match_file:
            bartle_y = row["Bartle Classification"].split("/")
            dedica_y = row["Gamer Dedication"].split("/")
            file2label[match_file] = bartle2mlb(bartle_y) + dedica2mlb(dedica_y)
""" Label Function. This will be called if label_attrib_name is function"""
label_func = lambda x: file2label[x]

categoric_att_dict = {}
sup_dict = {}
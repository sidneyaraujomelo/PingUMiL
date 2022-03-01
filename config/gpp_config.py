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

label_attrib_name = 'type'
""" Label Conditions is a dictionary that determines conditions for a node to
become a label node. For example, if we want test and validations sets to contain
only nodes with given attributes or tags values."""
label_conditions = {}

categoric_att_dict = {}
sup_dict = {}
"""
LIST OF THE ATTRIBUTES YOU WANT TO EXTRACT FROM THE ATTRIBUTE
NODE OF EACH VERTEX ON A PROVENANCE GRAPH XML. ONE OF THESE ATTRIBUTES MIGHT BE
OUR LABEL!
"""
""" MORPHWING
attrib_name_list = ['ObjectPosition_X', 'ObjectPosition_Y', 'Speed', 'HP', 'LocalTime' ]
attrib_type_list = ['numeric', 'numeric', 'numeric', 'numeric', 'numeric']
attrib_default_value_list = [0, 0, 0, 999, 0]
"""

""" CAR TUTORIAL """
attrib_name_list = ['Throttle','Speed', 'CurrentGear','CurrentEnginePower','TurnRate','CarMass','VelocityVector_X',
'VelocityVector_Y','VelocityVector_Z','AngularVelocity_X','AngularVelocity_Y','AngularVelocity_Z','DragVector_X', 'DragVector_Y',
'DragVector_Z','ObjectPosition_X', 'ObjectPosition_Y', 'ObjectPosition_Z']
attrib_type_list = ['numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric', 
'numeric', 'numeric','numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric']
attrib_default_value_list = [1, 0, 0, 0, 0, 1500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

tag_name_list = ['label','type']
tag_type_list = ['categoric','categoric']
tag_default_value_list = ['inexistent','inexistent']

label_attrib_name = 'type'
""" Label Conditions is a dictionary that determines conditions for a node to
become a label node. For example, if we want test and validations sets to contain
only nodes with given attributes or tags values."""
label_conditions = {'type': 'Activity', 'ObjectName': 'Player'}

categoric_att_dict = {}
sup_dict = {}
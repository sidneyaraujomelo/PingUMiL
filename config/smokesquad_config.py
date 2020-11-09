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

""" SMOKE SQUADRON """
attrib_name_list = ['RollEffect', 'PitchEffect', 'YawlEffect', 'Throttle', 'AirBrakes',\
'IsAccelerating', 'ForwardSpeed', 'EnginePower', 'MaxEnginePower', 'RollAngle','PitchAngle',\
'RollInput','PitchInput','YawInput','ThrottleInput','Health','Life','SmokeWeapon1CooldownCounter',\
'SmokeWeapon2CooldownCounter','SmokeWeapon3CooldownCounter','RocketCooldownCounter',\
'SmokeCapacityCounter','RocketCounter','MachinGunOverheatCount','ObjectPosition_X', 'ObjectPosition_Y',\
'ObjectPosition_Z','ParentTag','Speed','DirectionX','DirectionY','DirectionZ','ExplosionTime',\
'ObjectTag']

attrib_type_list = ['numeric', 'numeric', 'numeric', 'numeric', 'categoric', 'categoric',\
'numeric', 'numeric', 'numeric', 'numeric', 'numeric', 'numeric','numeric','numeric', 'numeric',\
'numeric','numeric', 'numeric','numeric','numeric','numeric','numeric','numeric','numeric',\
'numeric','numeric','numeric','categoric','numeric','numeric','numeric','numeric','numeric',
'categoric']

attrib_default_value_list = [0,0,0,0,"False","False",0,0,0,0,0,0,0,0,0,9999,9999,0,0,0,0,0,0,0,0,0,\
0,"Inexistent",0,0,0,0,0,"-1"]

tag_name_list = ['label','type', 'date']
tag_type_list = ['categoric','categoric', 'numeric']
tag_default_value_list = ['inexistent','inexistent', 0]

label_attrib_name = 'ObjectTag'
""" Label Conditions is a dictionary that determines conditions for a node to
become a label node. For example, if we want test and validations sets to contain
only nodes with given attributes or tags values."""
label_conditions = {'type': 'Activity', 'ObjectName': 'Player'}

categoric_att_dict = {}
sup_dict = {}
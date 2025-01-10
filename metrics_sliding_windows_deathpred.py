import os
import json
import math
import networkx as nx
import pandas as pd
from config.util import loadProvenanceXML
from prov_hnx_parser import ProvHnxParser

max_time_interval = 5
    
def get_player_attributed_nodes(player_label, nodes_attributes, label_values, object_tag_values):
    player_labeled_nodes = [node for node, value in label_values.items() if value == player_label]
    player_tagged_nodes = [node for node, value in object_tag_values.items() if value == player_label]
    player_nodes = player_labeled_nodes+player_tagged_nodes
    player_attributed_nodes = [(node, attr) for node, attr in nodes_attributes if node in player_nodes]
    player_attributed_nodes = sorted(player_attributed_nodes, key=lambda x: x[1]["date"])
    return player_attributed_nodes

def get_player_metrics(player_attributed_nodes, player_label, start_time):
    
    # Initialize variables to track previous positions and time
    previous_positions = None
    previous_angles = None
    previous_time = start_time
    start_timestamp = start_time
    time_interval = 0
    accumulated_distance = 0
    accumulated_angular_movement = 0
    count_label_attributes = [
        "FiringMachineGun",
        "AirBrakes",
        "IsAccelerating",
        "GotItem",
        "FiredMissile",
        "Death"
    ]
    count_label_dict = { k:0 for k in count_label_attributes}
    
    player_attributes_dict = {
        "node_idx" : [],
        "timestamp" : [],
        "distance": [],
        "angular_movement": []
    }
    for k in count_label_attributes:
        player_attributes_dict[k] = []

    # Iterate over the sorted nodes and calculate distance traveled
    for i, node_attributes_tup in enumerate(player_attributed_nodes):
        attributes = node_attributes_tup[-1]
        time = attributes["date"]
        x = attributes["ObjectPosition_X"]
        y = attributes["ObjectPosition_Y"]
        z = attributes["ObjectPosition_Z"]
        roll = attributes["RollAngle"]
        pitch = attributes["PitchAngle"]
        label = attributes["label"]
        node_idx = int(attributes["ID"].split("_")[-1])
        
        # Skip nodes without the necessary attributes
        if None in [time, x, y, z, roll, pitch, label]:
            continue
        
        # Check if it's the first time encountering the node
        if not previous_positions or not previous_angles or label == "Respawning":
            previous_positions = (x, y, z)
            previous_angles = (roll, pitch)
            continue
        
        # Count if a given label has a target attribute
        if label in count_label_attributes:
            count_label_dict[label] = count_label_dict[label]+1
        
        # Calculate the distance traveled since the previous time step
        time_difference = time - previous_time
        distance = math.sqrt((x - previous_positions[0])**2 +
                             (y - previous_positions[1])**2 + (z - previous_positions[2])**2)
        
        # Calculate the angular difference since the previous time step
        roll_diff = roll - previous_angles[0]
        pitch_diff = pitch - previous_angles[1]
        
        # Wrap the angular differences to [-pi, pi] range
        roll_diff = (roll_diff + math.pi) % (2 * math.pi) - math.pi
        pitch_diff = (pitch_diff + math.pi) % (2 * math.pi) - math.pi
        
        # Accumulate the absolute angular movement
        angular_movement = abs(roll_diff) + abs(pitch_diff)
        
        # Print the result for every x seconds
        if time_interval + time_difference >= max_time_interval or i == len(player_attributed_nodes)-1:
            if time_interval + time_difference >= max_time_interval:
                # Separate distance according to the amount of time to reach max_time_interval
                ratio_time = (max_time_interval - time_interval)/time_difference
                # Update accumulated distance, accumulated angular movement and time
                accumulated_distance = accumulated_distance + ratio_time * distance
                accumulated_angular_movement = accumulated_angular_movement + ratio_time * angular_movement
                time_interval = time_interval + ratio_time*time_difference
            else:
                accumulated_distance = accumulated_distance + distance
                accumulated_angular_movement = accumulated_angular_movement + angular_movement
                time_interval = time_interval + time_difference
            # Calculate the speed in units per second
            #print(f"{player_label} traveled {accumulated_distance} units in {time_interval} seconds. Speed: {speed} units/second")
            #print(f"{player_label} angular movement: {accumulated_angular_movement}")
            #for event_label, count_event in count_label_dict.items():
                #print(f"{player_label} triggered the {event_label} event {count_event} times.")
            
            #Populate dictionary
            player_attributes_dict["timestamp"].append(start_timestamp)
            player_attributes_dict["distance"].append(accumulated_distance)
            player_attributes_dict["angular_movement"].append(accumulated_angular_movement)
            player_attributes_dict["node_idx"].append(node_idx)
            start_timestamp = start_timestamp + max_time_interval
            for event_label, count_event in count_label_dict.items():
                player_attributes_dict[event_label].append(count_event)
            
            time_interval = (1-ratio_time) * time_difference
            accumulated_distance = accumulated_distance + (1-ratio_time) * distance
            accumulated_angular_movement = accumulated_angular_movement + (1-ratio_time) * angular_movement
        else:
            time_interval = time_interval + time_difference
            accumulated_distance = accumulated_distance + distance
            accumulated_angular_movement = accumulated_angular_movement + angular_movement
        previous_positions = (x, y, z)
        previous_time = time
    return player_attributes_dict

def process_prov_xml(path):
    input_file = path
    parse_config = json.load(open("config/parse_config.json","r"))
    data_config = json.load(open("config/smokesquad_config.json","r"))
    assert len(data_config["attrib_name_list"]) == len(data_config["attrib_type_list"])
    assert len(data_config["attrib_name_list"]) == len(data_config["attrib_default_value_list"])
    print(parse_config, data_config)
    parser = ProvHnxParser(parse_config, data_config, True, input_file)
    parser.parse()
    #parser.save()
    G = parser.dataset[0].g
    #First, we get player nodes
    label_values = nx.get_node_attributes(G, "label")
    object_tag_values = nx.get_node_attributes(G, "ObjectTag")

    nodes_attributes = G.nodes(data=True)
    player_labels = ["Player01", "Player02"]
    player_atbs_nodes_dict = {x:get_player_attributed_nodes(x, nodes_attributes, label_values, object_tag_values) for x in player_labels}
    player_atbs_dict = {x:None for x in player_labels}

    for player_label, player_attributed_nodes in player_atbs_nodes_dict.items():
        print(player_label)
        player_atbs_dict[player_label] = get_player_metrics(player_attributed_nodes, player_label, nodes_attributes[0]["date"])
        print(player_atbs_dict[player_label])

    player_01_df = pd.DataFrame.from_dict(player_atbs_dict[player_labels[0]])
    player_01_df["will_die"] = player_01_df["Death"].shift(-1)
    player_01_df["WillDiePlayer01"] = ((player_01_df["Death"] != player_01_df["will_die"])).astype(int)
    player_01_df.drop(columns=["will_die", "Death"],inplace=True)
    
    player_02_df = pd.DataFrame.from_dict(player_atbs_dict[player_labels[1]])
    player_02_df["will_die"] = player_02_df["Death"].shift(-1)
    player_02_df["WillDiePlayer02"] = ((player_02_df["Death"] != player_02_df["will_die"])).astype(int)
    player_02_df.drop(columns=["will_die", "Death"],inplace=True)
    
    players_df = player_01_df.merge(player_02_df, on="timestamp", suffixes=player_labels, how="outer")
    start_columns = ["node_idxPlayer01","node_idxPlayer02"]
    last_columns = ["WillDiePlayer01","WillDiePlayer02"]
    new_column_order = start_columns + [x for x in players_df.columns if x not in start_columns and x not in last_columns] + last_columns
    players_df = players_df[new_column_order]
    
    #last_row_values = players_df.loc[players_df.index[-1], ['DeathPlayer01', 'DeathPlayer02']]
    #assert (last_row_values['DeathPlayer02'] == 3 or last_row_values['DeathPlayer01'] == 3) and not (last_row_values['DeathPlayer01'] == 3 and last_row_values['DeathPlayer02'] == 3)
    #winner_tag = "player01" if last_row_values['DeathPlayer02'] == 3 else "player02"
    #players_df["winner"] = winner_tag
    players_df["timestamp"] = round(players_df["timestamp"] - players_df.loc[players_df.index[0],'timestamp'] + max_time_interval,0)
    print(players_df)
    return players_df

def main():
    basepaths = ["../smokesquadrondataset/all_graphs"]
    match_dict = {}
    for basepath in basepaths:
        for root, dirs, files in os.walk(basepath):
            for input_file in files:
                death_df = process_prov_xml(os.path.join(root, input_file))
                match_dict[os.path.basename(input_file)] = death_df
    matches_df = pd.concat([df.assign(Source=name) for name, df in match_dict.items()], ignore_index=True)
    matches_df.to_csv("death_prediction_new_base.csv")
    
if __name__ == "__main__":
    main()
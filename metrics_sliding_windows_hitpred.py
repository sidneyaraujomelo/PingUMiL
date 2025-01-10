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

def get_player_metrics(player_attributed_nodes, player_label, start_time, end_time=None):
    
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
        vertex_id = attributes["ID"]
        
        # Skip nodes without the necessary attributes
        if None in [time, x, y, z, roll, pitch, label]:
            continue
        
        
        # Check if it's the first time encountering the node, or the node is before start_time
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
        
        if  time <= start_time:
            accumulated_distance = accumulated_distance + distance
            accumulated_angular_movement = accumulated_angular_movement + angular_movement
        else:
            # Print the result for every x seconds
            if time_interval + time_difference >= max_time_interval or i == len(player_attributed_nodes)-1 or (end_time != None and time > end_time):
                if time_interval + time_difference >= max_time_interval or (end_time != None and time > end_time):
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
        
        if (end_time != None and start_timestamp >= end_time):
            break    
    return player_attributes_dict

def process_prov_xml(G):
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
    player_02_df = pd.DataFrame.from_dict(player_atbs_dict[player_labels[1]])
    players_df = player_01_df.merge(player_02_df, on="timestamp", suffixes=player_labels, how="outer")
    last_row_values = players_df.loc[players_df.index[-1], ['DeathPlayer01', 'DeathPlayer02']]
    assert (last_row_values['DeathPlayer02'] == 3 or last_row_values['DeathPlayer01'] == 3) and not (last_row_values['DeathPlayer01'] == 3 and last_row_values['DeathPlayer02'] == 3)
    winner_tag = "player01" if last_row_values['DeathPlayer02'] == 3 else "player02"
    players_df["winner"] = winner_tag
    players_df["timestamp"] = round(players_df["timestamp"] - players_df.loc[players_df.index[0],'timestamp'] + max_time_interval,0)
    return players_df

def get_metrics_sliding_window(G, target_vertex, source_vertex, y):
    #First, we get player nodes
    label_values = nx.get_node_attributes(G, "label")
    object_tag_values = nx.get_node_attributes(G, "ObjectTag")
    nodes_attributes = G.nodes(data=True)
    first_node = int(G.nodes(data=True)[0]['ID'].split('_')[-1])
    print(source_vertex, first_node)
    source_vertex_attributes = nodes_attributes[source_vertex-first_node]
    assert f"vertex_{source_vertex}" == source_vertex_attributes["ID"]
    end_timestamp = source_vertex_attributes["date"]
    start_timestamp = max(end_timestamp - 25, 0)
    #print(start_timestamp, end_timestamp)
    
    player_labels = ["Player01", "Player02"]
    player_atbs_nodes_dict = {x:get_player_attributed_nodes(x, nodes_attributes, label_values, object_tag_values) for x in player_labels}
    player_atbs_dict = {x:None for x in player_labels}

    for player_label, player_attributed_nodes in player_atbs_nodes_dict.items():
        #print(player_label)
        player_atbs_dict[player_label] = get_player_metrics(player_attributed_nodes, player_label, start_timestamp, end_timestamp)
        #print(player_atbs_dict[player_label])
        
    player_01_df = pd.DataFrame.from_dict(player_atbs_dict[player_labels[0]])
    player_02_df = pd.DataFrame.from_dict(player_atbs_dict[player_labels[1]])
    hit_df = player_01_df.merge(player_02_df, on="timestamp", suffixes=player_labels, how="outer")
    hit_df["class"] = y
    # provenance edges are reversed w.r.t. to timestamp
    hit_df["source_id"] = target_vertex
    hit_df["target_id"] = source_vertex
    return hit_df

def open_game_xml(path):
    input_file = path
    parse_config = json.load(open("config/parse_config.json","r"))
    data_config = json.load(open("config/smokesquad_config.json","r"))
    assert len(data_config["attrib_name_list"]) == len(data_config["attrib_type_list"])
    assert len(data_config["attrib_name_list"]) == len(data_config["attrib_default_value_list"])
    #print(parse_config, data_config)
    parser = ProvHnxParser(parse_config, data_config, True, input_file)
    parser.parse()
    #parser.save()
    G = parser.dataset[0].g
    return G

def main():
    basepath = "../smokesquadrondataset/ss_hitpred2/no_target_edge_graph"
    edge_dataset = json.load(open("../smokesquadrondataset/ss_hitpred2/edge_dataset/edgedataset.json", "r"))
    edge_dataset_df = pd.DataFrame.from_dict(edge_dataset)
    
    xml_files = edge_dataset_df["origin"].unique()
    hits_dict = {}
    for xml_file in xml_files:
        xml_dfs = []
        game_edge_df = edge_dataset_df[edge_dataset_df["origin"]==xml_file]
        #print(game_edge_df)
        G = open_game_xml(os.path.join(basepath, f"r-{xml_file}"))
        for index, row in game_edge_df.iterrows():
            hit_df = get_metrics_sliding_window(G, row["source"], row["target"], row["class"])
            xml_dfs.append(hit_df)
        hits_dict[f"{xml_file}"] = pd.concat(xml_dfs)
            
    hits_df = pd.concat([df.assign(Source=name) for name, df in hits_dict.items()], ignore_index=True)
    hits_df.to_csv("test_hits2.csv")
    
if __name__ == "__main__":
    main()
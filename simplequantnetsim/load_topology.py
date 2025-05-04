import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

# Define input and output paths
input_folder = Path(__file__).parent.parent.joinpath('real_topologies')
output_folder = input_folder.parent / 'graphs_json'
class_0_output_folder = output_folder / 'class_0'
class_1_output_folder = output_folder / 'class_1'
class_2_output_folder = output_folder / 'class_2'

# Create output directories if they do not exist
output_folder.mkdir(exist_ok=True)
class_0_output_folder.mkdir(exist_ok=True)
class_1_output_folder.mkdir(exist_ok=True)
class_2_output_folder.mkdir(exist_ok=True)

# Predefined class labels for each topology
graph_class = [1, 2, 2, 2, 0, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2, 
               2, 2, 1, 2, 0, 1, 1, 2, 2, 2, 1, 2, 1, 0, 1, 0, 0, 1, 1, 0, 2, 2, 2, 0, 0, 2, 2, 
               1, 2, 2, 0, 2, 0, 1, 0, 1, 2, 0, 2, 1, 0, 0, 0, 2, 2, 2, 2, 2, 0, 2, 0, 0, 2, 0,
               2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1] 

def store_by_class(G, graph_class, filename):
    """
    Store a NetworkX graph object as a JSON file in a folder according to its class label.

    The graph is stored using the node-link format and written to a JSON file.
    The file is placed into a subfolder named by its class: class_0, class_1, or class_2.

    Args:
        G (networkx.Graph): The graph object to store.
        graph_class (int): The class label (0, 1, or 2) used to determine the output folder.
        filename (str): The base filename (without extension) for the output JSON file.

    Returns:
        None
    """
    # Choose output path based on topology class
    if graph_class == 0:
        json_path = class_0_output_folder / f"{filename}.json"
    elif graph_class == 1:
        json_path = class_1_output_folder / f"{filename}.json"
    else:
        json_path = class_2_output_folder / f"{filename}.json"
    
    # Convert graph to node-link format and write to JSON file
    json_data = nx.node_link_data(G)
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

# Process each Excel file in the input folder
for index, file_path in enumerate(input_folder.glob('*.xlsx')):
    # Construct sheet names from filename
    nodes_sheet_name = f'Nodes_{file_path.stem.split("_")[-1]}'
    edge_sheet_name = f'Edges_{file_path.stem.split("_")[-1]}'
    
    # Load node and edge data from Excel
    nodes_df = pd.read_excel(file_path, sheet_name=nodes_sheet_name)
    edges_df = pd.read_excel(file_path, sheet_name=edge_sheet_name)

    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes and node attributes
    center_x = nodes_df['Latitude'].mean()
    center_y = nodes_df['Longitude'].mean()
    nodes_df['Distance_to_Center'] = np.sqrt((nodes_df['Latitude'] - center_x) ** 2 + (nodes_df['Longitude'] - center_y) ** 2)
    nodes_df = nodes_df.sort_values(by='Distance_to_Center')

    for _, row in nodes_df.iterrows():
        node_id = int(row['Node_ID'])
        latitude = float(row['Latitude'])
        longitude = float(row['Longitude'])
        center_distance = float(row['Distance_to_Center'])
        G.add_node(
            node_id,
            latitude=latitude,
            longitude=longitude,
            location=row['Location Name'],
            country=row['Country'],
            center_distance=center_distance
        )

    # Add edges and edge attributes
    for _, row in edges_df.iterrows():
        source = int(row['Source'])
        destination = int(row['Destination'])
        length = float(row['Computed Length (km)'])
        G.add_edge(
            source,
            destination,
            length=length
        )

    # Store the graph in the appropriate class folder
    store_by_class(G, graph_class[index], file_path.stem)

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

input_folder = Path(__file__).parent.parent.joinpath('real_topologies')
output_folder = input_folder.parent / 'graphs_json'
class_0_output_folder = output_folder / 'class_0'
class_1_output_folder = output_folder / 'class_1'
class_2_output_folder = output_folder / 'class_2'

output_folder.mkdir(exist_ok=True)
class_0_output_folder.mkdir(exist_ok=True)
class_1_output_folder.mkdir(exist_ok=True)
class_2_output_folder.mkdir(exist_ok=True)

graph_class = [1, 2, 2, 2, 0, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2, 
               2, 2, 1, 2, 0, 1, 1, 2, 2, 2, 1, 2, 1, 0, 1, 0, 0, 1, 1, 0, 2, 2, 2, 0, 0, 2, 2, 
               1, 2, 2, 0, 2, 0, 1, 0, 1, 2, 0, 2, 1, 0, 0, 0, 2, 2, 2, 2, 2, 0, 2, 0, 0, 2, 0,
               2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1] 


def store_by_class(G, graph_class, filename):
    # choose the path according to the class of topology
    if graph_class == 0:
        json_path = class_0_output_folder / f"{filename}.json"
    elif graph_class == 1:
        json_path = class_1_output_folder / f"{filename}.json"
    else:
        json_path = class_2_output_folder / f"{filename}.json"
    
    # Store graph data into corresponding file
    json_data = nx.node_link_data(G)
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

for index, file_path in enumerate(input_folder.glob('*.xlsx')):
    # Load the Excel file
    nodes_sheet_name = f'Nodes_{file_path.stem.split("_")[-1]}'
    edge_sheet_name = f'Edges_{file_path.stem.split("_")[-1]}'
    
    # read node and edge data
    nodes_df = pd.read_excel(file_path, sheet_name=nodes_sheet_name)
    edges_df = pd.read_excel(file_path, sheet_name=edge_sheet_name)

    # create graph object
    G = nx.Graph()

    # add node and node attributes to the graph object
    center_x = nodes_df['Latitude'].mean()
    center_y = nodes_df['Longitude'].mean()
    nodes_df['Distance_to_Center'] = np.sqrt((nodes_df['Latitude'] - center_x) ** 2 + (nodes_df['Longitude'] - center_y) ** 2)
    nodes_df = nodes_df.sort_values(by='Distance_to_Center')

    for _, row in nodes_df.iterrows():
        node_id = int(row['Node_ID'])  # 转换为标准的 Python int 类型
        latitude = float(row['Latitude'])  # 转换为标准的 Python float 类型
        longitude = float(row['Longitude'])  # 转换为标准的 Python float 类型
        center_distance = float(row['Distance_to_Center'])  # 转换为标准 float
        G.add_node(
            node_id,
            latitude=latitude,
            longitude=longitude,
            location=row['Location Name'],
            country=row['Country'],
            center_distance=center_distance
        )

    for _, row in edges_df.iterrows():
        source = int(row['Source'])  # 转换为标准 int
        destination = int(row['Destination'])  # 转换为标准 int
        length = float(row['Computed Length (km)'])  # 转换为标准 float
        G.add_edge(
            source,
            destination,
            length=length
        )

    
    # store the graph into correct folder
    store_by_class(G, graph_class[index], file_path.stem)

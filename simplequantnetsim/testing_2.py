import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

def convert_type(df):
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype(int)  # 转换为 Python 的 int 类型
        else:
            df[col] = df[col].astype(float)  # 转换为 Python 的 float 类型
    return df

input_folder = Path(__file__).parent.parent.joinpath('real_topologies')
output_folder = input_folder.parent / 'graphs_json'
output_folder.mkdir(exist_ok=True)

for file_path in input_folder.glob('*.xlsx'):
    # Load the Excel file
    nodes_sheet_name = f'Nodes_{file_path.stem.split("_")[-1]}'
    edge_sheet_name = f'Edges_{file_path.stem.split("_")[-1]}'
    
    # 读取节点和边的数据
    nodes_df = pd.read_excel(file_path, sheet_name=nodes_sheet_name)
    edges_df = pd.read_excel(file_path, sheet_name=edge_sheet_name)

    # 转换数据类型
    nodes_df = convert_type(nodes_df)
    edges_df = convert_type(edges_df)

    # 创建 Graph 对象
    G = nx.Graph()

    # Step 1: 添加节点及其属性
    distances = {}
    for _, row in nodes_df.iterrows():
        node_id = row['Node_ID']
        latitude = row['Latitude']
        longitude = row['Longitude']
        
        # 计算节点与原点 (0, 0) 的欧几里得距离
        distance = np.sqrt(latitude**2 + longitude**2)
        distances[node_id] = distance

        # 添加节点到图
        G.add_node(
            node_id,
            latitude=latitude,
            longitude=longitude,
            location=row['Location Name'],
            country=row['Country']
        )

    # Step 2: 找到距离原点 (0, 0) 最短的节点作为中心节点
    center_node = min(distances, key=distances.get)
    print(1)
    # Step 3: 添加边及其属性
    for _, row in edges_df.iterrows():
        G.add_edge(
            row['Source'],
            row['Destination'],
            length=row['Computed Length (km)']
        )

    # Step 4: 重新编号节点ID，确保中心节点的ID为1，其他节点依次增加
    new_graph = nx.Graph()
    node_mapping = {}

    # 先添加中心节点，设为ID为1
    new_node_id = 1
    node_mapping[center_node] = new_node_id
    new_graph.add_node(new_node_id, **G.nodes[center_node])

    # 处理其他节点，ID依次递增
    new_node_id += 1
    for old_node_id in G.nodes:
        if old_node_id == center_node:
            continue
        node_mapping[old_node_id] = new_node_id
        new_graph.add_node(new_node_id, **G.nodes[old_node_id])
        new_node_id += 1

    # 添加边，使用新的节点ID
    for source, target, edge_data in G.edges(data=True):
        new_graph.add_edge(node_mapping[source], node_mapping[target], **edge_data)

    # Step 5: 将图保存为 JSON 格式
    json_data = nx.node_link_data(new_graph)

    # 保存 JSON 文件
    json_path = output_folder / f"{file_path.stem}_reordered.json"
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

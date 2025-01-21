import json
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


pos = {}
user = []

# Step 1: 读取 JSON 文件
filepath = Path(__file__).parent.parent.joinpath('graphs_json','class_0', 'TOP_20_CONUS75.json')
with open(filepath, "r") as f:
    data = json.load(f)

# Step 2: 初始化图
G = nx.Graph()

# Step 3: 添加节点

for node in data["nodes"]:
    node_id = node["id"]
    x, y = node["latitude"], node["longitude"]
    G.add_node(node_id, location=node["location"], country=node["country"])  # 添加节点到图
    pos[node_id] = (y, x)  # 保存节点位置，注意 (longitude, latitude)

# Step 4: 添加边
for edge in data["links"]:
    source = int(edge["source"])
    target = int(edge["target"])
    G.add_edge(source, target, length=edge["length"])  # 添加边到图

degree_dict = dict(G.degree())
degree_items = list(degree_dict.items())
first_node,first_degree = degree_items[0]
print(f"First node ID: {first_node}, Degree: {first_degree}")
print(type(first_degree))

user.append(data["nodes"][0]["id"])
for item in itertools.islice(reversed(list(data["nodes"])), first_degree):
    user.append(item['id'])
print(user)

# Step 5: 计算中心点
x_coords = [pos[node][0] for node in pos]
y_coords = [pos[node][1] for node in pos]
center_x = sum(x_coords) / len(x_coords)
center_y = sum(y_coords) / len(y_coords)
center = (center_x, center_y)
min_length = float('inf')  # 将最小距离初始化为正无穷
center_node = None

for node in data["nodes"]:
    x_distance = node["latitude"] - center_y  # 注意这里是 latitude 对应 y 坐标
    y_distance = node["longitude"] - center_x  # longitude 对应 x 坐标
    distance = np.sqrt(x_distance**2 + y_distance**2)
    
    if distance < min_length:
        source_node = node['id']
        min_length = distance
print(source_node)

# Step 6: 绘制图并标记中心
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
plt.scatter(center_x, center_y, color='red', marker='x', s=100, label='Center')  # 标记中心点

# 添加标题和图例
plt.title("Network Topology of TOP_1_ABILENE")
plt.legend()
plt.show()

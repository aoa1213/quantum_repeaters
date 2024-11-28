import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1: 读取 JSON 文件
filepath = Path(__file__).parent.parent.joinpath('graphs_json', 'TOP_1_ABILENE_reordered.json')
with open(filepath, "r") as f:
    data = json.load(f)

# Step 2: 初始化图
G = nx.Graph()

# Step 3: 添加节点
pos = {}
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

# Step 5: 绘制图
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
plt.title("Network Topology of TOP_1_ABILENE")
plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import csv
import json
import math
import random
import itertools
import numpy as np
import networkx as nx
import tqdm as tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from protocols import MPC_protocol, MPG_protocol, SP_protocol
from graph import network, set_p_edge

from networkx.algorithms.community import greedy_modularity_communities
from networkx.drawing.layout import *



# In[ ]:


mkr = ['x','+','d','o','1','2']+['x']*100
dashs = ['-.','--',':','-']+['-']*100
cols = ['gray','g','b','orange','r','k','purple']+['k']*100
linewidth = 2.2
mks = 5.5
fontsize = 14


# Find the ER for the MPC, MPG, and SP protocols

# In[ ]:


def load_data(filepath):
    pos = {}
    user = []

    # Step 1: 读取 JSON 文件
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

    user.append(data["nodes"][0]["id"])

    return G,user,pos


# In[ ]:


import networkx as nx
import random
import matplotlib.pyplot as plt
from itertools import product

def multi_iterative_score_partition_with_drawing(
    G, 
    fixed_node, 
    alpha=1.0, 
    beta=1.0,
    max_rounds=10,
    shuffle_nodes=True,
    pos=None,
    output_path=None
):
    """
    多轮迭代版“打分 + 搬家”分社区并绘制图：
      - 社区数 = degree(fixed_node) + 1
      - 初始: 0 号社区放 fixed_node，其余每个邻居一个社区
      - 分配时: score = alpha * distance + beta * community_size
      - 多轮: 如果搬家能让节点的 score 更低, 就搬家, 直到稳定或 max_rounds
      - 绘图可选。

    同时返回：
      - 最终的社区划分 communities
      - 对所有社区(除了 0 号) 进行“从每个社区选一个节点”的所有组合 all_key_nodes_combos，
        并在每个组合前面加上固定节点 fixed_node。
    """
    # ============ 1) 初始化社区容器 ============
    neighbors = list(G.neighbors(fixed_node))
    num_communities = len(neighbors)  # 除 0 号社区外的社区数
    communities = [set() for _ in range(num_communities + 1)]
    visited = set()

    # 第 0 号社区放 fixed_node
    communities[0].add(fixed_node)
    visited.add(fixed_node)

    # 其余邻居各一个社区
    for i, nb in enumerate(neighbors, start=1):
        communities[i].add(nb)
        visited.add(nb)

    # ============ 2) 初次分配剩余节点 ============
    for node in G.nodes():
        if node not in visited:
            best_score = float('inf')
            best_index = None
            # 尝试放进 1~num_communities 各个社区，看 score 哪个最小
            for i, nb in enumerate(neighbors, start=1):
                dist = nx.shortest_path_length(G, source=node, target=nb)
                size = len(communities[i])
                score = alpha * dist + beta * size
                if score < best_score:
                    best_score = score
                    best_index = i
            communities[best_index].add(node)
            visited.add(node)

    # ============ 3) 多轮迭代搬家 ============
    round_num = 0
    while round_num < max_rounds:
        round_num += 1
        moved_count = 0

        # 获取所有节点（除 fixed_node 外）
        all_nodes = [n for n in G.nodes() if n != fixed_node]

        if shuffle_nodes:
            random.shuffle(all_nodes)

        for node in all_nodes:
            # 找到当前所在社区
            current_idx = None
            for i, comm in enumerate(communities):
                if node in comm:
                    current_idx = i
                    break
            
            # 如果在 0 号社区，则说明节点是 fixed_node，不搬家
            if current_idx == 0:
                continue

            # 当前社区分值
            nb_current = neighbors[current_idx - 1]  # 对应邻居
            dist_current = nx.shortest_path_length(G, source=node, target=nb_current)
            size_current = len(communities[current_idx])
            current_score = alpha * dist_current + beta * size_current

            # 尝试搬去别的社区
            best_score = current_score
            best_index = current_idx

            for i, nb in enumerate(neighbors, start=1):
                if i == current_idx:
                    continue
                dist = nx.shortest_path_length(G, source=node, target=nb)
                size = len(communities[i])
                score = alpha * dist + beta * size

                if score < best_score:
                    best_score = score
                    best_index = i

            # 如果找到更好的社区 => 搬家
            if best_index != current_idx:
                communities[current_idx].remove(node)
                communities[best_index].add(node)
                moved_count += 1

        # 如果没人搬家 => 收敛 => 停止
        if moved_count == 0:
            break

    # ============ 4) 计算所有可能的 key_nodes 组合 ============
    # 注意：0 号社区只包含 fixed_node，它是永远固定的，不做枚举
    #       其余社区(1..num_communities)从中各自选 1 个节点，形成一个组合
    all_key_nodes_combos = []

    # 如果有社区是空的，那么不可能从该社区选出节点
    if all(len(communities[i]) > 0 for i in range(1, num_communities + 1)):
        # itertools.product(...) 会返回所有从各社区选1个节点的组合
        # communities[i] 本身是一个 set，不影响 product 的用法
        # 例如 [setA, setB, setC] => product(setA, setB, setC) 会返回 (a, b, c) 等所有组合
        all_products = product(*(communities[i] for i in range(1, num_communities + 1)))

        # 为每个组合都把 fixed_node(第0社区) 放在开头
        for combo in all_products:
            combo_list = [fixed_node] + list(combo)
            all_key_nodes_combos.append(combo_list)
    else:
        # 如果有社区为空，那么没有可行的枚举
        all_key_nodes_combos = []

    # ============ 5) 绘制社区图 ============
    if pos is None:
        pos = nx.spring_layout(G, seed=42)  # 默认布局

    colors = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "pink"]
    
    plt.figure(figsize=(8, 6))
    
    # 逐个社区画节点
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(
            G, pos, nodelist=community,
            node_color=colors[i % len(colors)], 
            label=f"Community {i}",
            alpha=0.8,
            node_size=100
        )

    # 特别标注固定节点（用红色圆形）
    nx.draw_networkx_nodes(
        G, pos, nodelist=[fixed_node],
        node_color="red", node_shape="o",
        node_size=100, alpha=0.9, label="Fixed Node"
    )

    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    plt.legend(
        fontsize=6,  
        borderaxespad=0.5, 
        labelspacing=0.2,   
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
    )
    plt.title("Graph with Colored Communities (no farthest-node highlight)")
    plt.tight_layout(pad=2.0)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {output_path}")
    else:
        plt.show()
    plt.close()

    # 返回：最终社区划分，以及所有 key_nodes 组合
    return communities, all_key_nodes_combos


# In[ ]:


def plot_er_vs_p(p_range, ER, funcs, cols, output_path, fontsize=12, figsize=(10, 6), dpi=600):
    """
    绘制 ER 与链接生成概率 p 的关系图，并保存为文件。

    参数:
        p_range (array-like): p 的取值范围。
        ER (list of lists): 每个函数对应的 ER 数据。
        funcs (list): 包含函数的列表，用于生成图例名称。
        cols (list): 每个函数对应的颜色列表。
        output_path (str): 保存图片的路径。
        fontsize (int, 可选): 坐标轴和标签字体大小，默认 12。
        figsize (tuple, 可选): 图形大小，默认 (10, 6)。
        dpi (int, 可选): 图像分辨率，默认 600。
    """
    nom_list = [str(f).split(' ')[1] for f in funcs]
    plt.figure(figsize =(10,6),dpi=600 )
    plt.grid(linewidth=0.5)
    for i in range(len(funcs)):
        y = plt.plot(p_range, ER[i],
                    color = cols[i],
                    marker = "x",
                    linestyle='None',
                    markersize = 3,
                    #  alpha = 0.5,
                    #  linewidth=linewidth,
                    label = nom_list[i])
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.tick_params(labelsize=fontsize)

    plt.xlabel('Link generation probability p',fontsize=fontsize)
    plt.ylabel('ER ($\mathregular{GHZ}_5/\ \\mathregular{T_{slot}}$)',fontsize=fontsize)

    ax = plt.gca()
    ax.set_xlim([0.2, 1])
    ax.set_ylim([0.0001, 1])

    # 保存图片
    plt.savefig(output_path, dpi=dpi)
    print(f"Plot saved to {output_path}")


# In[ ]:


funcs = [MPC_protocol, MPG_protocol, SP_protocol]
p_range = np.linspace(1, 0.2, 50)

timesteps = 10
reps = 20
alpha = 1.4
beta = 0.105
max_rounds = 10
shuffle_nodes = True
root_path = Path.cwd().parent.joinpath("graphs_json")

# 用于存储每个子文件夹中每个协议的 SR 结果
sr_results = []

# 遍历 graphs_json 文件夹中的每个子文件夹
for subfolder in root_path.iterdir():
    if subfolder.is_dir():  # 判断是否是子文件夹
        print(f"Processing subfolder: {subfolder}")
        failure_counts = {func.__name__: 0 for func in funcs}
        combination_counter = 0

        # 遍历当前子文件夹中的所有文件
        for file in subfolder.iterdir():
            if file.is_file():  # 判断是否是文件
                print(f"  Processing file: {file}")
                
                class_folder = Path(subfolder).name
                file_path = file.with_suffix(".png")
                file_name = Path(file_path).name
                er_folder_path = Path.cwd().parent.joinpath("new_result", class_folder)
                er_folder_path.mkdir(exist_ok=True)
                er_topology_folder_path = er_folder_path.joinpath(file_name)
                er_topology_folder_path.mkdir(exist_ok=True)
                communities_output_path = Path.cwd().parent.joinpath("communitie", class_folder, file_name)
                
                G, users, pos = load_data(file)
                nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
                G = network(G)
                communities, users_node_combination = multi_iterative_score_partition_with_drawing(
                    G, users[0], alpha, beta, max_rounds, shuffle_nodes, pos, communities_output_path
                )
                
                for combo_idx, combo in enumerate(users_node_combination):
                    # 先重置 ER，确保每个组合一张空表来累加
                    print(combo)
                    combination_counter += 1
                    ER = np.zeros((len(funcs), len(p_range)))
                    
                    for i, p in enumerate(tqdm.tqdm(p_range)):  # tqdm_notebook
                        set_p_edge(G, p_op=p)
                        for j, function in enumerate(funcs):
                            er, multipartite_gen_time, links_used = function(G, users, timesteps=timesteps, reps=reps)
                            ER[j, i] += er
                    
                    all_zeros = np.all(ER == 0, axis=1)
                    for idx, is_zero in enumerate(all_zeros):
                        if is_zero:  # 如果该 protocol 的结果全为 0
                            protocol_name = funcs[idx].__name__
                            failure_counts[protocol_name] += 1
                    
                    # er_output_path = er_topology_folder_path.joinpath(f'result_{combo_idx}')
                    # plot_er_vs_p(p_range, ER, funcs, cols, er_output_path)
        
        # 计算每个协议的 SR
        sr_for_protocols = {}
        for protocol_name, failures in failure_counts.items():
            sr_for_protocols[protocol_name] = (combination_counter - failures) / combination_counter if combination_counter > 0 else 0
        
        print(f"Subfolder: {subfolder}, SR: {sr_for_protocols}")
        
        # 保存结果到列表
        sr_for_protocols["Subfolder"] = subfolder.name
        sr_results.append(sr_for_protocols)

# 保存结果为 CSV 文件
output_csv_path = er_topology_folder_path.joinpath("protocol_sr_results.csv")
with open(output_csv_path, mode="w", newline="") as csvfile:
    fieldnames = ["Subfolder"] + [func.__name__ for func in funcs]
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(sr_results)

print(f"协议 SR 结果已保存到 {output_csv_path}")


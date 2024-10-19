import networkx as nx

# G = nx.Graph()
# G.add_edge(1, 2, weight=4.7)
# G.add_edge(2, 3, weight=6.7)

# # 获取 edges
# edges = G.edges().values()
# edges_with_data = G.edges(data= True)
# print(edges)  # 这是一个 EdgeView 对象
# print(edges_with_data)
# #print(list(edges_with_data))
# #print(list(edges))  # 转换为列表 [(1, 2), (2, 3)]

import networkx as nx

# 创建一个简单的图
G = nx.grid_2d_graph(3, 3)  # 3x3的网格图

# 此时，节点没有任何属性
print(G.nodes(data=True))  # 打印节点和属性
# 输出: [(0, 0, {}), (0, 1, {}), ..., (2, 2, {})]，所有节点属性为空字典

# 通过 set_node_attributes 动态为节点添加属性
nx.set_node_attributes(G, False, "entangled")
nx.set_node_attributes(G, 0, "age")

# 再次查看节点的属性
print(G.nodes(data=True))
print(type(G.nodes))
# 输出: [(0, 0, {'entangled': False, 'age': 0}), ..., (2, 2, {'entangled': False, 'age': 0})]

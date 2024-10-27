import networkx as nx
import matplotlib.pyplot as plt

# Function to create an approximate Steiner tree using MST
def steiner_tree_approximation(graph, terminal_nodes):
    # Create a subgraph from the original graph containing only the terminal nodes
    subgraph = graph.subgraph(terminal_nodes)
    
    # Compute the Minimum Spanning Tree (MST) for the subgraph
    mst = nx.minimum_spanning_tree(subgraph)
    
    return mst

# Create a sample graph (can be customized)
G = nx.Graph()

# Add nodes and edges with weights (edges can represent distances, costs, etc.)
edges = [
    (0, 1, 2), (0, 2, 3), (1, 2, 1), (1, 3, 4), (2, 4, 5), (3, 4, 1),
    (3, 5, 2), (4, 5, 3), (0, 6, 6), (6, 5, 4), (6, 3, 5), (6, 2, 3)
]
G.add_weighted_edges_from(edges)

# Define the terminal nodes (the nodes that must be connected by the Steiner tree)
terminal_nodes = [0, 3, 5]  # Example terminal nodes

# Compute the approximate Steiner tree (using MST approximation)
steiner_tree = steiner_tree_approximation(G, terminal_nodes)

# Draw the original graph with all edges
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)  # Layout for positioning nodes
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=600, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)})

# Highlight the Steiner tree
nx.draw(steiner_tree, pos, with_labels=True, node_color='green', node_size=700, font_size=10, edge_color='red', width=2)

# Set the title and display the graph
plt.title('Steiner Tree Approximation with MST for Terminal Nodes', size=15)
plt.show()



import networkx as nx
import numpy as np
import json
import os
from graph import (
    reset_graph_usage,
    update_graph_usage,
    update_usage_from_subgraph,
    reset_graph_state,
    get_entangled_subgraph,
)
from sim import run_entanglement_step

from networkx.algorithms.approximation.steinertree import steiner_tree
from protocols import MPC_protocol, MPG_protocol, SP_protocol
from graph import network, set_p_edge


def load_data(filename):
    file_path = os.path.abspath(os.path.join(os.getcwd(), "../graphs_json/", f"{filename}.json"))
    with open(file_path, "r") as file:
        return json.loads(file.read())
    
data = load_data("TOP_1_ABILENE_reordered")
G = nx.node_link_graph(data)
G = network(G)
nodes = list(G.nodes)

# funcs= [MPC_protocol,MPG_protocol,SP_protocol]
p_range = np.linspace(1, 0.2, 50)
ER = np.zeros(len(p_range))
timesteps = 100
reps = 200

users = [node["id"] for node in data["nodes"]]
er, multipartite_gen_time, avg_links_used,J = SP_protocol(G,users,timesteps,reps)

print(J.edges())
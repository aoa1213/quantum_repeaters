import numpy as np


def run_entanglement_step(G, used_nodes, nodes=False):

    """
    simulate the link generation and decoherence for a single timeslot (step)

    Input Pararmeters:
    G          - Networkx graph G(V,E) which defines the topology of the network. see graphs.py for more details
    used_nodes - List of paths of used nodes (only updated if nodes parameter is True)
    nodes      - (optional) include simulating "node" entanglement in model
    """

    r_list = np.random.rand(
        len(G.edges())
    )  
    # array of random numbers between 0 and 1 (size of number of edges), which is the probability of generating entanglement

    for edge, r in zip(G.edges().values(), r_list):


        #   zip() took out pairs of data from G.edges().values() and r_list and assign to edge and r correspondingly
        #   G.edges().values() is the dictionary of each edges in the G graph
        #   e.g. edge =  (0, 1, {“entangled” ： True, "p_edge" : 0.7, "age" : 2, "Qc"(quantum capacity) : 5})
        #   r_list is a array of float number between 0 to 1 with the same size as number of edge, which is actually the probability of 


        if edge["entangled"]:  #  if entangled edge exists increase age
            edge["age"] += 1

            if (
                edge["age"] >= edge["Qc"]
            ):  #   If the edge is now too old then discard it - only required for entangled edges
                #   Qc is the threshold to dicard the entagnled states
                edge["entangled"] = False
                edge["age"] = 0

        if (
            not edge["entangled"] and edge["p_edge"] > r
        ):  #   greater is correct (hint p_edge = 0, rand =  0) and (hint p_edge = 1, rand =  0.999...)
            #   if the current edge is not in entangled state and the probebility of tanglemnet generation is larger than r
            #   r is the random number generated at the start
            edge["entangled"] = True 
            #   entangled is in the dictionary 
            edge["age"] = 0

    if nodes:
        for node_name in G.nodes():
            node = G.nodes[node_name]

            if node["entangled"]:
                node["age"] += 1

            if node["age"] >= node["Qc"]:
                node["entangled"] = False
                edge["age"] = 0

        for path in used_nodes:
            path["age"] += 1

        used_nodes[:] = [
            path for path in used_nodes if path["age"] < G.nodes[path["destination_node"]]["Qc"]
        ]

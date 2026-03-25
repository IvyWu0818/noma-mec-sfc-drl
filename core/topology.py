import networkx as nx

def create_topology():
    G = nx.Graph()

    G.add_node("bs")
    G.add_node("mec0")
    G.add_node("mec1")
    G.add_node("mec2")

    G.add_edge("bs", "mec0", delay=2)
    G.add_edge("bs", "mec1", delay=3)
    G.add_edge("mec0", "mec2", delay=2)
    G.add_edge("mec1", "mec2", delay=2)

    return G
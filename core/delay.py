import networkx as nx

def uplink_delay(data_size, rate=10.0):
    return data_size / rate


def compute_delay(cpu_cycles, cpu_alloc):
    return cpu_cycles / cpu_alloc


def queue_delay(node):
    return node.estimate_waiting_time()


def forwarding_delay(graph, src, dst):
    path = nx.shortest_path(graph, src, dst, weight="delay")
    delay = 0
    for u, v in zip(path[:-1], path[1:]):
        delay += graph[u][v]["delay"]
    return delay


def local_stage_delay(graph, prev_node, node_id, vnf, cpu_alloc, mec_nodes):
    node = mec_nodes[node_id]
    total = queue_delay(node) + compute_delay(vnf.cpu_cycles, cpu_alloc)

    if prev_node is not None and prev_node != node_id:
        total += forwarding_delay(graph, prev_node, node_id)

    return total


def total_delay(task, graph, mec_nodes):
    total = uplink_delay(task.data_size)

    prev_node = None
    for i, vnf in enumerate(task.sfc_chain.vnfs):
        node_id = task.vnf_placement[i]
        cpu_alloc = task.cpu_alloc[i]
        total += local_stage_delay(graph, prev_node, node_id, vnf, cpu_alloc, mec_nodes)
        prev_node = node_id

    return total
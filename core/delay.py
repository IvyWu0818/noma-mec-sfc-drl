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


def total_delay(task, graph, mec_nodes):
    total = uplink_delay(task.data_size)

    for i, vnf in enumerate(task.sfc_chain.vnfs):
        node_id = task.vnf_placement[i]
        node = mec_nodes[node_id]

        total += queue_delay(node)
        total += compute_delay(vnf.cpu_cycles, task.cpu_alloc[i])

        if i < len(task.sfc_chain.vnfs) - 1:
            next_node = task.vnf_placement[i + 1]
            if node_id != next_node:
                total += forwarding_delay(graph, node_id, next_node)

    return total
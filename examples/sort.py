"""
My implementation of sorting with graph networks (GNs) in pytorch.
Original tf implementation here: https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/sort.ipynb
"""

import torch
import torch.nn as nn

from pgn.graph import Graph
from pgn.blocks import NodeBlock, EdgeBlock, GlobalBlock, GraphNetwork

import numpy as np
import matplotlib.pyplot as plt

def graph_from_list(input_list):
    """
    Takes a list with the data, generates a fully connected graph with values of the list as nodes
    Parameters
    ----------
    input_list: list
    List with the data

    Returns
    -------
    pgn.Graph with the input_list values as nodes
    """

    g = Graph()
    node_ids = [g.add_node(torch.Tensor([v])) for v in input_list]

    for sid in node_ids:
        for rid in node_ids:
            if sid != rid:
                # no loops
                _ = g.add_edge(sid, rid, data=torch.zeros(1))

    return g

def create_target_graph(input_graph):
    # two nodes might have true since they might have similar values

    target_graph = Graph()

    values = [(nid, nval) for nid, nval in input_graph.nodes.items()]

    min_value = min([v[1].data for v in values])

    for n in input_graph.nodes.values():
        # prob_true, prob_false
        target_graph.add_node(data=torch.Tensor([1.0, 0.0]) if n.data == min_value else torch.Tensor([0.0, 1.0]))

    sorted_values = sorted(values, key=lambda x: x[1].data)
    sorted_ids = [v[0] for v in sorted_values]

    for sidx, sid in enumerate(sorted_ids):
        for ridx, rid in enumerate(sorted_ids):
            if sid != rid:
                data = [0.0, 1.0]
                # we look for exact comparison here since we sort
                if input_graph.get_node_by_id(sid) == input_graph.get_node_by_id(rid) or \
                        (sidx < len(sorted_ids) - 1 and ridx == sidx + 1):
                    data = [1.0, 0.0]
                data = torch.Tensor(data)
                target_graph.add_edge(sid, rid, data=data)
    return target_graph

N_EPOCHS = 5000

if __name__ == '__main__':

    # build the graph
    unsorted = [3.0, 1.0, 2.0, 5.0, 4.0]
    input_g = graph_from_list(unsorted)

    target_graph = create_target_graph(input_g)
    _, target_nodes = torch.stack([n.data for n in target_graph.nodes.values()]).max(dim=1)
    _, target_edges = torch.stack([n.data for n in target_graph.edges.values()]).max(dim=1)

    input_node_size = 1
    input_edge_size = 1
    input_global_size = 1

    output_node_size = 2
    output_edge_size = 2
    output_global_size = 1

    # build the GN
    edge_updater = nn.Sequential(
        nn.Linear(input_edge_size + 2*input_node_size + input_global_size, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
        nn.Softmax(),
    )

    node_updater = nn.Sequential(
                       nn.Linear(input_edge_size + output_node_size + input_global_size, 10),
                       nn.ReLU(),
                       nn.Linear(10, 2),
                       nn.Softmax(),
                   )

    global_updater = nn.Sequential(
        nn.Linear(output_edge_size + output_node_size + output_global_size, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )

    input_g._global_attribute._data = torch.Tensor([0])

    nb = NodeBlock(node_updater)
    eb = EdgeBlock(edge_updater)
    gb = GlobalBlock(global_updater)

    gn = GraphNetwork(nb, eb, gb)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(lr=0.001, params=gn.parameters())

    # train
    for e in range(N_EPOCHS):
        optimiser.zero_grad()
        g = gn.forward(input_g)
        nodes_out = torch.stack([n.data for n in g.nodes.values()])
        edges_out = torch.stack([n.data for n in g.edges.values()])

        node_loss = criterion(nodes_out, target_nodes)
        edge_loss = criterion(edges_out, target_edges)

        loss = node_loss + edge_loss
        loss.backward()

        optimiser.step()
        if e % 100 == 0:
            print("Epoch %d, training loss: %f." % (e, loss.item()))


    # evaluate and plot
    mx = np.zeros((len(unsorted), len(unsorted)))
    for e in g.edges.values():
        mx[e.sender.id][e.receiver.id] = e.data[0]
        #mx[e.receiver.id][e.sender.id] = e.data[0]

    plt.imshow(mx[np.argsort(unsorted)][:, np.argsort(unsorted)], cmap="viridis")
    plt.show()

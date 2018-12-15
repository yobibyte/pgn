"""
My implementation of sorting with graph networks (GNs) in pytorch.
Original tf implementation here: https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/sort.ipynb
"""

import itertools

import torch
import torch.nn as nn

from pgn.graph import Graph, concat_graphs, copy_graph, copy_graph_topology
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
    connectivity = [el for el in itertools.product(range(len(input_list)), repeat=2) if el[0]!=el[1]]
    return Graph(nodes_data=torch.Tensor([[v] for v in input_list]),
                 edges_data=torch.zeros(len(connectivity), 1),
                 connectivity=connectivity
                 )


def create_target_graph(input_graph):
    # two nodes might have true since they might have similar values

    target_graph = copy_graph_topology(input_graph)


    nodes_data = [v.item() for v in input_graph.nodes_data]
    values = [(nid, ndata) for nid, ndata in enumerate(nodes_data)]
    min_value = min([v[1] for v in values])

    target_graph.nodes_data = torch.Tensor(
        # prob_true, prob_false
        [[1.0, 0.0] if v == min_value else [0.0, 1.0] for v in nodes_data])

    sorted_values = sorted(values, key=lambda x: x[1])
    sorted_ids = [v[0] for v in sorted_values]

    data = torch.zeros(input_graph.num_edges, 2)

    for sidx, sid in enumerate(sorted_ids):
        for ridx, rid in enumerate(sorted_ids):
            if sid != rid:
                # get edge id by sid and rid
                eid = input_graph.identify_edge_by_sender_and_receiver(sid, rid)

                # we look for exact comparison here since we sort
                if input_graph.nodes_data[sid] == input_graph.nodes_data[rid] or \
                        (sidx < len(sorted_ids) - 1 and ridx == sidx + 1):
                    data[eid][0] = 1.0
                else:
                    data[eid][1] = 1.0
    target_graph.edges_data = data

    return target_graph

N_EPOCHS = 5000

if __name__ == '__main__':

    # build the graph
    unsorted = [3.0, 1.0, 2.0, 5.0, 4.0, 24, 25, 42, 56, 1, 23]
    input_g = graph_from_list(unsorted)
    input_g.global_data = torch.Tensor([0])

    target_graph = create_target_graph(input_g)
    _, target_nodes = target_graph.nodes_data.max(dim=1)
    _, target_edges = target_graph.edges_data.max(dim=1)

    input_node_size = 1
    input_edge_size = 1
    input_global_size = 1

    output_node_size = 2
    output_edge_size = 2
    output_global_size = 1

    # build the GN
    # as in DM demo, 2 hidden layers of size 16 for enc/dec

    enc_edge_updater = nn.Sequential(
        nn.Linear(input_edge_size, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.LayerNorm(16),
    )

    enc_node_updater = nn.Sequential(
                       nn.Linear(input_node_size, 16),
                       nn.ReLU(),
                       nn.Linear(16, 16),
                       nn.ReLU(),
                       nn.LayerNorm(16),
                   )

    enc_global_updater = nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.LayerNorm(16),
    )

    encoder = GraphNetwork(NodeBlock(enc_node_updater, independent=True),
                           EdgeBlock(enc_edge_updater, independent=True),
                           GlobalBlock(enc_global_updater, independent=True))

    enc_global_output_size = 16
    # build the core ###########
    core_edge_input_size = 2*(16 + 2*16 + enc_global_output_size)
    core_node_input_size = 16 + 32 + 2*enc_global_output_size

    core_edge_updater = nn.Sequential(
        nn.Linear(core_edge_input_size, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.LayerNorm(16),
    )

    core_node_updater = nn.Sequential(
        nn.Linear(core_node_input_size, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.LayerNorm(16),
    )

    core_global_updater = nn.Sequential(
        nn.Linear(2*enc_global_output_size + 16 + 16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.LayerNorm(16),
    )

    NUM_PROCESSING_STEPS = 1
    core = GraphNetwork(NodeBlock(core_node_updater),
                        EdgeBlock(core_edge_updater),
                        GlobalBlock(core_global_updater))

    # build the decoder here
    dec_edge_updater = nn.Sequential(
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.LayerNorm(16),
        nn.Linear(16, 2),
    )

    dec_node_updater = nn.Sequential(
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.LayerNorm(16),
        nn.Linear(16, 2),
    )


    dec_global_updater = nn.Sequential(
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.LayerNorm(16),
        nn.Linear(16, 2),
    )

    decoder = GraphNetwork(NodeBlock(dec_node_updater, independent=True),
                           EdgeBlock(dec_edge_updater, independent=True),
                           GlobalBlock(dec_global_updater, independent=True),
                           )

    models = [encoder] + [core]*NUM_PROCESSING_STEPS + [decoder]


    criterion = nn.CrossEntropyLoss()
    parameters = list(encoder.parameters())+list(core.parameters())+list(decoder.parameters())
    optimiser = torch.optim.Adam(lr=0.001, params=parameters)

    # train
    for e in range(N_EPOCHS):
        optimiser.zero_grad()

        input_copy = copy_graph(input_g)
        latent = encoder(input_copy)
        latent0 = copy_graph(latent)

        for s in range(NUM_PROCESSING_STEPS):
            concatenated = concat_graphs([latent0, latent])
            latent = core(concatenated)

        g = decoder(latent)

        node_loss = criterion(g.nodes_data, target_nodes)
        edge_loss = criterion(g.edges_data, target_edges)

        loss = node_loss + edge_loss
        loss.backward()

        optimiser.step()
        if e % 100 == 0:
            print("Epoch %d, training loss: %f." % (e, loss.item()))

    # evaluate and plot
    mx = np.zeros((len(unsorted), len(unsorted)))
    for eid in range(g.num_edges):
        mx[g.senders[eid]][g.receivers[eid]] = g.edges_data[eid, 0]

    plt.imshow(mx[np.argsort(unsorted)][:, np.argsort(unsorted)], cmap="viridis")
    plt.show()

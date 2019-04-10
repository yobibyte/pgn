"""
My implementation of sorting with graph networks (GNs) in pytorch.
Original tf implementation here: https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/sort.ipynb
"""

import itertools

import torch
import torch.nn as nn

from pgn.graph import DirectedGraphWithContext, Vertex, DirectedEdge, Context
from pgn.blocks import NodeBlock, EdgeBlock, GlobalBlock, GraphNetwork
from pgn.aggregators import MeanAggregator

import argparse

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
    connectivity = [el for el in itertools.product(range(len(input_list)), repeat=2)]
    vertices = [Vertex(i) for i in range(len(input_list))]
    entities = [
        {'data': torch.Tensor([[v] for v in input_list]),
         'info': vertices},
        {'data': torch.zeros(len(connectivity), 1),
         'info': [DirectedEdge(i, vertices[connectivity[i][0]], vertices[connectivity[i][1]]) for i in range(len(connectivity))]},
        {'data': torch.Tensor([0]), 'info': [Context(0)]}
    ]

    return DirectedGraphWithContext(entities)


def create_target_graph(input_graph):
    # two nodes might have true since they might have similar values

    target_graph = input_graph.get_graph_with_same_topology()

    vertex_data = [v.item() for v in input_graph.vertex_data('vertex')]
    values = [(nid, ndata) for nid, ndata in enumerate(vertex_data)]
    min_value = min([v[1] for v in values])

    # [prob_true, prob_false]
    target_graph.set_vertex_data(torch.Tensor([[1.0, 0.0] if v == min_value else [0.0, 1.0] for v in vertex_data]))

    sorted_values = sorted(values, key=lambda x: x[1])
    sorted_ids = [v[0] for v in sorted_values]

    data = torch.zeros(input_graph.num_edges('edge'), 2)
    for sidx, sid in enumerate(sorted_ids):
        for ridx, rid in enumerate(sorted_ids):
            eid = input_graph.identify_edge_by_sender_and_receiver(sid, rid).id
            # we look for exact comparison here since we sort
            if (sidx < len(sorted_ids) - 1 and ridx == sidx + 1):
                data[eid][0] = 1.0
            else:
                data[eid][1] = 1.0

    target_graph.set_edge_data(data)

    return target_graph


def get_mlp(input_size, units, activation=nn.ReLU):
    arch = []
    inpt_size = input_size
    for l in units:
        arch.append(nn.Linear(inpt_size, l))
        arch.append(activation())
        inpt_size = l
    nn.LayerNorm(inpt_size)
    return nn.Sequential(*arch)


def get_mlp_updaters(input_node_size,
             input_edge_size,
             input_global_size,
             output_node_size,
             output_edge_size,
             output_global_size,
             independent):
    if independent:
        edge_updater = get_mlp(input_edge_size, [16, output_edge_size])
        node_updater = get_mlp(input_node_size, [16, output_node_size])
        global_updater = get_mlp(input_global_size, [16, output_global_size])
    else:
        # no aggregation of the outgoing for this example
        edge_updater = get_mlp(input_edge_size + 2*input_node_size + input_global_size, [16, output_edge_size])
        node_updater = get_mlp(input_node_size + output_edge_size + input_global_size, [16, output_node_size])
        global_updater = get_mlp(input_global_size + output_edge_size + output_node_size, [16, output_global_size])
    return node_updater, edge_updater, global_updater

def forward(input_g, encoder, core, decoder, core_steps):
    input_copy = input_g.get_copy()
    latent = encoder(input_copy)
    latent0 = latent.get_copy()
    output = []
    for s in range(core_steps):
        concatenated = DirectedGraphWithContext.concat([latent0, latent])
        latent = core(concatenated)
        output.append(decoder(latent))
    return output

def generate_graph_batch(n_examples, sample_length, target=True):
    input_graphs = [graph_from_list(np.random.uniform(size=sample_length)) for _ in range(n_examples)]
    target_graphs = [create_target_graph(g) for g in input_graphs] if target else None
    if target_graphs is not None:
        return input_graphs, target_graphs
    else:
        return input_graphs

def process_batch(encoder, core, decoder, core_steps, input_graphs, target_graphs, compute_grad=True):
    if not compute_grad:
        encoder.eval()
        core.eval()
        decoder.eval()

    node_loss = 0
    edge_loss = 0
    for input_g, target_g in zip(input_graphs, target_graphs):
        g_out = forward(input_g, encoder, core, decoder, core_steps)
        node_loss += sum([criterion(g.vertex_data('vertex'), target_g.vertex_data('vertex')) for g in g_out])
        edge_loss += sum([criterion(g.edge_data('edge'), target_g.edge_data('edge')) for g in g_out])
    loss = node_loss + edge_loss

    if not compute_grad:
        encoder.train()
        core.train()
        decoder.train()
    return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sorting with graph networks')
    parser.add_argument('--num-train', type=int, default=10, help='number of training examples')
    parser.add_argument('--num-eval', type=int, default=10, help='number of evaluation examples')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--core-steps', type=int, default=10, help='number of core processing steps')
    parser.add_argument('--sample-length', type=int, default=10, help='number of elements in the list to sort')
    parser.add_argument('--eval-freq', type=int, default=100, help='Evaluation/logging frequency')
    args = parser.parse_args()

    train_input_graphs, train_target_graphs = generate_graph_batch(args.num_train, sample_length=args.sample_length)
    eval_input_graphs, eval_target_graphs = generate_graph_batch(args.num_train, sample_length=args.sample_length)


    enc_node_updater, enc_edge_updater, enc_global_updater = get_mlp_updaters(1, 1, 1, 16, 16, 16, independent=True)
    encoder = GraphNetwork(NodeBlock({'vertex': enc_node_updater}),
                           EdgeBlock({'edge': enc_edge_updater}, independent=True),
                           GlobalBlock({'context': enc_global_updater}))

    core_node_updater, core_edge_updater, core_global_updater = get_mlp_updaters(32, 32, 32, 16, 16, 16, independent=False)
    core = GraphNetwork(NodeBlock({'vertex':core_node_updater}, {'edge': MeanAggregator()}),
                        EdgeBlock({'edge':core_edge_updater}),
                        GlobalBlock({'context':core_global_updater}, {'vertex': MeanAggregator()}, {'edge': MeanAggregator()}))

    dec_node_updater, dec_edge_updater, dec_global_updater = get_mlp_updaters(16, 16, 16, 16, 16, 16, independent=True)
    decoder = GraphNetwork(NodeBlock({'vertex':nn.Sequential(dec_node_updater, nn.Linear(16, 2))}),
                           EdgeBlock({'edge':nn.Sequential(dec_edge_updater, nn.Linear(16, 2))}, independent=True),
                           GlobalBlock({'context':nn.Sequential(dec_global_updater, nn.Linear(16, 2))}),
                           )
    models = [encoder] + [core] * args.core_steps + [decoder]

    criterion = nn.BCEWithLogitsLoss()

    parameters = []
    for el in [encoder, core, decoder]:
        for plist in el.parameters():
            parameters.extend(list(plist))
    optimiser = torch.optim.Adam(lr=0.001, params=parameters)

    # train
    for e in range(args.epochs):
        optimiser.zero_grad()
        train_loss = process_batch(encoder, core, decoder, args.core_steps, train_input_graphs, train_target_graphs)
        train_loss.backward()
        optimiser.step()
        if e % args.eval_freq == 0 or e == args.epochs-1:
            eval_loss = process_batch(encoder, core, decoder, args.core_steps, eval_input_graphs, eval_target_graphs,
                                      compute_grad=False)
            print("Epoch %d, mean training loss: %f, mean evaluation loss: %f."
                  % (e, train_loss.item()/args.num_train, eval_loss.item()/args.num_train))

    unsorted = np.random.uniform(size=args.sample_length)
    test_g = graph_from_list(unsorted)
    test_g.set_context_data(torch.Tensor([0]))
    g = forward(test_g, encoder, core, decoder, args.core_steps)[-1]

    # evaluate and plot
    mx = np.zeros((len(unsorted), len(unsorted)))
    for eid in range(g.num_edges('edge')):
        mx[g.senders(eid, 'edge').id][g.receivers(eid, 'edge').id] = g.edge_data('edge')[eid, 0]
    sort_indices = np.argsort(unsorted)
    plt.matshow(mx[sort_indices][:, sort_indices], cmap="viridis")
    plt.grid(False)
    plt.show()
"""
My implementation of finding the shortest path in graphs using graph networks (GNs) in pytorch.
Original tf implementation here: https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos/shortest_path.ipynb
"""
import argparse
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pgn.graph import Vertex, DirectedEdge, Context, DirectedGraphWithContext
from pgn.models import EncoderCoreDecoder
from pgn.utils import pgn2nx, plot_graph

import networkx as nx
from shortest_path_helper import *

seed = 1  #@param{type: 'integer'}
rand = np.random.RandomState(seed=seed)

torch.set_num_threads(4)

def add_connectivity(entities):
    """Add the information about incoming/outgoing edges for the graph vertices.

    Parameters
    ----------
    entities: dict
        dict of graph entities

    """

    for v in entities['vertex']['vertex']['info']:
        incoming = {'edge': [e for e in entities['edge']['edge']['info'] if e.receiver.id == v.id]}
        outgoing = {'edge': [e for e in entities['edge']['edge']['info'] if e.sender.id == v.id]}
        v.incoming_edges = incoming
        v.outgoing_edges = outgoing

def networkx_to_digraph(graph):
    """"Converts a networkx graph to a torch graph

        Parameters
        ----------
        graph: nx.Graph
    """
    connectivity = [edge for edge in graph.edges()]
    vertices = [Vertex(i) for i in range(graph.number_of_nodes())]
    edges = [DirectedEdge(i, vertices[connectivity[i][0]], vertices[connectivity[i][1]]) for i in
             range(len(connectivity))]
    node_features = [node[1]["features"] for node in graph.nodes(data=True)]
    edge_features = [edge[2]["features"] for edge in graph.edges(data=True)]
    entities = {'vertex': {'vertex': {'data': torch.Tensor(node_features), 'info': vertices}},
                  'edge': {'edge': {'data': torch.Tensor(edge_features), 'info': edges}},
                  'context': {'context': {'data': torch.Tensor([[0]]), 'info': [Context(0)]}}
               }
    add_connectivity(entities)
    return entities

def generate_graph_batch(num_examples, theta, num_nodes_min_max):
    """"Generates a batch of graphs

        Parameters
        ----------
        num_examples: int
            number of graphs to generate
        theta: float
            a threshold parameter for generating
        num_nodes_min_max: (int, int)
            a tuple bounding the minimum and maximum number of nodes in the graph

    """
    input_graphs, target_graphs, _ = generate_networkx_graphs(
        rand, num_examples, num_nodes_min_max, theta)
    input_data = [networkx_to_digraph(graph) for graph in input_graphs]
    target_data = [networkx_to_digraph(graph) for graph in target_graphs]
    return input_data, target_data

def batch_loss(outs, targets, criterion):
    """get the loss for the network outputs

    Parameters
    ----------
    outs: list
        list of lists of the graph network output, time is 0-th dimension, batch is 1-th dimension
    targets: list
        list of the graph entities for the expected output
    criterion: torch._Loss object
        loss to use
    Returns
    -------
    loss: float
        Shows how good your mode is.
    """
    loss = 0
    for out in outs:
        loss += sum([criterion(g['vertex']['vertex']['data'], t['vertex']['vertex']['data']) for g, t in zip(out, targets)])
        loss += sum([criterion(g['edge']['edge']['data'], t['edge']['edge']['data']) for g, t in zip(out, targets)])
    return loss

def accuracy(outs, targets):
    print(outs)
    print(targets)
def run():
    """Run an experiment"""
    parser = argparse.ArgumentParser(description='Finding the shortest path in a graph with graph networks')
    parser.add_argument('--num-train', type=int, default=32, help='number of training examples')
    parser.add_argument('--num-eval', type=int, default=32, help='number of evaluation examples')
    parser.add_argument('--epochs', type=int, default=5000, help='number of training epochs')
    parser.add_argument('--core-steps', type=int, default=10, help='number of core processing steps')
    parser.add_argument('--sample-length', type=int, default=10, help='number of elements in the list to sort')
    parser.add_argument('--eval-freq', type=int, default=100, help='Evaluation/logging frequency')
    parser.add_argument('--cuda', action="store_true", help='Use a GPU if the system has it.')
    parser.add_argument('--verbose', action="store_true", help='Print diagnostircs.')
    parser.add_argument('--plot_graph_sample', action="store_true", help='Plot one of the input graphs')
    args = parser.parse_args()



    # generate training and testing graphs
    # Large values (1000+) make trees. Try 20-60 for good non-trees.
    theta = 20  #@param{type: 'integer'}
    train_size = 32
    eval_size = 100
    # Number of nodes per graph sampled uniformly from this range.
    train_num_nodes = 8
    eval_num_nodes = 16
    train_min_max = (8, 17)
    eval_min_max = (16, 33)

    train_input, train_target = generate_graph_batch(args.num_train, theta, train_min_max)
    eval_input, eval_target = generate_graph_batch(args.num_eval, theta, eval_min_max)
    # print("train_input")
    # print(train_input)
    model = EncoderCoreDecoder(args.core_steps,
                               enc_vertex_shape=(5, 16),
                               core_vertex_shape=(32, 16),
                               dec_vertex_shape=(16, 16),
                               out_vertex_size=2,
                               enc_edge_shape=(1, 16),
                               core_edge_shape=(32, 16),
                               dec_edge_shape=(16, 16),
                               out_edge_size=2,
                               enc_global_shape=(1, 16),
                               core_global_shape=(32, 16),
                               dec_global_shape=(16, 16),
                               out_global_size=1,
                               )

    # plot one of the input graphs
    if args.plot_graph_sample:
        ng = pgn2nx(train_input[0])
        plot_graph(ng, fname='input_graph.pdf')

    if args.cuda and torch.cuda.is_available():
        for el in train_input + eval_input:
            for d in el.values():
                d['data'] = d['data'].to('cuda')
        for el in train_target + eval_target:
            for k in el.keys():
                el[k] = el[k].to('cuda')
        model.to('cuda')

    optimiser = torch.optim.Adam(lr=0.001, params=model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    for e in range(args.epochs):
        st_time = time.time()
        #process one graph at a time, ie. batch_size = 1
        for input_graph, target_graph in zip(train_input, train_target):
            train_out = model.process_batch([input_graph])
            train_loss = batch_loss(train_out, [target_graph], criterion)
            optimiser.zero_grad()
            train_loss.backward()
            optimiser.step()

        end_time = time.time()
        if args.verbose:
            print('Epoch {} is done. {:.2f} sec spent.'.format(e, end_time - st_time))

        if e % args.eval_freq == 0 or e == args.epochs - 1:
            eval_loss = 0.0
            for input_graph, target_graph in zip(eval_input, eval_target):
                eval_outs = model.process_batch([input_graph], compute_grad=False)
                eval_loss += batch_loss(eval_outs, [target_graph], criterion)
            print("Epoch %d, mean training loss: %f, mean evaluation loss: %f."
                  % (e, train_loss.item() / args.num_train, eval_loss.item() / args.num_train))

if __name__ == '__main__':
    run()

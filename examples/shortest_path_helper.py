"""
My implementation of finding the shortest path in graphs using graph networks (GNs) in pytorch.
File containing helper functions
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

import collections
import networkx as nx
from scipy import spatial

DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot


def get_node_dict(graph, attr):
    """Return a `dict` of node:attribute pairs from a graph."""
    return {k: v[attr] for k, v in graph.node.items()}

def generate_graph(rand,
                   num_nodes_min_max,
                   dimensions=2,
                   theta=1000.0,
                   rate=1.0):
    """Creates a connected graph.
    Extracted from original tf implementation.

    The graphs are geographic threshold graphs, but with added edges via a
    minimum spanning tree algorithm, to ensure all nodes are connected.

    Args:
    rand: A random seed for the graph generator. Default= None.
    num_nodes_min_max: A sequence [lower, upper) number of nodes per graph.
    dimensions: (optional) An `int` number of dimensions for the positions.
      Default= 2.
    theta: (optional) A `float` threshold parameters for the geographic
      threshold graph's threshold. Large values (1000+) make mostly trees. Try
      20-60 for good non-trees. Default=1000.0.
    rate: (optional) A rate parameter for the node weight exponential sampling
      distribution. Default= 1.0.

    Returns:
    The graph.
    """
    # Sample num_nodes.
    num_nodes = rand.randint(*num_nodes_min_max)

    # Create geographic threshold graph.
    pos_array = rand.uniform(size=(num_nodes, dimensions))
    pos = dict(enumerate(pos_array))
    weight = dict(enumerate(rand.exponential(rate, size=num_nodes)))
    geo_graph = nx.geographical_threshold_graph(
      num_nodes, theta, pos=pos, weight=weight)

    # Create minimum spanning tree across geo_graph's nodes.
    distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
    i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
    weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
    mst_graph = nx.Graph()
    mst_graph.add_weighted_edges_from(weighted_edges, weight=DISTANCE_WEIGHT_NAME)
    mst_graph = nx.minimum_spanning_tree(mst_graph, weight=DISTANCE_WEIGHT_NAME)
    # Put geo_graph's node attributes into the mst_graph.
    for i in mst_graph.nodes():
        mst_graph.node[i].update(geo_graph.node[i])

    # Compose the graphs.
    combined_graph = nx.compose_all((mst_graph, geo_graph.copy()))
    # Put all distance weights into edge attributes.
    for i, j in combined_graph.edges():
      combined_graph.get_edge_data(i, j).setdefault(DISTANCE_WEIGHT_NAME,
                                                  distances[i, j])
    return combined_graph, mst_graph, geo_graph


def add_shortest_path(rand, graph, min_length=1):
    """Samples a shortest path from A to B and adds attributes to indicate it.
    Extracted from original tf implementation.

    Args:
    rand: A random seed for the graph generator. Default= None.
    graph: A `nx.Graph`.
    min_length: (optional) An `int` minimum number of edges in the shortest
      path. Default= 1.

    Returns:
    The `nx.DiGraph` with the shortest path added.

    Raises:
    ValueError: All shortest paths are below the minimum length
    """
    # Map from node pairs to the length of their shortest path.
    pair_to_length_dict = {}
    try:
        # This is for compatibility with older networkx.
        lengths = nx.all_pairs_shortest_path_length(graph).items()
    except AttributeError:
        # This is for compatibility with newer networkx.
        lengths = list(nx.all_pairs_shortest_path_length(graph))
    for x, yy in lengths:
        for y, l in yy.items():
            if l >= min_length:
                pair_to_length_dict[x, y] = l
    if max(pair_to_length_dict.values()) < min_length:
        raise ValueError("All shortest paths are below the minimum length")
    # The node pairs which exceed the minimum length.
    node_pairs = list(pair_to_length_dict)

    # Computes probabilities per pair, to enforce uniform sampling of each
    # shortest path lengths.
    # The counts of pairs per length.
    counts = collections.Counter(pair_to_length_dict.values())
    prob_per_length = 1.0 / len(counts)
    probabilities = [
        prob_per_length / counts[pair_to_length_dict[x]] for x in node_pairs
    ]

    # Choose the start and end points.
    i = rand.choice(len(node_pairs), p=probabilities)
    start, end = node_pairs[i]
    path = nx.shortest_path(
        graph, source=start, target=end, weight=DISTANCE_WEIGHT_NAME)

    # Creates a directed graph, to store the directed path from start to end.
    digraph = graph.to_directed()

    # Add the "start", "end", and "solution" attributes to the nodes and edges.
    digraph.add_node(start, start=True)
    digraph.add_node(end, end=True)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [start]), start=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [end]), end=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), path), solution=False)
    digraph.add_nodes_from(path, solution=True)
    path_edges = list(pairwise(path))
    digraph.add_edges_from(set_diff(digraph.edges(), path_edges), solution=False)
    digraph.add_edges_from(path_edges, solution=True)

    return digraph

def graph_to_input_target(graph):
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
    graph: An `nx.DiGraph` instance.

    Returns:
    The input `nx.DiGraph` instance.
    The target `nx.DiGraph` instance.

    Raises:
    ValueError: unknown node type
    """

    def create_feature(attr, fields):
        return np.hstack([np.array(attr[field], dtype=float) for field in fields])

    input_node_fields = ("pos", "weight", "start", "end")
    input_edge_fields = ("distance",)
    target_node_fields = ("solution",)
    target_edge_fields = ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    solution_length = 0
    for node_index, node_feature in graph.nodes(data=True):
        input_graph.add_node(
            node_index, features=create_feature(node_feature, input_node_fields))
        target_node = to_one_hot(
            create_feature(node_feature, target_node_fields).astype(int), 2)[0]
        target_graph.add_node(node_index, features=target_node)
        solution_length += int(node_feature["solution"])
    solution_length /= graph.number_of_nodes()

    for receiver, sender, features in graph.edges(data=True):
        input_graph.add_edge(
            sender, receiver, features=create_feature(features, input_edge_fields))
        target_edge = to_one_hot(
            create_feature(features, target_edge_fields).astype(int), 2)[0]
        target_graph.add_edge(sender, receiver, features=target_edge)

    input_graph.graph["features"] = np.array([0.0])
    target_graph.graph["features"] = np.array([solution_length], dtype=float)

    return input_graph, target_graph


def generate_networkx_graphs(rand, num_examples, num_nodes_min_max, theta):
    """Generate graphs for training.

    Args:
    rand: A random seed (np.RandomState instance).
    num_examples: Total number of graphs to generate.
    num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
      graph. The number of nodes for a graph is uniformly sampled within this
      range.
    theta: (optional) A `float` threshold parameters for the geographic
      threshold graph's threshold. Default= the number of nodes.

    Returns:
    input_graphs: The list of input graphs.
    target_graphs: The list of output graphs.
    graphs: The list of generated graphs.
    """
    input_graphs = []
    target_graphs = []
    graphs = []
    for _ in range(num_examples):
        graph = generate_graph(rand, num_nodes_min_max, theta=theta)[0]
        graph = add_shortest_path(rand, graph)
        input_graph, target_graph = graph_to_input_target(graph)
        input_graphs.append(input_graph)
        target_graphs.append(target_graph)
        graphs.append(graph)
    return input_graphs, target_graphs, graphs

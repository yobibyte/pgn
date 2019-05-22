"""
My implementation of sorting with graph networks (GNs) in pytorch.
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

torch.set_num_threads(4)


def generate_graph():
    # Sample num_nodes
    # Create geographic threshold graph
    # Create minimum spanning tree across geo_graph's nodes
    # Put geo_graph's node attributes into the mst_graph
    # Compose the graphs.
    # Put all distance weights into edge attributes.
    return

def run():
    """Run an experiment"""
    parser = argparse.ArgumentParser(description='Finding the shortest path in a graph with graph networks')


if __name__ == '__main__':
    run()

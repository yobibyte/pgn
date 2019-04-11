import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

import torch
from pgn.graph import DirectedEdge, DirectedGraph, Vertex

def generate_graph():
    vinfo = [Vertex(id=i) for i in range(2)]
    vertices = {'data': torch.Tensor([[0], [1]]), 'info': vinfo}

    edges = {'data': torch.Tensor([[0], [1], [2]]),
             'info': [
                 DirectedEdge(0, vinfo[0], vinfo[1]),
                 DirectedEdge(1, vinfo[0], vinfo[1]),
                 DirectedEdge(2, vinfo[1], vinfo[0])
             ]}

    return DirectedGraph([vertices, edges])


def pgn2nx(ig):
    G = nx.MultiDiGraph()

    for v in ig.vertex_info('vertex'):
        G.add_node(v.id)

    for e in ig.edge_info('edge'):
        G.add_edge(e.sender.id, e.receiver.id, key=e.id)

    pos = nx.spring_layout(G)

    for n in G.node:
        G.node[n].update({'pos': pos[n]})
    return G

def plot_graph(g, fname='graph.pdf'):
    # based on https://stackoverflow.com/a/49386888/1768248

    # add graphviz layout options (see https://stackoverflow.com/a/39662097)
    g.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
    g.graph['graph'] = {'scale': '3'}

    # adding attributes to edges in multigraphs is more complicated but see
    # https://stackoverflow.com/a/26694158
    g[1][0][2].update({'color': 'red'})

    a = to_agraph(g)
    a.layout('dot')
    a.draw(fname)

g = generate_graph()
ng = pgn2nx(g)
plot_graph(ng)
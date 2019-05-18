import networkx as nx
import torch
from networkx.drawing.nx_agraph import to_agraph
from pgn.graph import DirectedEdge, DirectedGraph, Vertex
import cProfile
import time

EDGE_COLOURS = {'edge': 'black', 'action': 'black', 'relation': 'orange'}
LAYOUTS = ['neato', 'dot', 'twopi', 'circo', 'fdp', 'sfdp']


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
    ig = DirectedGraph(ig)
    G = nx.MultiDiGraph()
    for t, vinfo in ig.vertex_info().items():
        for v in vinfo:
            colour = 'black' if v.hidden_info is None or 'colour' not in v.hidden_info else v.hidden_info['colour']
            label = str(v.id) if v.hidden_info is None or 'label' not in v.hidden_info else 'v.id: {},\n'.format(v.id) + str(v.hidden_info) + ' ' + str(ig.vertex_data(t)[v.id])
            G.add_node((v.type, v.id), color=colour, label=label)

    for t, einfo in ig.edge_info().items():
        for e in einfo:
            colour = 'black' if e.type not in EDGE_COLOURS else EDGE_COLOURS[e.type]
            basic_info = 'id: {}, data: {}'.format(e.id, ig.edge_data(t)[e.id])
            G.add_edge((e.sender.type, e.sender.id), (e.receiver.type, e.receiver.id), color=colour, label=basic_info)
    pos = nx.spring_layout(G)

    for n in G.node:
        G.node[n].update({'pos': pos[n], })
    return G


def plot_graph(g, fname='graph.pdf'):
    # based on https://stackoverflow.com/a/49386888/1768248

    # add graphviz layout options (see https://stackoverflow.com/a/39662097)
    g.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
    g.graph['node'] = {'shape': 'box'}
    g.graph['graph'] = {'scale': '1'}

    # adding attributes to edges in multigraphs is more complicated but see
    # https://stackoverflow.com/a/26694158

    a = to_agraph(g)
    a.layout('circo')
    a.draw(fname)


def profileit(name):
    # https://stackoverflow.com/a/5376616
    def inner(func):
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            # Note use of name from outer scope
            prof.dump_stats(name)
            return retval
        return wrapper
    return inner

def timeit(name):
    def inner(func):
        def wrapper(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            print('%s, took: %2.4f sec' % (name, te-ts))
            return result
        return wrapper
    return inner

if __name__ == '__main__':
    g = generate_graph()
    ng = pgn2nx(g)
    plot_graph(ng)

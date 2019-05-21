import cProfile
import time

import networkx as nx
import torch
from networkx.drawing.nx_agraph import to_agraph
from pgn.graph import DirectedEdge, DirectedGraph, Vertex

# TODO this, probably, should go to the pymarl code
EDGE_COLOURS = {'edge': 'black', 'action': 'black', 'relation': 'orange'}
LAYOUTS = ['neato', 'dot', 'twopi', 'circo', 'fdp', 'sfdp']  # layouts that worked on my machine


def generate_graph():
    """Method to generate a simple graph for testing purposes"""

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
    """

    Parameters
    ----------
    ig: dict
        A dictionary of entities

    Returns
    -------
    G: nx.MultiDiGraph
        A graph converted to the format networkx can plot.
    """

    ig = DirectedGraph(ig)
    G = nx.MultiDiGraph()
    for t, vinfo in ig.vertex_info().items():
        for v in vinfo:
            colour = 'black' if v.hidden_info is None or 'colour' not in v.hidden_info else v.hidden_info['colour']
            label = str(v.id) if v.hidden_info is None or 'label' not in v.hidden_info else 'v.id: {},\n'.format(
                v.id) + str(v.hidden_info) + ' ' + str(ig.vertex_data(t)[v.id])
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
    """Plot a networkx directed multigraph

    This was not easy to do at all. Plotting directed multigraphs neatly in python is an UNSOLVED problem.

    * based on https://stackoverflow.com/a/49386888/1768248
    * adding attributes to edges in multigraphs is more complicated but see
        https://stackoverflow.com/a/26694158
    * add graphviz layout options (see https://stackoverflow.com/a/39662097)

    Parameters
    ----------
    g
    fname

    Returns
    -------

    """

    g.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
    g.graph['node'] = {'shape': 'box'}
    g.graph['graph'] = {'scale': '1'}

    a = to_agraph(g)
    a.layout('circo')  # circo was the friendliest layout for me
    a.draw(fname)


def profileit(name):
    """Profiling decorator

    Add @profileit('your_name') for a function to run cprof on it
    Based on https://stackoverflow.com/a/5376616

    Parameters
    ----------
    name: str
        name of the file to save profiling results to

    Returns
    -------

    """

    #
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
    """Timing decorator

    Add @timeit('name') to time execution of the function.
    Built in the same way as profileit above.

    Parameters
    ----------
    name: str
        Name to see in the output.

    Returns
    -------

    """

    def inner(func):
        def wrapper(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            print('%s, took: %2.4f sec' % (name, te - ts))
            return result

        return wrapper

    return inner


if __name__ == '__main__':
    g = generate_graph()
    ng = pgn2nx(g)
    plot_graph(ng)

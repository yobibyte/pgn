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


def concat_entities(entities, template):
    res = {k: {et: {} for et in v} for k, v in template}
    for et, ed in res.items():
        for est in ed:
            res[et][est]['data'] = torch.cat([el[et][est]['data'] for el in entities], dim=1)
            res[et][est]['info'] = entities[0][et][est]['info']
    return res

if __name__ == '__main__':
    g = generate_graph()
    ng = pgn2nx(g)
    plot_graph(ng)

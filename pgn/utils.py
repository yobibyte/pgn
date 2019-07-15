import networkx as nx
import torch
import numpy as np
from networkx.drawing.nx_agraph import to_agraph
from pgn.graph import DirectedGraph

# TODO this, probably, should go to the pymarl code
EDGE_COLOURS = {'edge': 'black', 'action': 'black', 'relation': 'orange'}
LAYOUTS = ['neato', 'dot', 'twopi', 'circo', 'fdp', 'sfdp']  # layouts that worked on my machine

def batch_data(graph_list):
    vdata = [el[0] for el in graph_list]
    edata = [el[1] for el in graph_list]
    connectivity = [el[2] for el in graph_list]
    if len(graph_list[0]) > 3:
        cdata = torch.cat([el[3] for el in graph_list])
    else:
        cdata = None

    if type(edata[0]) != dict:
        edata = [{'default': el} for el in edata]
        connectivity = [{'default': el} for el in connectivity]

    vsizes = [v.shape[0] for v in vdata]
    vdata = torch.cat(vdata)
    # now we need to fix connectivity since we stacked different graphs
    vcumsum = np.cumsum(vsizes)


    esizes = {}
    edata_d = {}
    connectivity_d = {}
    for et in edata[0]:
        esizes[et] = [e[et].shape[0] for e in edata]
        edata_d[et] = torch.cat([el[et] for el in edata])
        connectivity_d[et] = torch.cat([conn[et] for conn in connectivity],dim=1)
        # # TODO replace this with roll and setting the last element to 0
        # correction = torch.cat([torch.zeros(esizes[et][0],dtype=torch.long), *[torch.tensor([el]*esizes[et][i],dtype=torch.long) for i,el in enumerate(vcumsum[:-1])]])
        # connectivity_d[et] += correction

    return vdata, edata_d, connectivity_d, cdata, {'vsizes': vsizes, 'esizes': esizes}


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


def concat_entities(entities):
    has_global = len(entities[0]) == 3 and entities[0][2] is not None

    v = torch.cat([el[0] for el in entities], dim=1)

    e = {}
    for k in entities[0][1].keys():
        e[k] = torch.cat([el[1][k] for el in entities], dim=1)

    c = None
    if has_global:
        c = torch.cat([el[2] for el in entities], dim=1)

    return v, e, c

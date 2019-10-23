import networkx as nx
import torch
import numpy as np
from networkx.drawing.nx_agraph import to_agraph

LAYOUTS = [
    "neato",
    "dot",
    "twopi",
    "circo",
    "fdp",
    "sfdp",
]  # layouts that worked on my machine


def batch_data(graph_list):
    """
    Batch multiple graphs into one huge graph consisting of disconnected subgraphs.
    Parameters
    ----------
    graph_list: list
    List of graph tuples.

    Returns a graph tuple and its metadata
    -------

    """
    vdata = [el[0] for el in graph_list]
    edata = [el[1] for el in graph_list]
    connectivity = [el[2] for el in graph_list]
    if len(graph_list[0]) > 3:
        cdata = torch.cat([el[3] for el in graph_list])
    else:
        cdata = None

    if type(edata[0]) != dict:
        edata = [{"default": el} for el in edata]
        connectivity = [{"default": el} for el in connectivity]

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
        connectivity_d[et] = torch.cat([conn[et] for conn in connectivity], dim=1)
        # # TODO replace this with roll and setting the last element to 0
        correction = torch.cat(
            [
                torch.zeros(esizes[et][0], dtype=torch.long, device=vdata.device),
                *[
                    torch.tensor(
                        [el] * esizes[et][i], dtype=torch.long, device=vdata.device
                    )
                    for i, el in enumerate(vcumsum[:-1])
                ],
            ]
        )
        connectivity_d[et] += correction

    return vdata, edata_d, connectivity_d, cdata, {"vsizes": vsizes, "esizes": esizes}


def pgn2nx(g):
    # TODO this is broken after refactoring. Needs moving everything from classes to tuples.

    """

    Parameters
    ----------
    g: A tuple (vdata, edata, connectivity)

    Returns
    -------
    G: nx.MultiDiGraph
        A graph converted to the format networkx can plot.
    """

    vdata, edata, connectivity = g
    G = nx.MultiDiGraph()
    for v_id, v_data in enumerate(vdata):
            G.add_node((v_id), color="black", label="v.id: {},\n data: {}".format(v_id, v_data.numpy()))

    for e_id, e_data in enumerate(edata):
            G.add_edge(connectivity[0][e_id].item(), connectivity[1][e_id].item(), color="black",
                label="id: {}, data: {}".format(e_id, e_data.numpy()))

    pos = nx.spring_layout(G)
    for n in G.nodes:
        G.nodes[n].update({"pos": pos[n]})
    return G


def plot_graph(g, fname="graph.pdf"):
    """Plot a networkx directed multigraph

    This was not easy to do at all. Plotting directed multigraphs neatly in python is an UNSOLVED problem.

    * based on https://stackoverflow.com/a/49386888/1768248
    * adding attributes to edges in multigraphs is more complicated but see
        https://stackoverflow.com/a/26694158
    * add graphviz layout options (see https://stackoverflow.com/a/39662097)

    Parameters
    ----------
    g: A tuple (vdata, edata, connectivity)
        graph to plot
    fname: string
        filepath to save with

    Returns
    -------

    """

    g = pgn2nx(g)

    g.graph["edge"] = {"arrowsize": "0.6", "splines": "curved"}
    g.graph["node"] = {"shape": "box"}
    g.graph["graph"] = {"scale": "1"}

    a = to_agraph(g)
    a.layout("circo")  # circo was the friendliest layout for me
    a.draw(fname)


def concat_entities(entities):
    """
    Concat multiple graphs via concatenation of their entities tensors.

    Parameters
    ----------
    entities: a list of graph tuples.
        Either [(v,e,c),...] or [(v,e),...] when the graph has no global attribute.
    Returns v,e,c - concatenated entities tensors, None for c if no global in the graph.
    -------

    """


    has_global = len(entities[0]) == 3 and entities[0][2] is not None

    v = torch.cat([el[0] for el in entities], dim=1)

    e = {}
    for k in entities[0][1].keys():
        e[k] = torch.cat([el[1][k] for el in entities], dim=1)

    c = None
    if has_global:
        c = torch.cat([el[2] for el in entities], dim=1)

    return v, e, c

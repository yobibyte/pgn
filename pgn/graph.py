import copy
import torch

class AttributeType:
    def __init__(self, name, type):
        self._name = name
        self._type = type

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

class GraphAttribute:
    def __init__(self, data=None, type=None, id=None):
        self._data = data
        self._type = type
        self._id = id

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def type(self):
        return self._type

class Edge(GraphAttribute):
    def __init__(self, sender, receiver, data=None, type=None, id=None):
        self._sender = sender
        self._receiver = receiver
        super().__init__(data, type, id)

    @property
    def sender(self):
        return self._sender

    @property
    def receiver(self):
        return self._receiver

class Node(GraphAttribute):
    def __init__(self, data=None, type=None, id=None):
        super().__init__(data, type, id)
        self._incoming_edges = {}
        self._outcoming_edges = {}

    @property
    def incoming_edges(self):
        return self._incoming_edges

    @property
    def outcoming_edges(self):
        return self._outcoming_edges

    def add_incoming_edge(self, edge):
        self._incoming_edges[edge.id] = edge

    def add_outcoming_edge(self, edge):
        self._outcoming_edges[edge.id] = edge

    def remove_incoming_edge(self, edge_id):
        del self._incoming_edges[edge_id]

    def remove_outcoming_edge(self, edge_id):
        del self._outcoming_edges[edge_id]

class Graph:
    def __init__(self):
        self._nodes = {}
        self._edges = {}
        self._available_node_id = 0
        self._available_edge_id = 0
        self._global_attribute = GraphAttribute()

    def add_node(self, data, type=None):
        node = Node(data, type, self._available_node_id)
        self._nodes[node.id] = node
        self._available_node_id += 1
        return node.id

    def add_edge(self, sender_id, receiver_id, data=None, type=None):
        edge = Edge(self._nodes[sender_id], self._nodes[receiver_id], data, type, self._available_edge_id)
        self._edges[edge.id] = edge
        self._nodes[sender_id].add_outcoming_edge(edge)
        self._nodes[receiver_id].add_incoming_edge(edge)
        self._available_edge_id+=1
        return edge.id

    def remove_edge(self, id):
        # clean sender/receiver first
        self._edges[id].sender.remove_outcoming_edge(id)
        self._edges[id].receiver.remove_incoming_edge(id)
        # now we can safely remove the edge
        del self._edges[id]

    def remove_node(self, id):
        # remove all edges first
        edges_to_remove = [e.id for e in self._edges.values() if e.sender == id or e.receiver == id]
        for idx in edges_to_remove:
            del self._edges[idx]

        # remove node itself
        del self._nodes[id]

    def get_node_by_id(self, id):
        return self._nodes[id]

    def get_edge_by_id(self, id):
        return self._edges[id]

    @property
    def global_attribute(self):
        return self._global_attribute

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    def _check_graph_consistency(self):
        #TODO finish me
        # every edge should connect on
        raise NotImplementedError

    def _get_ajacency_matrix(self):
        # TODO finish me
        raise NotImplementedError

    def _graph_summary(self):
        for n in self._nodes.values():
            print("Node with id: %d, data: %s, incoming edges: %s, outcoming edges: %s" % (n.id, str(n.data), n.incoming_edges, n.outcoming_edges))

        for e in self._edges.values():
            print("Edge with id: %d, data: %s, sender id: %d, receiver id: %d." % (e.id, e.data, e.sender.id, e.receiver.id))

def concat_graphs(graph_list):
    """
    Concatenate features of the graphs with the same topology

    Parameters
    ----------
    graph_list: list with pgn.graph.Graph entries

    Returns
    -------
    A concatenated graph
    """

    if len(graph_list) == 0:
        raise ValueError("Nothing to concatenate. Give me some graphs, man!")

    if len(graph_list) > 1:
        G = copy_graph(graph_list[0])

        # concatenate nodes
        for n in G.nodes:
            # TODO cat or stack here?
            G.nodes[n].data = torch.cat([g.nodes[n].data for g in graph_list])

        # concatenate edges
        for e in G.edges:
            # TODO cat or stack here?
            G.edges[e].data = torch.cat([g.edges[e].data for g in graph_list])
        # concatenate globals
        #TODO add to docs
        #however if we just concatenate the values and node_updater is None, we will always increas the dimensionallity
        #and we will be fucked in the recurremnt case
        G.global_attribute.data = torch.cat([g.global_attribute.data for g in graph_list])

        return G

    return graph_list[0]


def copy_graph(G):
    #TODO check and recheck this
    newG = Graph()

    newG._nodes = {}
    for nid, n in G.nodes.items():
        new_node = Node(n.data.clone(), n.type, n.id)
        newG._nodes[nid] = new_node

    newG._edges = {}
    for eid, e in G.edges.items():
        new_edge = Edge(newG.get_node_by_id(e.sender.id), newG.get_node_by_id(e.receiver.id), e.data.clone(), e.type, e.id)
        newG._nodes[e.sender.id].add_outcoming_edge(new_edge)
        newG._nodes[e.receiver.id].add_incoming_edge(new_edge)
        newG._edges[eid] = new_edge

    ga = G._global_attribute
    newG._global_attribute = GraphAttribute(ga.data.clone(), ga.type, ga.id)

    newG._available_edge_id = G._available_edge_id
    newG._available_node_id = G._available_node_id
    return newG
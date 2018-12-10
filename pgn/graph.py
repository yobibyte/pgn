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

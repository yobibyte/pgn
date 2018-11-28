from collections import defaultdict

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
    def __init__(self, data=None, type=None):
        self._data = data
        self._type = type

class Edge(GraphAttribute):
    def __init__(self, sender, receiver, data=None, type=None):
        self._sender = sender
        self._receiver = receiver
        super().__init__(data, type)

class Node(GraphAttribute):
    def __init__(self, data=None, type=None):
        super().__init__(data, type)

class Graph:
    def __init__(self):
        self._nodes = defaultdict(list)
        self._edges = defaultdict(list)

    def add_node(self, node):
        self._nodes[node.type].append(node)

    def add_edge(self, edge):
        self._edges[edge.type].append(edge)

# TODO indexing nodes/edges via id?
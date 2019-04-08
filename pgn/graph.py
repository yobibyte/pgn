import torch

class Entity(object):
    def __init__(self, id, type):
        self._id = id
        self._type = type

    @property
    def type(self):
        return self._type

    @property
    def id(self):
        return self._id


class DirectedEdge(Entity):
    def __init__(self, id, sender_id, receiver_id, sender_type='vertex', receiver_type='vertex', type='edge'):
        super().__init__(id, type)
        self._sender_id = sender_id
        self._receiver_id = receiver_id
        self._sender_type = sender_type
        self._receiver_type = receiver_type

    @property
    def sender_id(self):
        return self._sender_id

    @property
    def receiver_id(self):
        return self._receiver_id
    
    @property
    def sender_type(self):
        return self._sender_type

    @property
    def receiver_type(self):
        return self._receiver_type


class Vertex(Entity):
    def __init__(self, id, type='vertex'):
        super().__init__(id, type)


class Context(Entity):
    def __init__(self, id, type='context'):
        super().__init__(id, type)


class DirectedGraph(object):
    def __init__(self, entities):
        # entities is a dict, where the key stands for the entity type,
        # and the value is the dict with the 'data' and 'info' which has a list of all the entities where their index
        # in the array corresponds to the index in the fist dimension of the data tensor
        # [{'data': tf.Tensor, 'info': []}, ]

        self._vertices = {el['info'][0].type: el for el in entities if isinstance(el['info'][0], Vertex)}
        self._edges = {el['info'][0].type: el for el in entities if isinstance(el['info'][0], DirectedEdge)}
        for t, v in self._vertices.items():
            if self.num_vertices(t) != len(v['info']):
                raise ValueError("The first dimension of vertices data of type %s is different "
                                 "from the number of edges in the 'info' list" % t)

        for t, v in self._edges.items():
            if self.num_edges(t) != len(v['info']):
                raise ValueError("The first dimension of edges data of type %s is different "
                                 "from the number of edges in the 'info' list" % t)

            # get corresponding vertex type
            valid_receivers = range(self.num_vertices(v['info'][0].sender_type))
            for e in v['info']:
                if e.sender_id not in range(self.num_vertices(v['info'][0].sender_type)):
                    raise ValueError(
                        "Sender %d for edge %d of type %s is invalid. It's either its id is negative or bigger "
                        "then number of your nodes. " % (e.sender_id, e.id, e.type))
                if e.receiver_id not in range(self.num_vertices(v['info'][0].receiver_type)):
                    raise ValueError(
                        "Receiver %d for edge %d of type %s is invalid. It's either its id is negative or bigger "
                        "then number of your nodes. " % (e.receiver_id, e.id, e.type))

    def num_vertices(self, type):
        return self._vertices[type]['data'].shape[0]

    def num_edges(self, type):
        return self._edges[type]['data'].shape[0]

    @property
    def num_edge_types(self):
        return len(self.edge_types)

    @property
    def num_vertex_types(self):
        return len(self.vertex_types)

    def identify_edge_by_sender_and_receiver(self, sender_id, receiver_id, vertex_type=None):
        if vertex_type is None and len(self.vertex_types) > 1:
            raise ValueError("I have more than one vertex type, you need to provide a type to identify an edge")

        if vertex_type is None:
            vertices = next(iter(self._vertices.values()))
        else:
            vertices = self._vertices[vertex_type]

        if sender_id > self.num_edges - 1 or receiver_id > self.num_edges -1:
            return -1

        for eid in range(self.num_edges):
            if vertices[eid].sender_id == sender_id and vertices[eid].receiver_id == receiver_id:
                return eid

        return -1

    def edge_data(self, type=None):
        if type is None and len(self.edge_types) > 1:
            raise ValueError("I have more than one edge type, you need to provide a type to get the data")
        if type is None:
            return next(iter(self._edges.values()))['data']
        else:
            return self._edges[type]['data']

    def vertex_data(self, type=None):
        if type is None and len(self.vertex_types) > 1:
            raise ValueError("I have more than one vertex type, you need to provide a type to get the data")
        if type is None:
            return next(iter(self._vertices.values()))['data']
        else:
            return self._vertices[type]['data']

    @property
    def edge_types(self):
        return list(self._edges.keys())

    @property
    def vertex_types(self):
        return list(self._vertices.keys())

    def senders(self, id=None, edge_type=None):
        edge_type = self.default_edge_type if edge_type is None else edge_type

        if id is None:
            return [el.sender_id for el in self._edges[edge_type]['info']]
        else:
            return self._edges[edge_type]['info'][id].sender_id

    def receivers(self, id=None, edge_type=None):
        edge_type = self.default_edge_type if edge_type is None else edge_type

        if id is None:
            return [el.receiver_id for el in self._edges[edge_type]['info']]
        else:
            return self._edges[edge_type]['info'][id].receiver_id

    def incoming_edges(self, id=None, vertex_type=None):
        vertex_type = self.default_vertex_type if vertex_type is None else vertex_type
        incoming = []

        if id is None:
            for v_id in range(self.num_vertices(vertex_type)):
                curr_v = []
                for k, v in self._edges.items():
                    curr_v.append({k: [el.id for el in v['info'] if el.receiver_id == v_id and el.receiver_type == vertex_type]})
                if self.num_edge_types == 1:
                    curr_v = curr_v[0][self.default_edge_type]
                incoming.append(curr_v)
        else:
            for k, v in self._edges.items():
                incoming.append({k: [el.id for el in v['info'] if el.receiver_id == id and el.receiver_type == vertex_type]})
            if self.num_edge_types == 1:
                return incoming[0][self.default_edge_type]
        return incoming

    def outgoing_edges(self, id=None, vertex_type=None):
        vertex_type = self.default_vertex_type if vertex_type is None else vertex_type
        outgoing = []

        if id is None:
            for v_id in range(self.num_vertices(vertex_type)):
                curr_v = []
                for k, v in self._edges.items():
                    curr_v.append({k: [el.id for el in v['info'] if el.sender_id == v_id and el.sender_type == vertex_type]})
                if self.num_edge_types == 1:
                    curr_v = curr_v[0][self.default_edge_type]
                outgoing.append(curr_v)
        else:
            for k, v in self._edges.items():
                outgoing.append({k: [el.id for el in v['info'] if el.sender_id == id and el.sender_type == vertex_type]})
            if self.num_edge_types == 1:
                return outgoing[0][self.default_edge_type]

        return outgoing


    @property
    def default_vertex_type(self):
        if len(self._vertices.keys()) > 1:
            raise ValueError("I have more than one vertex type, you need to provide a type to get the data")
        return next(iter(self._vertices.keys()))

    @property
    def default_edge_type(self):
        if len(self._edges.keys()) > 1:
            raise ValueError("I have more than one edge type, you need to provide a type to get the data")
        return next(iter(self._edges.keys()))

    def _graph_summary(self):
        print('Graph summary')
        print('-------------')

        for vt in self.vertex_types:
            print("Vertices of type '%s'" % vt)

            for vid, vinfo in enumerate(self._vertices[vt]['info']):
                print("Vertex with id: %d, data: %s, incoming edges: %s, outcoming edges: %s, type: %s." %
                      (vid,
                       str(self._vertices[vt]['data'][vid]),
                       self.incoming_edges(id=vid, vertex_type=vinfo.type),
                       self.outgoing_edges(id=vid, vertex_type=vinfo.type),
                       vinfo.type)
                      )

        for et in self.edge_types:
            print("Edges of type '%s'" % et)
            for eid, einfo in enumerate(self._edges[et]['info']):
                print("Edge with id: %d, data: %s, sender id: %d, receiver id: %d, type: %s." %
                      (eid,
                       str(self._edges[et]['data'][eid]),
                       einfo.sender_id,
                       einfo.receiver_id,
                       einfo.type)
                      )


class DirectedGraphWithContext(DirectedGraph):
    def __init__(self, entities):
        super().__init__(entities)
        self._context = [el for el in entities if isinstance(el, Context)][0]


# def copy_graph_topology(G):
#     nodes_data = torch.zeros_like(G.nodes_data)
#     edges_data = torch.zeros_like(G.edges_data)
#     global_data = torch.zeros_like(G.global_data)
#     return Graph(nodes_data, edges_data, global_data, G.connectivity, G.nodes_types, G.edges_types)
#
# def concat_graphs(graph_list):
#     """
#     Concatenate features of the graphs with the same topology
#
#     Parameters
#     ----------
#     graph_list: list with pgn.graph.Graph entries
#
#     Returns
#     -------
#     A concatenated graph
#     """
#
#     if len(graph_list) == 0:
#         raise ValueError("Nothing to concatenate. Give me some graphs, man!")
#
#     if len(graph_list) == 1:
#         return graph_list[0]
#
#     # TODO check that topology is the same for everyone
#     res = copy_graph_topology(graph_list[0])
#     res._nodes_data = torch.cat([g._nodes_data for g in graph_list], dim=1)
#     res._edges_data = torch.cat([g._edges_data for g in graph_list], dim=1)
#     res._global_data = torch.cat([g._global_data for g in graph_list])
#     return res
#
#
# def copy_graph(G):
#     return Graph(G._nodes_data.clone(),
#                  G._edges_data.clone(),
#                  G._global_data.clone(),
#                  G.connectivity,
#                  G.nodes_types,
#                  G.edges_types)

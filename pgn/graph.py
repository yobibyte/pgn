import torch


class Entity(object):
    def __init__(self, id, type, hidden_info=None):
        self.id = id
        self.type = type
        self.hidden_info = hidden_info

class DirectedEdge(Entity):
    def __init__(self, id, sender, receiver, type='edge', hidden_info=None):
        super().__init__(id, type, hidden_info=hidden_info)
        self.sender = sender
        self.receiver = receiver


class Vertex(Entity):
    def __init__(self, id, type='vertex', hidden_info=None):
        super().__init__(id, type, hidden_info=hidden_info)
        self.incoming_edges = None
        self.outgoing_edges = None

class Context(Entity):
    def __init__(self, id, type='context', hidden_info=None):
        super().__init__(id, type, hidden_info=hidden_info)

class Graph(object):
    def __init__(self):
        '''Metadata stores any information (in tensors) we might use to help with using the output of the graph net. For instance, valid actions vectors.'''
        self._metadata = {}

class DirectedGraph(Graph):
    def __init__(self, entities, safemode=False):
        super().__init__()
        # entities is a dict, where the key stands for the entity type,
        # and the value is the dict with the 'data' and 'info' which has a list of all the entities where their index
        # in the array corresponds to the index in the fist dimension of the data tensor
        # [{'data': tf.Tensor, 'info': []}, ]
        self._vertices = {k: v.copy() for k, v in entities['vertex'].items()}
        self._edges = {k: v.copy() for k, v in entities['edge'].items()}

        if 'metadata' in entities:
            self._metadata = entities['medatada'].copy() # not a deep copy and I mean it

        if safemode:
            self.safety_check()

    def safety_check(self):
        for t, v in self._vertices.items():
            for vid, el in enumerate(v['info']):
                if vid != el.id:
                    raise ValueError(
                        "The vertex with index {} should be a {}-th element in the vertex array of type {}".format(
                            el.id, el.id, t))

            if self.num_vertices(t) != len(v['info']):
                raise ValueError("The first dimension of vertices data of type %s is different "
                                 "from the number of edges in the 'info' list" % t)

        for t, v in self._edges.items():
            for eid, el in enumerate(v['info']):
                if eid != el.id:
                    raise ValueError(
                        "The edge with index {} should be a {}-th element in the edge array of type {}".format(el.id,
                                                                                                               el.id,
                                                                                                               t))

            if self.num_edges(t) != len(v['info']):
                raise ValueError("The first dimension of edges data of type %s is different "
                                 "from the number of edges in the 'info' list" % t)

            # get corresponding vertex type
            for e in v['info']:
                if e.sender.id not in range(self.num_vertices(v['info'][0].sender.type)):
                    raise ValueError(
                        "Sender %d for edge %d of type %s is invalid. It's either its id is negative or bigger "
                        "then number of your nodes. " % (e.sender.id, e.id, e.type))
                if e.receiver.id not in range(self.num_vertices(v['info'][0].receiver.type)):
                    raise ValueError(
                        "Receiver %d for edge %d of type %s is invalid. It's either its id is negative or bigger "
                        "then number of your nodes. " % (e.receiver.id, e.id, e.type))


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

    def identify_edge_by_sender_and_receiver(self, sender_id, receiver_id, edge_type=None):
        for etype, einfo in self._edges.items():
            if sender_id > self.num_edges(etype) - 1 or receiver_id > self.num_edges(etype) - 1:
                continue
            for e in einfo['info']:
                if e.sender.id == sender_id and e.receiver.id == receiver_id:
                    return e
        return -1

    def vertex_data(self, type=None):
        if type is None:
            return {k: v['data'] for k, v in self._vertices.items()}
        else:
            return self._vertices[type]['data']

    def edge_data(self, type=None):
        if type is None:
            return {k: v['data'] for k, v in self._edges.items()}
        else:
            return self._edges[type]['data']

    def set_edge_data(self, data, type=None):
        if not isinstance(data, dict):
            if type is None:
                type = self.default_edge_type
            self._edges[type]['data'] = data
        else:
            for type, d in data.items():
                self._edges[type]['data'] = d

    def set_vertex_data(self, data, type=None):
        if not isinstance(data, dict):
            if type is None:
                type = self.default_vertex_type
            self._vertices[type]['data'] = data
        else:
            for type, d in data.items():
                self._vertices[type]['data'] = d

    def vertex_info(self, type=None):
        if type is None:
            return {k: v['info'] for k, v in self._vertices.items()}
        else:
            return self._vertices[type]['info']

    def edge_info(self, type=None):
        if type is None:
            return {k: v['info'] for k, v in self._edges.items()}
        else:
            return self._edges[type]['info']

    @property
    def edge_types(self):
        return list(self._edges.keys())

    @property
    def vertex_types(self):
        return list(self._vertices.keys())

    def senders(self, id=None, edge_type=None):
        edge_type = self.default_edge_type if edge_type is None else edge_type

        if id is None:
            return [el.sender for el in self._edges[edge_type]['info']]
        else:
            return self._edges[edge_type]['info'][id].sender

    def receivers(self, id=None, edge_type=None):
        edge_type = self.default_edge_type if edge_type is None else edge_type

        if id is None:
            return [el.receiver for el in self._edges[edge_type]['info']]
        else:
            return self._edges[edge_type]['info'][id].receiver

    def incoming_edges(self, vid, vertex_type, edge_type, ids_only=False):
        res = self._vertices[vertex_type]['info'][vid].incoming_edges[edge_type]
        if ids_only:
            return [el.id for el in res]
        return res

    def outgoing_edges(self, vid, vertex_type, edge_type, ids_only=False):
        res = self._vertices[vertex_type]['info'][vid].outgoing_edges[edge_type]
        if ids_only:
            return [el.id for el in res]
        return res

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
                print("Vertex with id: %d, data: %s, incoming edges ids: %s, outcoming edges ids: %s, type: %s." %
                      (vid,
                       str(self._vertices[vt]['data'][vid]),
                       [el.id for el in self.incoming_edges(vid, vinfo.type, 'edge')],
                       [el.id for el in self.outgoing_edges(vid, vinfo.type, 'edge')],
                       vinfo.type)
                      )

        for et in self.edge_types:
            print("Edges of type '%s'" % et)
            for eid, einfo in enumerate(self._edges[et]['info']):
                print("Edge with id: %d, data: %s, sender id: %d, receiver id: %d, type: %s." %
                      (eid,
                       str(self._edges[et]['data'][eid]),
                       einfo.sender.id,
                       einfo.receiver.id,
                       einfo.type)
                      )

    def get_entities(self, zero_data=False):
        entities = {'vertex': {}, 'edge': {}}

        for vtype, vdata in self._vertices.items():
            vdict = {'info': vdata['info']}
            if zero_data:
                vdict['data'] = None
            else:
                vdict['data'] = vdata['data']
            entities['vertex'][vtype] = vdict

        for etype, edata in self._edges.items():
            edict = {'info': edata['info']}
            if zero_data:
                edict['data'] = None
            else:
                edict['data'] = edata['data']
            entities['edge'][etype] = edict
        return entities

    def get_graph_with_same_topology(self):
        return self.__class__(self.get_entities(zero_data=True))

    @classmethod
    def concat(cls, graph_list, topology=None):
        """
        Concatenate features of the directed graphs with the same topology
        Parameters
        ----------
        graph_list: list with pgn.graph.Graph entries

        Returns
        -------
        A concatenated graph
        """

        assert all([el.__class__ == cls for el in graph_list])

        if len(graph_list) == 0:
            raise ValueError("Nothing to concatenate. Give me some graphs, man!")

        if len(graph_list) == 1:
            return graph_list[0]

        # TODO check that topology is the same for everyone
        # TODO it's wasteful to copy zeros every type, just get the stub for the data with Nones
        if topology is None:
            res = graph_list[0].get_graph_with_same_topology()
        else:
            res = topology

        for t in res.vertex_types:
            res._vertices[t]['data'] = torch.cat([g._vertices[t]['data'] for g in graph_list], dim=1)

        for t in res.edge_types:
            res._edges[t]['data'] = torch.cat([g._edges[t]['data'] for g in graph_list], dim=1)

        return res

    def to(self, device):
        for v in self._vertices.values():
            v['data'] = v['data'].to(device)

        for e in self._edges.values():
            e['data'] = e['data'].to(device)

    def set_data(self, vertex_data=None, edge_data=None):
        self.set_vertex_data(vertex_data)
        self.set_edge_data(edge_data)

    def get_data(self):
        return self.vertex_data(), self.edge_data()

class DirectedGraphWithContext(DirectedGraph):
    def __init__(self, entities):
        super().__init__(entities)
        self._context = {k:v.copy() for k,v in entities['context'].items()}

    @property
    def context(self):
        return self._context

    @classmethod
    def concat(cls, graph_list, topology=None):
        # TODO reuse the code from the parent class as much as possible
        # this is just a temporary (huh) copypaste from the parent class

        assert all([el.__class__ == cls for el in graph_list])

        if len(graph_list) == 0:
            raise ValueError("Nothing to concatenate. Give me some graphs, man!")

        if len(graph_list) == 1:
            return graph_list[0]

        # TODO check that topology is the same for everyone
        # TODO it's wasteful to copy zeros every type, just get the stub for the data with Nones
        if topology is None:
            res = graph_list[0].get_graph_with_same_topology()
        else:
            res = topology

        for t in res.vertex_types:
            res._vertices[t]['data'] = torch.cat([g._vertices[t]['data'] for g in graph_list], dim=1)

        for t in res.edge_types:
            res._edges[t]['data'] = torch.cat([g._edges[t]['data'] for g in graph_list], dim=1)

        for t in res._context:
            res._context[t]['data'] = torch.cat([g._context[t]['data'] for g in graph_list], dim=1)

        return res

    def context_data(self, type=None, concat=False):
        if type is None:
            if concat:
                # TODO this was never tested with several types of context variables
                return torch.cat([v['data'] for v in self._context.values()]).squeeze()
            else:
                return {k: v['data'] for k, v in self._context.items()}
        else:
            return self._context[type]['data']

    def set_context_data(self, data, type=None):
        if not isinstance(data, dict):
            if type is None:
                type = self.default_context_type
            self._context[type]['data'] = data
        else:
            for type, d in data.items():
                self._context[type]['data'] = d

    def get_entities(self, zero_data=False):
        entities = super().get_entities(zero_data)
        entities['context'] = {}
        for ctype, cdata in self._context.items():
            entities['context'][ctype] = {
                                'data': cdata['data'] if not zero_data else None,
                                'info': cdata['info']
                               }
        return entities

    @property
    def context_types(self):
        return self._context.keys()

    @property
    def default_context_type(self):
        if len(self._context.keys()) > 1:
            raise ValueError("I have more than one vertex type, you need to provide a type to get the data")
        return next(iter(self._context.keys()))

    def to(self, device):
        super().to(device)

        for c in self._context.values():
            c['data'] = c['data'].to(device)

    def set_data(self, vertex_data=None, edge_data=None, context_data=None):
        super().set_data(vertex_data, edge_data)
        self.set_context_data(context_data)

    def get_data(self):
        vdata, edata = super().get_data()
        return vdata, edata, self.context_data()
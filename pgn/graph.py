from copy import deepcopy

import torch


class Entity(object):
    def __init__(self, id, type, hidden_info=None):
        self._id = id
        self._type = type
        self._hidden_info = hidden_info

    @property
    def type(self):
        return self._type

    @property
    def id(self):
        return self._id

    @property
    def hidden_info(self):
        return self._hidden_info


class DirectedEdge(Entity):
    def __init__(self, id, sender, receiver, type='edge', hidden_info=None):
        super().__init__(id, type, hidden_info=hidden_info)
        self._sender = sender
        self._receiver = receiver

    @property
    def sender(self):
        return self._sender

    @property
    def receiver(self):
        return self._receiver


class Vertex(Entity):
    def __init__(self, id, type='vertex', hidden_info=None):
        super().__init__(id, type, hidden_info=hidden_info)


class Context(Entity):
    def __init__(self, id, type='context', hidden_info=None):
        super().__init__(id, type, hidden_info=hidden_info)


class DirectedGraph(object):
    def __init__(self, entities):

        # entities is a dict, where the key stands for the entity type,
        # and the value is the dict with the 'data' and 'info' which has a list of all the entities where their index
        # in the array corresponds to the index in the fist dimension of the data tensor
        # [{'data': tf.Tensor, 'info': []}, ]

        self._vertices = {el['info'][0].type: el for el in entities if isinstance(el['info'][0], Vertex)}
        self._edges = {el['info'][0].type: el for el in entities if isinstance(el['info'][0], DirectedEdge)}

        for t, v in self._vertices.items():
            if not isinstance(v['data'], torch.Tensor):
                v['data'] = torch.Tensor(v['data'])

            for vid, el in enumerate(v['info']):
                if vid != el.id:
                    raise ValueError(
                        "The vertex with index {} should be a {}-th element in the vertex array of type {}".format(
                            el.id, el.id, t))

            if self.num_vertices(t) != len(v['info']):
                raise ValueError("The first dimension of vertices data of type %s is different "
                                 "from the number of edges in the 'info' list" % t)

        for t, v in self._edges.items():
            if not isinstance(v['data'], torch.Tensor):
                v['data'] = torch.Tensor(v['data'])

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

    def incoming_edges(self, id=None, vertex_type=None):
        #TODO implement 'all' for vertex_type and make it reserved key for the dict
        vertex_type = self.default_vertex_type if vertex_type is None else vertex_type

        if id is None:
            # list of dicts of lists here
            incoming = []
            for v_id in range(self.num_vertices(vertex_type)):
                curr_v = {}
                for k, v in self._edges.items():
                    curr_v[k] = [el for el in v['info'] if el.receiver.id == v_id and el.receiver.type == vertex_type]
                incoming.append(curr_v)
        else:
            # dict of lists here
            incoming = {}
            for k, v in self._edges.items():
                incoming[k] = [el for el in v['info'] if el.receiver.id == id and el.receiver.type == vertex_type]
        return incoming

    def outgoing_edges(self, id=None, vertex_type=None):
        vertex_type = self.default_vertex_type if vertex_type is None else vertex_type

        if id is None:
            outgoing = []
            for v_id in range(self.num_vertices(vertex_type)):
                curr_v = {}
                for k, v in self._edges.items():
                    curr_v[k] = [el for el in v['info'] if el.sender.id == v_id and el.sender.type == vertex_type]
                outgoing.append(curr_v)
        else:
            outgoing = {}
            for k, v in self._edges.items():
                outgoing[k] = [el for el in v['info'] if el.sender.id == id and el.sender.type == vertex_type]
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
                print("Vertex with id: %d, data: %s, incoming edges ids: %s, outcoming edges ids: %s, type: %s." %
                      (vid,
                       str(self._vertices[vt]['data'][vid]),
                       [el.id for el in self.incoming_edges(id=vid, vertex_type=vinfo.type)['edge']],
                       [el.id for el in self.outgoing_edges(id=vid, vertex_type=vinfo.type)['edge']],
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
        entities = []
        for vtype, vdata in self._vertices.items():
            vdict = {'info': deepcopy(vdata['info'])}
            if zero_data:
                vdict['data'] = torch.zeros_like(vdata['data'])
            else:
                vdict['data'] = vdata['data'].clone()
            entities.append(vdict)
        for etype, edata in self._edges.items():
            edict = {'info': deepcopy(edata['info'])}
            if zero_data:
                edict['data'] = torch.zeros_like(edata['data'])
            else:
                edict['data'] = edata['data'].clone()
            entities.append(edict)
        return entities

    def get_copy(self):
        return self.__class__(self.get_entities(zero_data=False))

    def get_graph_with_same_topology(self):
        return self.__class__(self.get_entities(zero_data=True))

    @classmethod
    def concat(cls, graph_list):
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
        res = graph_list[0].get_graph_with_same_topology()

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

class DirectedGraphWithContext(DirectedGraph):
    def __init__(self, entities):
        super().__init__(entities)
        self._context = {el['info'][0].type: el for el in entities if isinstance(el['info'][0], Context)}

    @property
    def context(self):
        return self._context

    @classmethod
    def concat(cls, graph_list):
        # TODO reuse the code from the parent class as much as possible
        # this is just a temporary (huh) copypaste from the parent class

        assert all([el.__class__ == cls for el in graph_list])

        if len(graph_list) == 0:
            raise ValueError("Nothing to concatenate. Give me some graphs, man!")

        if len(graph_list) == 1:
            return graph_list[0]

        # TODO check that topology is the same for everyone
        # TODO it's wasteful to copy zeros every type, just get the stub for the data with Nones
        res = graph_list[0].get_graph_with_same_topology()

        for t in res.vertex_types:
            res._vertices[t]['data'] = torch.cat([g._vertices[t]['data'] for g in graph_list], dim=1)

        for t in res.edge_types:
            res._edges[t]['data'] = torch.cat([g._edges[t]['data'] for g in graph_list], dim=1)

        for t in res._context:
            res._context[t]['data'] = torch.cat([g._context[t]['data'] for g in graph_list])

        return res

    def context_data(self, type=None, concat=False):
        if type is None:
            if concat:
                return torch.cat([v['data'] for k, v in self._context.items()])
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
        for cdata in self._context.values():
            entities.append({
                'data': cdata['data'].clone(),
                'info': deepcopy(cdata['info'])
            })
        return entities

    @property
    def default_context_type(self):
        if len(self._context.keys()) > 1:
            raise ValueError("I have more than one vertex type, you need to provide a type to get the data")
        return next(iter(self._context.keys()))

    def to(self, device):
        super().to(device)

        for c in self._context.values():
            c['data'] = c['data'].to(device)

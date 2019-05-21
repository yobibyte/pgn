import torch

"""Stores all of the graphs providing convenient API for data handling

I use the term 'entities' quite extensively here. First usage is described in the Entity class.
Secondly, when I say a dict of entities, I mean the following:
entities = {'vertex':
                {
                  'vertex_type_1': {'info': [list of objects of class Entity, entity.id equals its index in the list]
                                   'data': torch.Tensor #data, first dim corresponds to entity.id in the info list},
                  'vertex_type_2: {...},
                },
            'edge':
                {
                  'edge_type_1': {'info': [...], 'data': [...]},
                  'edge_type_2': {'info': [...], 'data': [...]},
                }
            'metadata': {...}
            }

We might have a batched version of the above. Then it is a list of dicts shown above.
"""


class Entity(object):
    """Basic container for graph entities, i.e. edges, vertices, global attributes

    This used to have @properties instead of fields, but this was quite slow and I switched back to fields.
    """

    def __init__(self, id, type, hidden_info=None):
        """

        Parameters
        ----------
        id: int
            Id of an entity enables us to locate its data in the data tensor:
            ent_data = graph._vertex_data[self.type][self.id]
        type: str
            Type of an entity.
        hidden_info: dict
            A container to store metadata which might be useful for some semantical
            properties which the library does not care about. For instance, you might want to do something with some of
            the vertices of the output graph, but you still want to process them as a whole and not introduce a second
            vertex type for that.
        """
        self.id = id
        self.type = type
        self.hidden_info = hidden_info


class DirectedEdge(Entity):
    """A directed edge sender -> receiver"""

    def __init__(self, id, sender, receiver, type='edge', hidden_info=None):
        super().__init__(id, type, hidden_info=hidden_info)
        self.sender = sender
        self.receiver = receiver


class Vertex(Entity):
    def __init__(self, id, type='vertex', hidden_info=None):
        """Vertex entity

        We double the information we store in incoming_edges/outgoing_edges (senders/receivers for edges) since we
        do not want to recompute this each time we run the forward pass of a Graph Network.
        """

        super().__init__(id, type, hidden_info=hidden_info)
        self.incoming_edges = None
        self.outgoing_edges = None


class Context(Entity):
    """Graph context. In the Battaglia et al.,2018 paper it is called the global attribute u. I prefer to call it 'c'
    since in multi-agent RL we use 'u' for actions.
    """

    def __init__(self, id, type='context', hidden_info=None):
        super().__init__(id, type, hidden_info=hidden_info)


class Graph(object):
    def __init__(self):
        """Metadata stores any information (in tensors) we might use to help with using the output of the graph net.
        For instance, valid actions vectors."""

        self._metadata = {}


class DirectedGraph(Graph):
    """Directed graph

    Note, this is not a DAG! Loops are possible here. If you need a DAG, you need to add additional checks for cycles.

    #TODO vertices/edges do not differ much. But we have the same methods for them. Probably, we should
    #rewrite the methods in a way the code is not copypasted. I think it is quite easy.

    """

    def __init__(self, entities, safemode=False):
        """Constructor of a directed graph

        Parameters
        ----------

        safemode: bool
            if True, we check the data consistency when constructing the graph.
            Probably important when you are not sure about your data -> entities pipeline.
            Turn of to get the speed-of-light performance.
        """

        super().__init__()
        # entities is a dict, where the key stands for the entity type,
        # and the value is the dict with the 'data' and 'info' which has a list of all the entities where their index
        # in the array corresponds to the index in the fist dimension of the data tensor
        # [{'data': tf.Tensor, 'info': []}, ]
        self._vertices = {k: v.copy() for k, v in entities['vertex'].items()}
        self._edges = {k: v.copy() for k, v in entities['edge'].items()}

        if 'metadata' in entities:
            self._metadata = entities['metadata'].copy()  # not a deep copy and I mean it

        if safemode:
            self.safety_check()

    def safety_check(self):
        """Safety check described in the constructor of the class"""

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
        """How many vertices of 'type' type does the graph have?

        Parameters
        ----------
        type: str
            type of the entities you are interested in

        Returns
        -------
        num: int
            number of the entities of the requested type
        """
        return self._vertices[type]['data'].shape[0]

    def num_edges(self, type):
        """How many edges of 'type' type does the graph have?

        Parameters
        ----------
        type: str
            type of the entities you are interested in

        Returns
        -------
        num: int
            number of the entities of the requested type
        """

        return self._edges[type]['data'].shape[0]

    @property
    def num_edge_types(self):
        """How many edge types do we have?"""
        return len(self.edge_types)

    @property
    def num_vertex_types(self):
        """How many vertex types do we have?"""
        return len(self.vertex_types)

    def vertex_data(self, type=None):
        """Vertex data getter

        Parameters
        ----------
        type: str
            type of the entitity you are interested in.

        Returns
        -------
        data: torch.Tensor or dict
            If type is not None, you'll get a tensor for the type requested.
            Otherwise, you'll get a dict of the following format {'vertex_type_1': data_tensor, ...}
        """
        if type is None:
            return {k: v['data'] for k, v in self._vertices.items()}
        else:
            return self._vertices[type]['data']

    def edge_data(self, type=None):
        """Edge data getter

        Parameters
        ----------
        type: str
            type of the entitity you are interested in.

        Returns
        -------
        data: torch.Tensor or dict
            If type is not None, you'll get a tensor for the type requested.
            Otherwise, you'll get a dict of the following format {'vertex_type_1': data_tensor, ...}
        """
        if type is None:
            return {k: v['data'] for k, v in self._edges.items()}
        else:
            return self._edges[type]['data']

    def set_edge_data(self, data, type=None):
        """Edge data setter

        Parameters
        ----------
        data: torch.Tensor or dict of tensors
        type: str or None

        If data is the tensor, you assume that there is only one data type and it will update it.
        Otherwise, it will update all the datatypes present in the input dict.
        """
        if not isinstance(data, dict):
            if type is None:
                type = self.default_edge_type
            self._edges[type]['data'] = data
        else:
            for type, d in data.items():
                self._edges[type]['data'] = d

    def set_vertex_data(self, data, type=None):
        """Vertex data setter

        Parameters
        ----------
        data: torch.Tensor or dict of tensors
        type: str or None

        If data is the tensor, you assume that there is only one data type and it will update it.
        Otherwise, it will update all the datatypes present in the input dict.
        """
        if not isinstance(data, dict):
            if type is None:
                type = self.default_vertex_type
            self._vertices[type]['data'] = data
        else:
            for type, d in data.items():
                self._vertices[type]['data'] = d

    def vertex_info(self, type=None):
        """Vertex info getter.

        Parameters
        ----------
        type: str or None

        Returns
        -------
        info: list or dict
            If the type is provided, you'll get the list of entities of the type requested.
            Otherwise, you will get a dict of lists for the vertices of all types.
        """
        if type is None:
            return {k: v['info'] for k, v in self._vertices.items()}
        else:
            return self._vertices[type]['info']

    def edge_info(self, type=None):
        """Edge info getter.

        Parameters
        ----------
        type: str or None

        Returns
        -------
        info: list or dict
            If the type is provided, you'll get the list of entities of the type requested.
            Otherwise, you will get a dict of lists for the edges of all types.
        """
        if type is None:
            return {k: v['info'] for k, v in self._edges.items()}
        else:
            return self._edges[type]['info']

    @property
    def edge_types(self):
        """Which edge types does the graph have?"""
        return list(self._edges.keys())

    @property
    def vertex_types(self):
        """Which vertex types does the graph have?"""
        return list(self._vertices.keys())

    def senders(self, id=None, edge_type=None):
        """Get the sender of a particular edge of a particular type

        Parameters
        ----------
        id: int
            edge id
            If id is not provided, get the information for all of the edges of the requested type.
        edge_type: str or None
            type of an edge
            You do not need to provide an edge type if you have one type only.
            Otherwise, not providing a type will lead to an error.

        Returns
        -------
            pgn.graph.Vertex which is the sender of the provided edge
        """
        edge_type = self.default_edge_type if edge_type is None else edge_type

        if id is None:
            return [el.sender for el in self._edges[edge_type]['info']]
        else:
            return self._edges[edge_type]['info'][id].sender

    def receivers(self, id=None, edge_type=None):
        """Get the receiver of a particular edge of a particular type

        Parameters
        ----------
        id: int
            edge id
            If id is not provided, get the information for all of the edges of the requested type.
        edge_type: str or None
            type of an edge
            You do not need to provide an edge type if you have one type only.
            Otherwise, not providing a type will lead to an error.

        Returns
        -------
            pgn.graph.Vertex which is the receiver of the provided edge
        """
        edge_type = self.default_edge_type if edge_type is None else edge_type

        if id is None:
            return [el.receiver for el in self._edges[edge_type]['info']]
        else:
            return self._edges[edge_type]['info'][id].receiver

    def incoming_edges(self, vid, vertex_type, edge_type, ids_only=False):
        """Get all the incoming edges of a vertex

        Parameters
        ----------
        vid: int
            vertex id
        vertex_type: str
            vertex type
        edge_type: str
            edge type
        ids_only: bool
            if True, will return not the list of objects of type Entity, but their ids only.
            Useful for indexing for aggregation, for instance.

        Returns
        -------
            res: list
                based on the ids_only parameter, either list of ints or list of objests of type Entity
        """

        res = self._vertices[vertex_type]['info'][vid].incoming_edges[edge_type]
        if ids_only:
            return [el.id for el in res]
        return res

    def outgoing_edges(self, vid, vertex_type, edge_type, ids_only=False):
        """Get all the outgoing edges of a vertex

        Parameters
        ----------
        vid: int
            vertex id
        vertex_type: str
            vertex type
        edge_type: str
            edge type
        ids_only: bool
            if True, will return not the list of objects of type Entity, but their ids only.
            Useful for indexing for aggregation, for instance.

        Returns
        -------
            res: list
                based on the ids_only parameter, either list of ints or list of objests of type Entity
        """

        res = self._vertices[vertex_type]['info'][vid].outgoing_edges[edge_type]
        if ids_only:
            return [el.id for el in res]
        return res

    @property
    def default_vertex_type(self):
        """Will return the vertex type if the number of vertex types is 1, otherwise will throw an error."""

        if len(self._vertices.keys()) > 1:
            raise ValueError("I have more than one vertex type, you need to provide a type to get the data")
        return next(iter(self._vertices.keys()))

    @property
    def default_edge_type(self):
        """Will return the edge type if the number of edge types is 1, otherwise will throw an error."""

        if len(self._edges.keys()) > 1:
            raise ValueError("I have more than one edge type, you need to provide a type to get the data")
        return next(iter(self._edges.keys()))

    def _graph_summary(self):
        """Prints summary of a graph. Perfect for debugging purposes. Probably should be moved to the utils."""

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
        """Get dict of entities (see top of the current file for a description of the data structure

        Parameters
        ----------
        zero_data: bool
            If true, all the data is set to None, otherwise you get the data WITHOUT COPY. You get the real data!

        Returns
        -------
            Dict of entities with the data or not.
        """

        entities = {'vertex': {}, 'edge': {}, 'metadata': self._metadata}

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
        """Copies the graph topology, but does copy the data."""

        return self.__class__(self.get_entities(zero_data=True))

    @classmethod
    def concat(cls, graph_list, topology=None):
        """Concatenate features of the directed graphs with the same topology

        Parameters
        ----------
        graph_list: list
            list of graph entities dicts, clearly, all the graphs should have the same topology to get concatenated.
        topology: pgn.graph.Graph
            take the 0-th element topology as a template, otherwise take the provided one.

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
        """Move data tensors to a device

        Parameters
        ----------
        device: str
            device identifier, i.e. cuda:0

        Returns
        -------

        """
        for v in self._vertices.values():
            v['data'] = v['data'].to(device)

        for e in self._edges.values():
            e['data'] = e['data'].to(device)

    def set_data(self, vertex_data=None, edge_data=None):
        """ replace graph data

        Parameters
        ----------
        vertex_data: tensor or dict of tensors
            replace vertex data with the provided ones
        edge_data: tensor or dict of tensors
            replace edge data with the provided ones

        Returns
        -------

        """

        self.set_vertex_data(vertex_data)
        self.set_edge_data(edge_data)

    def get_data(self):
        """Get all entities data.

        Returns
        -------
        res: tuple
            (vertex_data, edge_data)
        """
        return self.vertex_data(), self.edge_data()


class DirectedGraphWithContext(DirectedGraph):
    """Same as the DirectedGraph but with the context (global) attribute."""

    def __init__(self, entities):
        """

        Parameters
        ----------
        entities: dict
            entities dict, the description of the data structure is given in the top of the file
        """

        super().__init__(entities)
        self._context = {k: v.copy() for k, v in entities['context'].items()}

    @property
    def context(self):
        """Get the context

        Returns
        -------
        res: dict
            {'context_type#1': {'info': [], 'data': torch.Tensor},
             'context_type#2': {'info': [], 'data': torch.Tensor},
            }
        """

        return self._context

    @classmethod
    def concat(cls, graph_list, topology=None):
        """Concate the graphs. Same as with parent class, but concats contexts as well."""
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
        """Get the context data

        Parameters
        ----------
        type: str
            type of the context data to return
        concat: bool
            if True, will concat all the context type to one tensor

        Returns
        -------
        res: torch.Tensor or dict of tensors
            context data of the requested type or all of them
        """
        if type is None:
            if concat:
                # TODO this was never tested with several types of context variables
                return torch.cat([v['data'] for v in self._context.values()]).squeeze()
            else:
                return {k: v['data'] for k, v in self._context.items()}
        else:
            return self._context[type]['data']

    def set_context_data(self, data, type=None):
        """context data setter

        Parameters
        ----------
        data: torch.Tensor or dict of tensors
            data to set
        type: str
            type of context data to set

        No need to provide the type if the context type is unique.
        Otherwise, will go through the dicts of the data provided
        and set each of the type with the data provided.

        Returns
        -------

        """
        if not isinstance(data, dict):
            if type is None:
                type = self.default_context_type
            self._context[type]['data'] = data
        else:
            for type, d in data.items():
                self._context[type]['data'] = d

    def get_entities(self, zero_data=False):
        """Get all the entities. Same as in parent class, but also returns the context."""

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
        """How many context types do we have?"""

        return self._context.keys()

    @property
    def default_context_type(self):
        """Return the context type key if it is unique, otherwise throw a ValueError"""

        if len(self._context.keys()) > 1:
            raise ValueError("I have more than one vertex type, you need to provide a type to get the data")
        return next(iter(self._context.keys()))

    def to(self, device):
        """Move data to a device"""

        super().to(device)

        for c in self._context.values():
            c['data'] = c['data'].to(device)

    def set_data(self, vertex_data=None, edge_data=None, context_data=None):
        """Data setter, same as in parent class, but with context"""

        super().set_data(vertex_data, edge_data)
        self.set_context_data(context_data)

    def get_data(self):
        """Data getter, same as in parent class, but with context"""

        vdata, edata = super().get_data()
        return vdata, edata, self.context_data()

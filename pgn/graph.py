import torch

class Graph(object):


    def __init__(self, nodes_data=None, edges_data=None, global_data=None, connectivity=None, nodes_types=None, edges_types=None):
        """

        Parameters
        ----------
        nodes_data torch.Tensor of dimensionality (n_nodes, nodes_feature_dim)
        edges_data torch.Tensor of dimensionality (n_edges, edges_feature_dim, can be None if no edges in the graph
        connectivity list of two-el tuples where 0-th el is the edge sender and 1-th is the edge receiver
        index of an element correspond to its index in the edges_data tensor
        nodes_types: list of types of the nodes
        edges_types: list of types of the edges
        """

        self._nodes_data = nodes_data
        self._edges_data = edges_data
        self.global_data = global_data

        if nodes_types is not None and len(nodes_types) != self.num_nodes:
            raise ValueError(
                "The length of nodes_types list should be the same as the first dimension of nodes_data")

        if self._edges_data is None and connectivity is not None:
            raise ValueError("You specify connectivity for the edges, but provide no data for them")

        if self._edges_data is not None:

            if connectivity is None:
                raise ValueError("All edges should have senders/receivers.")

            if len(connectivity) != self.num_edges:
                raise ValueError(
                    "The length of edges_connectivity list should be the same as the first dimension of edges_data")

            if (edges_types is not None) and (len(edges_types) != self.num_edges):
                raise ValueError(
                    "The length of edges_types list should be the same as the first dimension of edges_data")

            for s, r in connectivity:
                if s not in range(self.num_nodes):
                    raise ValueError(
                        "Sender %d is invalid. It's either its id is negative or bigger then number of your nodes. " % s)
                if r not in range(self.num_nodes):
                    raise ValueError(
                        "Receiver %d is invalid. It's either its id is negative or bigger then number of your nodes. " % r)

            self._edges_info = [{'connectivity': (conn[0], conn[1]),
                                 'type': None if edges_types is None else edges_types[eid]}
                                for eid, conn in enumerate(connectivity)]

            if edges_types is not None:
                for idx, t in enumerate(edges_types):
                    self._edges_info[idx]['type'] = t

        if nodes_data is not None:
            self._nodes_info = [
                {'incoming': [] if edges_data is None else [eid for eid, einfo in enumerate(self._edges_info) if einfo['connectivity'][1] == nid],
                 'outgoing': [] if edges_data is None else [eid for eid, einfo in enumerate(self._edges_info) if einfo['connectivity'][0] == nid],
                 'type': None if nodes_types is None else nodes_types[nid]
                 }
                for nid in range(self.num_nodes)]

    @property
    def nodes_data(self):
        return self._nodes_data

    @nodes_data.setter
    def nodes_data(self, nodes_data=None):
        #TODO check cosistency after modifications everywhere
        self._nodes_data = nodes_data

    @property
    def edges_data(self):
        return self._edges_data

    @edges_data.setter
    def edges_data(self, edges_data=None):
        # TODO check cosistency after modifications everywhere
        self._edges_data = edges_data

    @property
    def global_data(self):
        return self._global_data

    @global_data.setter
    def global_data(self, global_data=None):
        self._global_data = global_data

    @property
    def num_nodes(self):
        return self._nodes_data.shape[0]

    @property
    def num_edges(self):
        return self._edges_data.shape[0]

    @property
    def edges_types(self):
        return [e['type'] for e in self._edges_info]

    @property
    def nodes_types(self):
        return [n['type'] for n in self._nodes_info]

    @property
    def connectivity(self):
        return [(e['connectivity'][0], e['connectivity'][1]) for e in self._edges_info]

    @property
    def senders(self):
        return [e['connectivity'][0] for e in self._edges_info]

    @property
    def receivers(self):
        return [e['connectivity'][1] for e in self._edges_info]

    @property
    def incoming(self):
        return [n['incoming'] for n in self._nodes_info]

    @property
    def outgoing(self):
        return [n['outgoing'] for n in self._nodes_info]

    def identify_edge_by_sender_and_receiver(self, sender_id, receiver_id):
        if sender_id > self.num_edges - 1 or receiver_id > self.num_edges -1:
            return -1

        conn = self.connectivity
        for eid in range(self.num_edges):
            if conn[eid][0] == sender_id and conn[eid][1] == receiver_id:
                return eid

        return -1

    def _graph_summary(self):
        for nid, ninfo in enumerate(self._nodes_info):
            print("Node with id: %d, data: %s, incoming edges: %s, outcoming edges: %s" %
                  (nid, str(self._nodes_data[nid]), ninfo['incoming'], ninfo['outgoing']))

        for eid, einfo in self._edges_info:
            print("Edge with id: %d, data: %s, sender id: %d, receiver id: %d." %
                  (eid, self._edges_data[eid], einfo['connectivity'][0], einfo['connectivity'][1]))

def copy_graph_topology(G):
    nodes_data = torch.zeros_like(G.nodes_data)
    edges_data = torch.zeros_like(G.edges_data)
    global_data = torch.zeros_like(G.global_data)
    return Graph(nodes_data, edges_data, global_data, G.connectivity, G.nodes_types, G.edges_types)

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

    if len(graph_list) == 1:
        return graph_list[0]

    # TODO check that topology is the same for everyone
    res = copy_graph_topology(graph_list[0])
    res._nodes_data = torch.cat([g._nodes_data for g in graph_list], dim=1)
    res._edges_data = torch.cat([g._edges_data for g in graph_list], dim=1)
    res._global_data = torch.cat([g._global_data for g in graph_list])
    return res


def copy_graph(G):
    return Graph(G._nodes_data.clone(),
                 G._edges_data.clone(),
                 G._global_data.clone(),
                 G.connectivity,
                 G.nodes_types,
                 G.edges_types)

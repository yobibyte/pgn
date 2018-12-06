import copy

from pgn.aggregators import MeanAggregator
import torch.nn as nn

class NodeBlock(nn.Module):
    def __init__(self, proj, upd, node_aggregator=None):
        if type(proj) == dict:
            self.node_projectors = proj
        else:
            self.node_projectors = {None: proj}

        if type(upd) == dict:
            self._updaters = upd
        else:
            self._updaters = {None: upd}
        self._node_aggregator = node_aggregator if node_aggregator is not None else MeanAggregator()

    def forward(self, G):
        updated_attrs = {}

        for n in G.nodes:
            incoming_edge_attrs = {self.node_projectors[e.type](e.data) for e in n.incoming_edges}
            # Aggregate edge attributes per node
            aggregated = self._node_aggregator(incoming_edge_attrs)
            # Compute updated node attributes
            updated_attrs[n] = self._updaters(aggregated)

        return updated_attrs


class EdgeBlock(nn.Module):
    def __init__(self, upd):
        if type(upd) == dict:
            self._updaters = upd
        else:
            self._updaters = {None: upd}

    def forward(self, G):
        updated_attrs = {}

        for e in G.edges:
            updated_attrs[e.id] = self._updaters[e.type](
                [e.data, e.receiver.data, e.sender.data, G.global_attribute.data])

        return updated_attrs


class GlobalBlock(nn.Module):
    def __init__(self, node_projectors, edge_projectors, updater, node_aggregator=None, edge_aggregator=None):
        self._node_projectors = node_projectors
        self._edge_projectors = edge_projectors
        self._node_aggregator = node_aggregator if node_aggregator is None else MeanAggregator()
        self._edge_aggregator = edge_aggregator if edge_aggregator is None else MeanAggregator()
        self._updater = updater

    def forward(self, G):
        # Aggregate edge attributes globally
        aggregated_edge_attrs = self._edge_aggregator({self._edge_projectors(e.data) for e in G.edges})

        # Aggregate node attributes globally
        aggregated_node_attrs = self._node_aggregator({self._node_projectors(n.data) for n in G.nodes})

        # Compute updated global attribute
        return self._updater(aggregated_edge_attrs, aggregated_node_attrs, G.global_attribute.data)


class GraphNetwork(nn.Module):
    def __init__(self):
        self._global_block = GlobalBlock()
        self._node_block = NodeBlock()
        self._edge_block = EdgeBlock()

    def forward(self, G):

        G = copy.deepcopy(G)

        # make one pass as in the original paper

        # 1. Compute updated edge attributes
        updated_edges_attrs = self._edge_block(G)
        for e, e_data in updated_edges_attrs.items():
            G.edges[e.id].data = e_data

        # 2. Aggregate edge attributes per node
        # 3. Compute updated node attributes
        upd_node_attrs = self._node_block(G)
        for n, n_data in upd_node_attrs.items():
            G.nodes[n.id].data = n_data

        # 4. Aggregate edge attributes globally
        # 5. Aggregate node attributes globally
        # 6. Compute updated global attribute
        G.global_attribute.data = self._global_block(G)

        return G


#TODO implement batching
#TODO None: projector/updater/aggregator by default, make access automatic
#TODO should aggregators accept values? Or they might need receiver/sender indices? Not sure. Probably not
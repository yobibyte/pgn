import copy

from pgn.aggregators import MeanAggregator

import torch
import torch.nn as nn

class NodeBlock(nn.Module):
    def __init__(self, updater, node_aggregator=None):
        super().__init__()
        self._updater = updater
        self._node_aggregator = node_aggregator if node_aggregator is not None else MeanAggregator()

    def forward(self, G):
        updated_attrs = {}

        for nid, n in G.nodes.items():
            incoming_edge_attrs = torch.stack([e.data for e in n.incoming_edges.values()])
            # Aggregate edge attributes per node
            aggregated = self._node_aggregator(incoming_edge_attrs)
            # Compute updated node attributes
            updated_attrs[nid] = self._updater(torch.cat([aggregated, n.data, G.global_attribute.data]))
        return updated_attrs


class EdgeBlock(nn.Module):
    def __init__(self, updater):
        super().__init__()
        self._updater = updater

    def forward(self, G):
        updated_attrs = {}

        for e in G.edges.values():
            # TODO torch concat along axis 0? or 1?
            updated_attrs[e.id] = self._updater(torch.cat([e.data, e.receiver.data, e.sender.data, G.global_attribute.data]))

        return updated_attrs


class GlobalBlock(nn.Module):
    def __init__(self, updater, node_aggregator=None, edge_aggregator=None):
        super().__init__()
        self._node_aggregator = node_aggregator if node_aggregator is not None else MeanAggregator()
        self._edge_aggregator = edge_aggregator if edge_aggregator is not None else MeanAggregator()
        self._updater = updater

    def forward(self, G):
        # Aggregate edge attributes globally
        aggregated_edge_attrs = self._edge_aggregator([e.data for e in G.edges.values()])

        # Aggregate node attributes globally
        aggregated_node_attrs = self._node_aggregator([n.data for n in G.nodes.values()])

        # Compute updated global attribute
        return self._updater(torch.cat([aggregated_edge_attrs, aggregated_node_attrs, G.global_attribute.data]))


class GraphNetwork(nn.Module):
    def __init__(self, node_block, edge_block, global_block):
        super().__init__()
        self._node_block = node_block
        self._edge_block = edge_block
        self._global_block = global_block

    def forward(self, G):

        G = copy.deepcopy(G)

        # make one pass as in the original paper

        # 1. Compute updated edge attributes
        updated_edges_attrs = self._edge_block(G)
        for e, e_data in updated_edges_attrs.items():
            G.edges[e].data = e_data

        # 2. Aggregate edge attributes per node
        # 3. Compute updated node attributes
        upd_node_attrs = self._node_block(G)
        for n, n_data in upd_node_attrs.items():
            G.nodes[n].data = n_data

        # # 4. Aggregate edge attributes globally
        # # 5. Aggregate node attributes globally
        # # 6. Compute updated global attribute
        G.global_attribute.data = self._global_block(G)

        return G


#TODO implement batching
#TODO should aggregators accept values? Or they might need receiver/sender indices? Not sure. Probably not
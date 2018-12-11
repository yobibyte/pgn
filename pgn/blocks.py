import copy

from pgn.aggregators import MeanAggregator

import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, independent):
        super().__init__()
        self._independent = independent


class NodeBlock(Block):
    def __init__(self, updater=None, node_aggregator=None, independent=False):
        super().__init__(independent)
        self._updater = updater

        if independent:
            if node_aggregator is not None:
                raise ValueError("`independent` is set to False, but you're passing an aggregator. Is something wrong?")
        else:
            self._node_aggregator = MeanAggregator() if node_aggregator is None else node_aggregator


    def forward(self, G):
        if self._independent:
            to_updater = [n.data for n in G.nodes.values()]
        else:
            to_updater = []
            for n in G.nodes.values():
                # Aggregate edge attributes per node
                aggregated = self._node_aggregator(torch.stack([e.data for e in n.incoming_edges.values()]))
                # Compute updated node attributes
                to_updater.append(torch.cat([aggregated, n.data, G.global_attribute.data]))

        if self._updater is None:
            updater_output = to_updater
        else:
            updater_output = self._updater(torch.stack(to_updater))

        return {nid: out for nid, out in zip(G.nodes, updater_output)}


class EdgeBlock(Block):
    def __init__(self, updater=None, independent=False):
        super().__init__(independent)
        self._updater = updater

    def forward(self, G):
        if self._independent:
            updater_input = [e.data for e in G.edges.values()]
        else:
            # TODO torch concat along axis 0? or 1?
            updater_input = [torch.cat([e.data, e.receiver.data, e.sender.data, G.global_attribute.data]) for e in G.edges.values()]

        if self._updater is None:
            updater_output = updater_input
        else:
            updater_output = self._updater(torch.stack(updater_input))

        return {nid: out for nid, out in zip(G.edges, updater_output)}


class GlobalBlock(Block):
    def __init__(self, updater=None, node_aggregator=None, edge_aggregator=None, independent=False):
        super().__init__(independent)

        if independent:
            if node_aggregator is not None or edge_aggregator is not None:
                raise ValueError(
                    "`independent` is set to True, but you're passing an aggregator. Is something wrong?")
        else:
            self._node_aggregator = MeanAggregator() if node_aggregator is None else node_aggregator
            self._edge_aggregator = MeanAggregator() if edge_aggregator is None else edge_aggregator

        self._updater = updater

    def forward(self, G):

        upd_input = [G.global_attribute.data]

        if not self._independent:
            # Aggregate edge attributes globally
            aggregated_edge_attrs = self._edge_aggregator([e.data for e in G.edges.values()])
            upd_input.append(aggregated_edge_attrs)

            # Aggregate node attributes globally
            aggregated_node_attrs = self._node_aggregator([n.data for n in G.nodes.values()])
            upd_input.append(aggregated_node_attrs)

        # Compute updated global attribute
        if self._updater is None:
            updater_output = upd_input
        else:
            updater_output = self._updater(torch.cat(upd_input))

        return updater_output


class GraphNetwork(nn.Module):
    def __init__(self, node_block=None, edge_block=None, global_block=None):
        super().__init__()
        self._node_block = node_block
        self._edge_block = edge_block
        self._global_block = global_block

    def forward(self, G):
        #G = copy.deepcopy(G)

        # make one pass as in the original paper

        # 1. Compute updated edge attributes
        if self._edge_block is not None:
            updated_edges_attrs = self._edge_block(G)
            for e, e_data in updated_edges_attrs.items():
                G.edges[e].data = e_data

        # 2. Aggregate edge attributes per node
        # 3. Compute updated node attributes
        if self._node_block is not None:
            upd_node_attrs = self._node_block(G)
            for n, n_data in upd_node_attrs.items():
                G.nodes[n].data = n_data

        # # 4. Aggregate edge attributes globally
        # # 5. Aggregate node attributes globally
        # # 6. Compute updated global attribute
        if self._global_block is not None:
            G.global_attribute.data = self._global_block(G)

        return G
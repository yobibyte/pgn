from pgn.aggregators import MeanAggregator

import torch
import torch.nn as nn

from pgn.graph import copy_graph

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
            to_updater = G.nodes_data
        else:
            agg_input = [G.edges_data[G.incoming[nid]] for nid in range(G.num_nodes)]
            aggregated = self._node_aggregator(agg_input)
            to_updater = torch.stack([torch.cat([aggregated[nid], G.nodes_data[nid], G.global_data]) for nid in range(G.num_nodes)])

        if self._updater is None:
            return to_updater
        else:
            return self._updater(to_updater)

class EdgeBlock(Block):
    def __init__(self, updater=None, independent=False):
        super().__init__(independent)
        self._updater = updater

    def forward(self, G):

        if self._independent:
            updater_input = G.edges_data
        else:
            # TODO torch concat along axis 0? or 1?
            # TODO  pad till largest here

            inpt = [torch.cat([G.edges_data[e], G.nodes_data[G.receivers[e]], G.nodes_data[G.senders[e]], G.global_data]) for e in range(G.num_edges)]
            updater_input = torch.stack(inpt)

        if self._updater is None:
            return updater_input
        else:
            return self._updater(updater_input)


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
        if self._updater is None:
            return G.global_data

        upd_input = [G.global_data]
        if not self._independent:
            # Aggregate edge attributes globally
            upd_input.append(self._edge_aggregator([G.edges_data]).squeeze())

            # Aggregate node attributes globally
            upd_input.append(self._node_aggregator([G.nodes_data]).squeeze())

        return self._updater(torch.cat(upd_input))


class GraphNetwork(nn.Module):
    def __init__(self, node_block=None, edge_block=None, global_block=None):
        super().__init__()
        self._node_block = node_block
        self._edge_block = edge_block
        self._global_block = global_block

    def forward(self, G, modify_input=False):
        if not modify_input:
            G = copy_graph(G)

        # make one pass as in the original paper

        # 1. Compute updated edge attributes
        if self._edge_block is not None:
            G.edges_data = self._edge_block(G)

        # 2. Aggregate edge attributes per node
        # 3. Compute updated node attributes
        if self._node_block is not None:
            G.nodes_data = self._node_block(G)

        # # 4. Aggregate edge attributes globally
        # # 5. Aggregate node attributes globally
        # # 6. Compute updated global attribute
        if self._global_block is not None:
            G.global_data = self._global_block(G)

        return G
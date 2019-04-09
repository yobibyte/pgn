from pgn.aggregators import MeanAggregator
import pgn.graph as pg

import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, independent):
        super().__init__()
        self._independent = independent

    @property
    def independent(self):
        return self._independent


class NodeBlock(Block):
    def __init__(self, updaters=None, e2n_aggregators=None):
        super().__init__(e2n_aggregators is None)
        self._updaters = updaters
        self._e2n_aggregators = e2n_aggregators

    def forward(self, G):
        out = {}
        vdata = G.vertex_data
        edata = G.edges_data

        for t in vdata:
            if self.independent:
                to_updater = vdata[t]
            else:
                aggregated = {}
                # TODO rewrite if the order of concat matters
                for at in self._e2n_aggregators:
                    agg_input = [edata[t][G.incoming(nid, t)] for nid in range(G.num_vertices[t])]
                    aggregated[at] = self._e2n_aggregator(agg_input)

                    # TODO the dims should be [node, aggregated features]
                    aggregated = torch.cat([el for el in aggregated.values()], dim=1)
                if isinstance(G, pg.DirectedGraphWithContext):
                    torch.stack([torch.cat([aggregated[nid], vdata[t][nid], G.context]) for nid in range(G.num_vertices(t))])
                else:
                    to_updater = torch.stack([torch.cat([aggregated[nid], vdata[t][nid]]) for nid in range(G.num_vertices(t))])

            if t not in self._updaters:
                out[t] = to_updater[t]
            else:
                out[t] = self._updaters[t](to_updater[t])

        return out

# TODO do refactoring with the below code

class EdgeBlock(Block):
    def __init__(self, updaters=None, independent=False):
        super().__init__(independent)
        self._updaters = updaters

    def forward(self, G):
        out = {}
        for et, edata in G.edges_data:

            if self._independent:
                updater_input = G.edges_data
            else:
                # TODO torch concat along axis 0? or 1?
                # TODO  pad till largest here
                senders_types = []
                receivers_types =
                if isinstance(G, pg.DirectedGraphWithContext):

                    inpt = [torch.cat([edata[e],
                                       G.vertex_data[][G.receivers[e]],
                                       G.nodes_data[G.senders[e]],
                                       G.context]) for e in range(G.num_edges(et))]
                else:
                    inpt = [torch.cat(
                        [edata[e], G.vertex_data[G.receivers[e]], G.nodes_data[G.senders[e]]]) for e
                            in range(G.num_edges(et))]
                updater_input = torch.stack(inpt)

            if et not in self._updaters:
                out[et] = updater_input
            else:
                out[et] = self._updater(updater_input)


class GlobalBlock(Block):
    def __init__(self, updater=None, node_aggregator=None, edge_aggregator=None, independence_mode=IndependenceMode.DEPENDENT):
        super().__init__(independence_mode)

        if self.independent
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
        if not self.independent:
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
            G = G.get_copy()

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
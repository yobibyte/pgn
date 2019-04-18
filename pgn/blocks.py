import pgn.graph as pg

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, independent):
        super().__init__()
        self._independent = independent
        self._params_registered = False

    @property
    def independent(self):
        return self._independent


class NodeBlock(Block):
    def __init__(self, updaters=None, in_e2n_aggregators=None, out_e2n_aggregators=None):
        super().__init__(in_e2n_aggregators is None and out_e2n_aggregators is None)
        self._updaters = nn.ModuleDict(updaters)
        self._in_e2n_aggregators = in_e2n_aggregators
        self._out_e2n_aggregators = out_e2n_aggregators

    def forward(self, G):
        out = {}
        vdata = G.vertex_data()
        edata = G.edge_data()

        for t in vdata:
            if self.independent:
                to_updater = vdata[t]
            else:
                in_aggregated = []
                if self._in_e2n_aggregators is not None:
                    # TODO rewrite if the order of concat matters
                    for at in self._in_e2n_aggregators:
                        agg_input = [edata[at][list(map(lambda x: x.id, G.incoming_edges(nid, t)[at]))] for nid in range(G.num_vertices(t))]
                        in_aggregated.append(self._in_e2n_aggregators[at](agg_input))

                out_aggregated = []
                if self._out_e2n_aggregators is not None:
                    for at in self._out_e2n_aggregators:
                        agg_input = [edata[at][list(map(lambda x: x.id, G.outgoing_edges(nid, t)[at]))] for nid in range(G.num_vertices(t))]
                        out_aggregated.append(self._out_e2n_aggregators[at](agg_input))

                # TODO the dims should be [node, aggregated features], check this thoroughly
                aggregated = torch.cat(in_aggregated + out_aggregated, dim=1)
                
                if isinstance(G, pg.DirectedGraphWithContext):
                    to_updater = torch.stack(
                        [torch.cat([aggregated[nid], vdata[t][nid], G.context_data(concat=True)]) for nid in
                         range(G.num_vertices(t))])
                else:
                    to_updater = torch.stack(
                        [torch.cat([aggregated[nid], vdata[t][nid]]) for nid in range(G.num_vertices(t))])

            if t not in self._updaters:
                out[t] = to_updater
            else:
                out[t] = self._updaters[t](to_updater)

        return out

    def to(self, device):
        for el in self._updaters.values():
            el.to(device)


class EdgeBlock(Block):
    def __init__(self, updaters=None, independent=False):
        super().__init__(independent)
        self._updaters = nn.ModuleDict(updaters)

    def forward(self, G):
        out = {}
        vertex_data = G.vertex_data()
        for et, edata in G.edge_data().items():
            einfo = G.edge_info(et)
            if self._independent:
                updater_input = edata
            else:
                # TODO torch concat along axis 0? or 1?
                # TODO  pad till largest here
                if isinstance(G, pg.DirectedGraphWithContext):
                    inpt = [torch.cat([edata[e],
                                       vertex_data[einfo[e].receiver.type][einfo[e].receiver.id],
                                       vertex_data[einfo[e].sender.type][einfo[e].sender.id],
                                       G.context_data(concat=True)]) for e in range(G.num_edges(et))]
                else:
                    inpt = [torch.cat([edata[e],
                                       vertex_data[einfo[e].receiver.type][einfo[e].receiver.id],
                                       vertex_data[einfo[e].sender.type][einfo[e].sender.id],
                                       ]) for e in range(G.num_edges(et))]
                updater_input = torch.stack(inpt)

            if et not in self._updaters:
                out[et] = updater_input
            else:
                out[et] = self._updaters[et](updater_input)
        return out

    def to(self, device):
        for el in self._updaters.values():
            el.to(device)


class GlobalBlock(Block):
    def __init__(self, updaters=None, vertex_aggregators=None, edge_aggregators=None):
        super().__init__(vertex_aggregators is None or edge_aggregators is None)

        if (vertex_aggregators is None != edge_aggregators):
            raise NotImplementedError("Vertex aggregators should both be None (independent case) or not None. "
                                      "There is no implementation for other cases")

        self._vertex_aggregators = vertex_aggregators
        self._edge_aggregators = edge_aggregators
        self._updaters = nn.ModuleDict(updaters)
        # TODO Implement outgoing aggregators for the release

    def forward(self, G):
        out = {}
        for t, cdata in G.context_data().items():
            upd_input = [cdata]
            if not self.independent:
                for vtype, vdata in G.vertex_data().items():
                    # Aggregate vertex attributes globally
                    upd_input.append(self._vertex_aggregators[vtype]([vdata]).squeeze())

                for etype, edata in G.edge_data().items():
                    # Aggregate edge attributes globally
                    upd_input.append(self._edge_aggregators[etype]([edata]).squeeze())
            upd_input = torch.cat(upd_input)
            if t not in self._updaters:
                out[t] = upd_input
            else:
                out[t] = self._updaters[t](upd_input)
        return out
    
    def to(self, device):
        for el in self._updaters.values():
            el.to(device)


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
            G.set_edge_data(self._edge_block(G))

        # 2. Aggregate edge attributes per node
        # 3. Compute updated node attributes

        if self._node_block is not None:
            G.set_vertex_data(self._node_block(G))

        # 4. Aggregate edge attributes globally
        # 5. Aggregate node attributes globally
        # 6. Compute updated global attribute
        if self._global_block is not None:
            G.set_context_data(self._global_block(G))
        return G

    def to(self, device):
        if self._node_block is not None:
            self._node_block.to(device)
        if self._edge_block is not None:
            self._edge_block.to(device)
        if self._global_block is not None:
            self._global_block.to(device)

import pgn.graph as pg

import torch
import torch.nn as nn
import numpy as np

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

    def forward(self, Gs):
        if isinstance(Gs, pg.Graph):
            Gs = [Gs]
        out = [{} for _ in range(len(Gs))]
        vdata = [g.vertex_data() for g in Gs]

        if self._independent:
            for vt in vdata[0]:
                vtin = torch.stack([el[vt] for el in vdata])
                vtout = self._updaters[vt](vtin)
                for el_idx, el in enumerate(vtout):
                    out[el_idx][vt] = el
        else:
            edata = [g.edge_data() for g in Gs]
            cdata = [g.context_data(concat=True) for g in Gs] if isinstance(Gs[0], pg.DirectedGraphWithContext) else None

            for vt in vdata[0]:
                # TODO we can move aggregation outside of this loop and aggregate for all of the vertices first, and just access it further

                in_aggregated = []
                if self._in_e2n_aggregators is not None:
                    for at in self._in_e2n_aggregators:
                        agg_input = []
                        for g_idx, g in enumerate(Gs):
                            if self._in_e2n_aggregators is not None:
                                idx = [g.incoming_edges(nid, vt, at, ids_only=True) for nid in range(g.num_vertices(vt))]
                                flat = [item for sublist in idx for item in sublist]
                                agg_input.append(edata[g_idx][at][flat].split([len(el) for el in idx]))
                        in_aggregated.append(self._in_e2n_aggregators[at](agg_input))

                out_aggregated = []
                if self._out_e2n_aggregators is not None:
                    for at in self._out_e2n_aggregators:
                        out_agg_input = []
                        for g_idx, g in enumerate(Gs):
                            if self._out_e2n_aggregators is not None:
                                idx = [g.outgoing_edges(nid, vt, at, ids_only= True) for nid in
                                       range(g.num_vertices(vt))]
                                flat = [item for sublist in idx for item in sublist]
                                out_agg_input.append(edata[g_idx][at][flat].split([len(el) for el in idx]))
                        out_aggregated.append(self._out_e2n_aggregators[at](agg_input))

                aggregated = torch.cat(in_aggregated + out_aggregated, dim=2)

                # output for all the graphs
                if cdata is not None:
                    curr_cdata = [el.repeat(vdata[el_id][vt].shape[0], 1) for el_id, el in enumerate(cdata)]
                    curr_cdata = torch.stack(curr_cdata)
                    vtin = torch.cat((aggregated, torch.stack([el[vt] for el in vdata]), curr_cdata), dim=2)
                else:
                    vtin = torch.cat((aggregated, torch.stack([el[vt] for el in vdata])), dim=2)

                vtout = self._updaters[vt](vtin)
                for el_idx, el in enumerate(vtout):
                    out[el_idx][vt] = el
                    # TODO the dims should be [graph, node, aggregated features], check this thoroughly
        return out


class EdgeBlock(Block):
    def __init__(self, updaters=None, independent=False):
        super().__init__(independent)
        self._updaters = nn.ModuleDict(updaters)

    def forward(self, Gs):
        # I assume that graphs are homogenious here, i.e. they have the same types of entities,
        # but that makes sense since the model depends on the graph entities types

        if isinstance(Gs, pg.Graph):
            Gs = [Gs]
        out = [{} for _ in Gs]

        if self._independent:
            for et in Gs[0].edge_data():
                edata = torch.stack([g.edge_data(et) for g in Gs])
                gout = self._updaters[et](edata)
                for el_idx, el in enumerate(gout):
                    out[el_idx][et] = el
        else:
            vdata = [g.vertex_data() for g in Gs]
            cdata = [g.context_data(concat=True) for g in Gs] if isinstance(Gs[0],
                                                                            pg.DirectedGraphWithContext) else None

            for et in Gs[0].edge_data():

                n_edges = [g.num_edges(et) for g in Gs]
                einfo = [g.edge_info(et) for g in Gs]
                edata = torch.stack([g.edge_data(et) for g in Gs])

                if Gs[0].num_vertex_types == 1:
                    vtype = einfo[0][0].receiver.type

                    #megavertexdata = torch.stack([el[vtype] for el in vdata])
                    sender_ids = np.array([[einfo[el_id][e].sender.id for e in range(el)] for el_id, el in enumerate(n_edges)])
                    receiver_ids = np.array([[einfo[el_id][e].sender.id for e in range(el)] for el_id, el in enumerate(n_edges)])

                    sender_data = []
                    receiver_data = []
                    for el_id, el in enumerate(sender_ids):
                          sender_data.append(vdata[el_id][vtype][el])
                    for el_id, el in enumerate(receiver_ids):
                          receiver_data.append(vdata[el_id][vtype][el])
                    sender_data = torch.stack(sender_data)
                    receiver_data = torch.stack(receiver_data)
                else:
                    pass
                    # TODO torch concat along axis 0? or 1?
                    # TODO pad till largest here
                    raise NotImplementedError
                    # receiver_data = torch.stack(
                    #     [vertex_data[einfo[e].receiver.type][einfo[e].receiver.id] for e in range(n_edges)])
                    # sender_data = torch.stack([vertex_data[einfo[e].sender.type][einfo[e].sender.id] for e in
                    #                              range(n_edges)])

                if cdata is not None:
                    curr_cdata = [el.repeat(edata[el_id].shape[0], 1) for el_id, el in enumerate(cdata)]
                    curr_cdata = torch.stack(curr_cdata)
                    etin = torch.cat((edata, sender_data, receiver_data, curr_cdata), dim=2)
                else:
                    etin = torch.cat((edata, sender_data, receiver_data), dim=2)

                etout = self._updaters[et](etin)
                for el_idx, el in enumerate(etout):
                    out[el_idx][et] = el

        return out


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

    def forward(self, Gs):
        if isinstance(Gs, pg.Graph):
            Gs = [Gs]

        out = [{} for _ in range(len(Gs))]
        cdata = [g.context_data() for g in Gs]

        for t in cdata[0]:
            tin = torch.stack([el[t] for el in cdata])
            if self._independent:
                tout = self._updaters[t](tin)
                for el_idx, el in enumerate(tout):
                    out[el_idx][t] = el
            else:
                uin = []
                for vt, agg in self._vertex_aggregators.items():
                    uin.append(agg([[g.vertex_data(vt)] for g in Gs]))
                for et, agg in self._edge_aggregators.items():
                    uin.append(agg([[g.edge_data(et)] for g in Gs]))
                tin = torch.cat((tin, *uin), dim=2)
                tout = self._updaters[t](tin)
                for el_idx, el in enumerate(tout):
                    out[el_idx][t] = el
        return out
    

class GraphNetwork(nn.Module):
    def __init__(self, node_block=None, edge_block=None, global_block=None):
        super().__init__()
        self._node_block = node_block
        self._edge_block = edge_block
        self._global_block = global_block

    def forward(self, Gs):
        # make one pass as in the original paper
        # 1. Compute updated edge attributes
        if self._edge_block is not None:
            edge_outs = self._edge_block(Gs)
            for i, G in enumerate(Gs):
                G.set_edge_data(edge_outs[i])

        # 2. Aggregate edge attributes per node
        # 3. Compute updated node attributes
        if self._node_block is not None:
            v_outs = self._node_block(Gs)
            for i, G in enumerate(Gs):
                G.set_vertex_data(v_outs[i])

        if self._global_block is not None:
            # 4. Aggregate edge attributes globally
            # 5. Aggregate node attributes globally
            # 6. Compute updated global attribute
            g_outs = self._global_block(Gs)
            for i, G in enumerate(Gs):
                G.set_context_data(g_outs[i])
        return [G.get_entities() for G in Gs]

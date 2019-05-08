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

        updater_input_list = []

        for g in Gs:
            updater_input = {}
            vertex_data = g.vertex_data()

            cdata = g.context_data(concat=True) if isinstance(g, pg.DirectedGraphWithContext) else None
            for et, edata in g.edge_data().items():
                n_edges = g.num_edges(et)
                einfo = g.edge_info(et)
                if self._independent:
                    updater_input[et] = edata
                else:
                    if g.num_vertex_types == 1:
                        vtype = einfo[0].receiver.type
                        sender_ids = [einfo[e].sender.id for e in range(n_edges)]
                        receiver_ids = [einfo[e].receiver.id for e in range(n_edges)]
                        receiver_data = vertex_data[vtype][receiver_ids]
                        sender_data = vertex_data[vtype][sender_ids]
                    else:
                        # TODO torch concat along axis 0? or 1?
                        # TODO pad till largest here
                        receiver_data = torch.stack(
                            [vertex_data[einfo[e].receiver.type][einfo[e].receiver.id] for e in range(n_edges)])
                        sender_data = torch.stack([vertex_data[einfo[e].sender.type][einfo[e].sender.id] for e in
                                                     range(n_edges)])
                    if isinstance(g, pg.DirectedGraphWithContext):
                        cdatarepeat = cdata.repeat(n_edges, 1)

                        updater_input[et] = torch.cat((edata, sender_data, receiver_data, cdatarepeat), dim=1)
                    else:
                        updater_input[et] = torch.cat((edata, sender_data, receiver_data), dim=1)
            updater_input_list.append(updater_input)

        out = [{} for _ in range(len(Gs))]
        for et in Gs[0].edge_types:
            if et not in self._updaters:
                for inpt_idx, inpt in enumerate(updater_input_list):
                    out[inpt_idx][et] = inpt[et]
            else:
                # glue all the inputs for the same type
                all_inpt = torch.cat([el[et] for el in updater_input_list])
                # we need these to split after we get the output of a batch
                input_idx = [el[et].shape[0] for el in updater_input_list]
                all_output = self._updaters[et](all_inpt)
                for out_idx, el in enumerate(all_output.split(input_idx)):
                    out[out_idx][et] = el
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

        updater_input_list = []
        for g in Gs:
            updater_input = {}

            for t, cdata in g.context_data().items():
                updater_input[t] = [cdata]
                if not self.independent:
                    for vtype, vdata in g.vertex_data().items():
                        # Aggregate vertex attributes globally
                        updater_input[t].append(self._vertex_aggregators[vtype]([[vdata]])[0])

                    for etype, edata in g.edge_data().items():
                        # Aggregate edge attributes globally
                        updater_input[t].append(self._edge_aggregators[etype]([[edata]])[0])
                updater_input[t] = torch.cat(updater_input[t], dim=1)
            updater_input_list.append(updater_input)

        out = [{} for _ in range(len(Gs))]
        for ct in Gs[0].context_types:
            if ct not in self._updaters:
                for inpt_idx, inpt in enumerate(updater_input_list):
                    out[inpt_idx][ct] = inpt[ct]
            else:
                # glue all the inputs for the same type
                all_inpt = torch.cat([el[ct] for el in updater_input_list])
                # we need these to split after we get the output of a batch
                input_idx = [el[ct].shape[0] for el in updater_input_list]
                all_outputs = self._updaters[ct](all_inpt)
                for out_idx, el in enumerate(all_outputs.split(input_idx)):
                    out[out_idx][ct] = el
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

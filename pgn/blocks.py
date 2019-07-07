import numpy as np
import pgn.graph as pg
import torch
import torch.nn as nn

class Block(nn.Module):
    """A building block of my graph network kingdom.

    A block is a bundle of an updater and a bunch of aggregators to prepare data for the updater. Updater is the king.
    Each entity type has its own bundle. Updater takes the aggregated information from all of the types,
    but updates only its own.
    """

    def __init__(self, independent):
        """

        Parameters
        ----------
        independent: bool
            If the block is independent, the network takes only the particular entity tensor and updates it.
            No other entities data tensors affected.
            G' gets its data updated after all the blocks has finished their job.
        """
        super().__init__()
        self._independent = independent
        self._params_registered = False

    @property
    def independent(self):
        """Is the block independent?"""

        return self._independent


class NodeBlock(Block):
    """Node block"""

    def __init__(self, updaters=None, in_e2n_aggregators=None):
        """

        Parameters
        ----------
        updaters: dict
            dict of updaters, key is the data type to update, value is the updater itself
        in_e2n_aggregators:
            dict of aggregators, key is the data type, value is the pgn.aggregators.Aggregator object.

        There were outgoing edge aggregators here before
        (they also appeared at the DeepMind implementation at some point).
        When I did my 'need for speed' profiling, I decided to get rid of them for
        now and mirror an outgoing edge with an incoming one. This turned out to be faster for me.
        But I do not have any empirical evidence for which approach is better for some particular problem.

        """
        super().__init__(in_e2n_aggregators is None)
        self._updaters = nn.ModuleDict(updaters)  # this is needed to correctly register model parameters
        self._in_e2n_aggregators = in_e2n_aggregators

    def forward(self, Gs):
        """ make a forward pass for node entities

        Updaters work in a batched mode (take all the data tensor and feed to the updater),
        aggregators do their job in a for loop, one aggregator call per graph.

        Parameters
        ----------
        Gs: pgn.graph.Graph or list of them
            input data

        Returns
        -------
            list of dicts with data tensors
        """

        if isinstance(Gs, pg.Graph):
            Gs = [Gs]
        out = [{} for _ in Gs]
        vdata = [g.vertex_data() for g in Gs]

        if self._independent:
            for vt in vdata[0]:
                vtin = torch.stack([el[vt] for el in vdata])
                vtout = self._updaters[vt](vtin)
                for el_idx, el in enumerate(vtout):
                    out[el_idx][vt] = el
        else:
            edata = [g.edge_data() for g in Gs]
            cdata = [g.context_data(concat=True) for g in Gs] if isinstance(Gs[0],
                                                                            pg.DirectedGraphWithContext) else None
            for vt in vdata[0]:
                # TODO we can move aggregation outside of this loop and aggregate for all of the vertices first, and just access it further

                aggregated = []
                if self._in_e2n_aggregators is not None:
                    for at in self._in_e2n_aggregators:
                        stacked_edata = torch.stack([el[at] for el in edata])
                        receiver_ids = torch.tensor(
                            [g._receiver_ids[at] for g in Gs], requires_grad=False, device=edata[0][at].device).unsqueeze(-1).expand(-1,-1, *stacked_edata.shape[2:],)
                        agg = self._in_e2n_aggregators[at](stacked_edata, receiver_ids, dim_size=vdata[0][vt].shape[0], dim=1)
                        aggregated.append(agg)

                aggregated = torch.cat(aggregated, dim=2)

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
    """Edge block, nothing much to say"""

    def __init__(self, updaters=None, independent=False):
        """

        Parameters
        ----------
        updaters: dict
            one updater per type
        independent: bool
            independent or not
        """
        super().__init__(independent)
        self._updaters = nn.ModuleDict(updaters)  # this is needed to correctly register model parameters

    def forward(self, Gs):
        """ make a forward pass for node entities

        Updaters work in a batched mode (take all the data tensor and feed to the updater),
        aggregators do their job in a for loop, one aggregator call per graph.

        scatter operations (in the aggregators) were of great help for me and let me get rid of a lot
        of splits which slowed down the backward step a LOT.

        I assume that graphs are homogeneous here, i.e. they have the same types of entities,
        but that makes sense since the model depends on the graph entities types

        Parameters
        ----------
        Gs: pgn.graph.Graph or list of them
            input data

        Returns
        -------
            list of dicts with data tensors
        """

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
            vdata = torch.stack([g.vertex_data(g.default_vertex_type) for g in Gs])
            cdata = [g.context_data(concat=True) for g in Gs] if isinstance(Gs[0],
                                                                            pg.DirectedGraphWithContext) else None

            for et in Gs[0].edge_data():

                edata = torch.stack([g.edge_data(et) for g in Gs])

                if Gs[0].num_vertex_types == 1:
                    sender_ids = torch.tensor(
                        [g._sender_ids[et] for g in Gs], requires_grad=False, device=vdata.device).unsqueeze(-1).expand(-1,-1, *vdata.shape[2:])
                    receiver_ids = torch.tensor(
                        [g._receiver_ids[et] for g in Gs], requires_grad=False, device=vdata.device).unsqueeze(-1).expand(-1,-1, *vdata.shape[2:],)
                    sender_data = vdata.gather(1, sender_ids)
                    receiver_data = vdata.gather(1, receiver_ids)
                else:
                    raise NotImplementedError("Current implementation supports one vertex type only")

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
    """Context block

    #TODO rename this to context
    """

    def __init__(self, updaters=None, vertex_aggregators=None, edge_aggregators=None):
        """

        Parameters
        ----------
        updaters: dict
            one updater per type {type1: updater, type2: updater}
        vertex_aggregators: dict
            one aggregator per type, if an aggregator is absent for a type, it will not be aggregated
        edge_aggregators: dict
            one aggregator per type, if an aggregator is absent for a type, it will not be aggregated

        """

        super().__init__(vertex_aggregators is None or edge_aggregators is None)

        if (vertex_aggregators is None != edge_aggregators):
            raise NotImplementedError("Vertex aggregators should both be None (independent case) or not None. "
                                      "There is no implementation for other cases")

        self._vertex_aggregators = vertex_aggregators
        self._edge_aggregators = edge_aggregators
        self._updaters = nn.ModuleDict(updaters)
        # TODO Implement outgoing aggregators for the release

    def forward(self, Gs):
        """ make a forward pass for node entities

        Updaters work in a batched mode (take all the data tensor and feed to the updater),
        aggregators do their job in a for loop, one aggregator call per graph.

        scatter operations (in the aggregators) were of great help for me and let me get rid of a lot
        of splits which slowed down the backward step a LOT.

        Parameters
        ----------
        Gs: pgn.graph.Graph or list of them
            input data

        Returns
        -------
            list of dicts with data tensors
        """


        if isinstance(Gs, pg.Graph):
            Gs = [Gs]

        out = [{} for _ in Gs]
        cdata = [g.context_data() for g in Gs]

        if not self._independent:
            uin = []
            for vt, agg in self._vertex_aggregators.items():
                uin.append(torch.stack([agg(g.vertex_data(vt)) for g in Gs]))
            for et, agg in self._edge_aggregators.items():
                uin.append(torch.stack([agg(g.edge_data(et)) for g in Gs]))
        for t in cdata[0]:
            tin = torch.stack([el[t] for el in cdata])
            if self._independent:
                tout = self._updaters[t](tin)
                for el_idx, el in enumerate(tout):
                    out[el_idx][t] = el
            else:
                tin = torch.cat((tin, *uin), dim=2)
                tout = self._updaters[t](tin)
                for el_idx, el in enumerate(tout):
                    out[el_idx][t] = el
        return out


class GraphNetwork(nn.Module):
    """A default Graph Network as defined in Battaglia et al., 2018"""

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


class IndependentGraphNetwork(nn.Module):
    """An independent graph network, where each updater updates its own entity/type only,
    and all the G' data are update after all of the blocks finish their job"""

    def __init__(self, node_block=None, edge_block=None, global_block=None):
        super().__init__()
        self._node_block = node_block
        self._edge_block = edge_block
        self._global_block = global_block

    def forward(self, Gs):
        # make one pass as in the original paper
        # 1. Compute updated edge attributes

        edge_outs = None
        v_outs = None
        g_outs = None

        if self._edge_block is not None:
            edge_outs = self._edge_block(Gs)

        # 2. Aggregate edge attributes per node
        # 3. Compute updated node attributes
        if self._node_block is not None:
            v_outs = self._node_block(Gs)

        if self._global_block is not None:
            # 4. Aggregate edge attributes globally
            # 5. Aggregate node attributes globally
            # 6. Compute updated global attribute
            g_outs = self._global_block(Gs)

        for i, G in enumerate(Gs):
            if edge_outs is not None:
                G.set_edge_data(edge_outs[i])
            if v_outs is not None:
                G.set_vertex_data(v_outs[i])
            if g_outs is not None:
                G.set_context_data(g_outs[i])

        return [G.get_entities() for G in Gs]

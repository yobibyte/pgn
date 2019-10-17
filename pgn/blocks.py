import torch
import torch.nn as nn


class Block(nn.Module):
    """A building block of my graph network kingdom.

    A block is a bundle of an updater and a bunch of aggregators to prepare data for the updater. Updater is the king.
    Each entity type has its own bundle. Updater takes the aggregated information from all of the types,
    but updates only its own.
    """

    def __init__(self, independent, device):
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
        self._device = device

    @property
    def independent(self):
        """Is the block independent?"""

        return self._independent


class NodeBlock(Block):
    """Node block"""

    def __init__(self, updater, in_e2n_aggregators={}, device=torch.device("cpu")):
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
        super().__init__(in_e2n_aggregators == {}, device=device)
        self._updater = updater  # this is needed to correctly register model parameters
        self._in_e2n_aggregators = in_e2n_aggregators

    def forward(self, vdata, edata, connectivity, cdata=None):
        """ make a forward pass for node entities

        Updaters work in a batched mode (take all the data tensor and feed to the updater),
        aggregators do their job in a for loop, one aggregator call per graph.

        Parameters
        ----------
        vdata: vertex data of the input graph
        edata: edge data of the input graph
        connectivity:
            torch.Tensor of shape 2,* where first row is edge senders, and the second row is edge receivers
        cdata: global attribute of the input graph

        Returns
        -------
            output torch.Tensor
        """
        # TODO check all the dims
        if self._independent:
            return self._updater(vdata)
        else:
            aggregated = []
            for at in self._in_e2n_aggregators:
                agg = self._in_e2n_aggregators[at](
                    edata[at], connectivity[at][1, :], dim_size=vdata.shape[0]
                )
                aggregated.append(agg)

            aggregated = torch.cat(aggregated, dim=1)

            # output for all the graphs
            if cdata is not None:
                return self._updater(torch.cat([aggregated, vdata, cdata], dim=1))
            else:
                return self._updater(torch.cat([aggregated, vdata], dim=1))


class EdgeBlock(Block):
    """Edge block, nothing much to say"""

    def __init__(self, updaters=None, independent=False, device=torch.device("cpu")):
        """

        Parameters
        ----------
        updaters: dict
            one updater per type
        independent: bool
            independent or not
        """
        super().__init__(independent, device=device)
        self._updaters = nn.ModuleDict(
            updaters
        )  # this is needed to correctly register model parameters

    def forward(self, vdata, edata, connectivity, cdata):
        """ make a forward pass for node entities

        Updaters work in a batched mode (take all the data tensor and feed to the updater),
        aggregators do their job in a for loop, one aggregator call per graph.

        scatter operations (in the aggregators) were of great help for me and let me get rid of a lot
        of splits which slowed down the backward step a LOT.

        I assume that graphs are homogeneous here, i.e. they have the same types of entities,
        but that makes sense since the model depends on the graph entities types

        Parameters
        ----------
        vdata: vertex data of the input graph
        edata: edge data of the input graph
        connectivity:
            torch.Tensor of shape 2,* where first row is edge senders, and the second row is edge receivers
        cdata: global attribute of the input graph


        Returns
        -------
            list of dicts with data tensors
        """

        if self._independent:
            return {et: self._updaters[et](ed) for et, ed in edata.items()}
        else:
            # TODO implement context cdata = [g.context_data(concat=True) for g in Gs] if isinstance(Gs[0],
            #                                                                pg.DirectedGraphWithContext) else None
            out = {}
            for et in edata:
                senders, receivers = connectivity[et]

                if cdata is None:
                    out[et] = self._updaters[et](
                        torch.cat([edata[et], vdata[senders], vdata[receivers]], dim=1)
                    )
                else:
                    out[et] = self._updaters[et](
                        torch.cat(
                            [edata[et], vdata[senders], vdata[receivers], cdata[et]],
                            dim=1,
                        )
                    )

            return out


class GlobalBlock(Block):
    """Context block

    #TODO rename this to context
    """

    def __init__(
        self,
        updater=None,
        vertex_aggregator=None,
        edge_aggregators={},
        device=torch.device("cpu"),
    ):
        """

        Parameters
        ----------
        updaters: dict
            one updater per type {type1: updater, type2: updater}
        vertex_aggregators: dict
            one aggregator per type, if an aggregator is absent for a type, it will not be aggregated
        edge_aggregators: dict
            one aggregator per type, if an aggregator is absent for a type, it will not be aggregated
        device: torch.device
            Where should we keep the data? 'cpu' by default
        """
        super().__init__(
            (vertex_aggregator is None) and (edge_aggregators == {}), device=device
        )

        if (vertex_aggregator is None) != (edge_aggregators == {}):
            raise NotImplementedError(
                "Vertex aggregators should both be None (independent case) or not None. "
                "There is no implementation for other cases"
            )

        self._vertex_aggregator = vertex_aggregator
        self._edge_aggregators = edge_aggregators
        self._updater = updater
        # TODO Implement outgoing aggregators for the release

    def forward(self, cdata, vdata=None, edata=None, metadata=None):
        """ make a forward pass for node entities

        Updaters work in a batched mode (take all the data tensor and feed to the updater),
        aggregators do their job in a for loop, one aggregator call per graph.

        scatter operations (in the aggregators) were of great help for me and let me get rid of a lot
        of splits which slowed down the backward step a LOT.

        Parameters
        ----------
        cdata: global attribute of the input graph
        vdata: vertex data of the input graph
        edata: edge data of the input graph
        metadata: dict

        Returns
        -------
            output torch.Tensor
        """

        if self._independent:
            return self._updater(cdata)
        else:
            # we need to use scatter aggregator here where index will show the graph id in the batched graph data
            #     def forward(self, x, indices, dim_size=None, dim=0):
            vidx = torch.tensor(
                [[i] * vsize for i, vsize in enumerate(metadata["vsizes"])],
                device=self._device,
            ).flatten()
            tin = [cdata, self._vertex_aggregator(vdata, vidx)]
            # will fail for graphs with no edges/vertices #TODO
            for et in edata:
                eidx = torch.tensor(
                    [[i] * esize for i, esize in enumerate(metadata["esizes"][et])],
                    device=self._device,
                ).flatten()
                tin.append(
                    self._edge_aggregators[et](edata[et], eidx, dim_size=cdata.shape[0])
                )

            tin = torch.cat(tin, dim=1)
            return self._updater(tin)


class GraphNetwork(nn.Module):
    """A default Graph Network as defined in Battaglia et al., 2018"""

    def __init__(self, node_block=None, edge_block=None, global_block=None):
        super().__init__()
        self._node_block = node_block
        self._edge_block = edge_block
        self._global_block = global_block

    def forward(self, vdata, edata, connectivity, context=None, metadata=None):
        # make one pass as in the original paper
        # 1. Compute updated edge attributes
        eout = None
        vout = None
        cout = None
        if self._edge_block is not None:
            cdata = None
            if context is not None:
                # we need the same amount of cdata as esizes here
                cdata = {}
                for et in edata:
                    cdata[et] = torch.cat(
                        [
                            c.expand(metadata["esizes"][et][i], -1)
                            for i, c in enumerate(context)
                        ]
                    )

            eout = self._edge_block(vdata, edata, connectivity, cdata)

        # 2. Aggregate edge attributes per node
        # 3. Compute updated node attributes
        if self._node_block is not None:
            cdata = None
            if context is not None:
                cdata = torch.cat(
                    [c.expand(metadata["vsizes"][i], -1) for i, c in enumerate(context)]
                )
            vout = self._node_block(vdata, eout, connectivity, cdata)

        if self._global_block is not None:
            # 4. Aggregate edge attributes globally
            # 5. Aggregate node attributes globally
            # 6. Compute updated global attribute
            cout = self._global_block(context, vout, eout, metadata)
        return vout, eout, cout


class IndependentGraphNetwork(nn.Module):
    """An independent graph network, where each updater updates its own entity/type only,
    and all the G' data are update after all of the blocks finish their job"""

    def __init__(self, node_block=None, edge_block=None, global_block=None):
        super().__init__()
        self._node_block = node_block
        self._edge_block = edge_block
        self._global_block = global_block

    def forward(self, vdata, edata, connectivity, cdata, metadata):
        # make one pass as in the original paper
        # 1. Compute updated edge attributes
        eout = None
        vout = None
        cout = None

        if self._edge_block is not None:
            eout = self._edge_block(vdata, edata, connectivity, cdata)

        # 2. Aggregate edge attributes per node
        # 3. Compute updated node attributes
        if self._node_block is not None:
            vout = self._node_block(vdata, edata, connectivity, cdata)

        if self._global_block is not None:
            # 4. Aggregate edge attributes globally
            # 5. Aggregate node attributes globally
            # 6. Compute updated global attribute
            cout = self._global_block(cdata)

        return vout, eout, cout

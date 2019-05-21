import pgn.graph as pg
import torch.nn as nn
from pgn.aggregators import MeanAggregator
from pgn.blocks import NodeBlock, EdgeBlock, GlobalBlock, GraphNetwork, IndependentGraphNetwork


def get_mlp(input_size, units, activation=nn.ReLU):
    """Helper to build multilayer perceptrons

    Important! Last layer is activated as well! And there is a LayerNorm after.

    Parameters
    ----------
    input_size: int
        How many input neurons do you need?
    units: tuple
        How many units in hidden layers do you want?
    activation:
        What is the activation to use?

    Returns
    -------
        model: nn.Sequential
            nn.Sequential container of the architecture you requested
    """
    arch = []
    inpt_size = input_size
    for l in units:
        arch.append(nn.Linear(inpt_size, l))
        arch.append(activation())
        inpt_size = l
    arch.append(nn.LayerNorm(inpt_size))
    return nn.Sequential(*arch)


def get_mlp_updaters(input_node_size, output_node_size, input_edge_size, output_edge_size,
                     input_global_size=None, output_global_size=None, independent=False):
    """Helper to get updaters for a graph network given input/output parameters.

    Parameters
    ----------
    input_node_size: int
        Node feature dimensions of the input graph?
    output_node_size: int
        Node feature dimensions of the output graph?
    input_edge_size: int
        Edge feature dimensions of the input graph?
    output_edge_size: int
        Edge feature dimensions of the output graph?
    input_global_size: int
        Global feature dimensions of the input graph? If None, you will not get a Global updater in the output.
    output_global_size: int
        Global feature dimensions of the output graph? If None, you will not get a Global updater in the output.
    independent: bool
        Whether a GraphNetwork is independent or not. See EncoderCoreDecoder docs for more information.
    Returns
    -------
        node_updater, edge_updater, global_updater: nn.Module
            A tuple of updaters. global_updater might be None if you did not provide dimensions for it!
    """

    with_global = False if (input_global_size is None or output_global_size is None) else True

    if not with_global and input_global_size != output_global_size:
        raise ValueError("You specify only one of the following: input_global_size or output_global_size. "
                         "If you want the global, provide the missing, otherwise, remove the unneeded.")

    if independent:
        edge_updater = get_mlp(input_edge_size, [16, output_edge_size])
        node_updater = get_mlp(input_node_size, [16, output_node_size])
        global_updater = get_mlp(input_global_size, [16, output_global_size]) if with_global else None
    else:
        if not with_global:
            input_global_size = output_global_size = 0
        edge_updater = get_mlp(input_edge_size + 2 * input_node_size + input_global_size, [16, output_edge_size])
        node_updater = get_mlp(input_node_size + output_edge_size + input_global_size, [16, output_node_size])
        global_updater = get_mlp(input_global_size + output_edge_size + output_node_size, [16, output_global_size]) \
            if with_global else None
    return node_updater, edge_updater, global_updater


class EncoderCoreDecoder(nn.Module):
    """Encoder-Core-Decoder architecture as in DeepMind GraphNets sorting task.

    Encoder and Decoder are independent graph networks. This means, that the updaters operate only on the input of their
    corresponding entities. Output of the edge updater does NOT influence the vertex updater, for instance. G' data
    is updated only after all the updaters finish their work.

    Core is a recurrent structure which takes a concatenated input (input graph | output of the core at the previous
    iteration)
    """

    def __init__(self, core_steps, enc_vertex_shape, core_vertex_shape, dec_vertex_shape, out_vertex_size,
                 enc_edge_shape, core_edge_shape, dec_edge_shape, out_edge_size,
                 enc_global_shape=(None, None), core_global_shape=(None, None), dec_global_shape=(None, None),
                 out_global_size=None,
                 input_type=pg.DirectedGraphWithContext):
        """
        TODO this should be made easier for the user. It is very easy to mess up the shape and spend half a day on
        figuring out why you need +10 neurons here or there. The first thing we should do, probably is to make user
        enter input dimensions for the encoder only. Given all the other parameters, we should decide on our own,
        how many units are expected by the core etc.

        Parameters
        ----------
        core_steps: int
            Number of core iterations.
        enc_vertex_shape: tuple
            (in_dim, hidden units #1, #2, ...)
        core_vertex_shape: tuple
            (in_dim, hidden units #1, #2, ...)
        dec_vertex_shape: tuple
            (in_dim, hidden units #1, #2, ...)
        out_vertex_size: int
            Dimensionality of the G' vertex features.
        enc_edge_shape: tuple
            (in_dim, hidden units #1, #2, ...)
        core_edge_shape: tuple
            (in_dim, hidden units #1, #2, ...)
        dec_edge_shape: tuple
            (in_dim, hidden units #1, #2, ...)
        out_edge_size: int
            Dimensionality of the G' edge features.
        enc_global_shape: tuple
            (in_dim, hidden units #1, #2, ...)
        core_global_shape: tuple
            (in_dim, hidden units #1, #2, ...)
        dec_global_shape: tuple
            (in_dim, hidden units #1, #2, ...)
        out_global_size: int
            Dimensionality of the G' global attribute.
        input_type: pgn.Graph
            Before input data is fed to the network it is converted to the pgn.Graph class
            to provide api for graph methods. Which graph do you have? pgn.graph.DirectedGraphWithContext by default.
        """

        super().__init__()
        self._input_type = input_type

        self._core_steps = core_steps
        enc_node_updater, enc_edge_updater, enc_global_updater = get_mlp_updaters(*enc_vertex_shape,
                                                                                  *enc_edge_shape,
                                                                                  *enc_global_shape,
                                                                                  independent=True)

        self.encoder = IndependentGraphNetwork(NodeBlock({'vertex': enc_node_updater}),
                                               EdgeBlock({'edge': enc_edge_updater}, independent=True),
                                               GlobalBlock(
                                                   {'context': enc_global_updater}) if enc_global_updater else None)

        core_node_updater, core_edge_updater, core_global_updater = get_mlp_updaters(*core_vertex_shape,
                                                                                     *core_edge_shape,
                                                                                     *core_global_shape,
                                                                                     independent=False)

        self.core = GraphNetwork(NodeBlock({'vertex': core_node_updater}, {'edge': MeanAggregator()}),
                                 EdgeBlock({'edge': core_edge_updater}),
                                 GlobalBlock({'context': core_global_updater}, {'vertex': MeanAggregator()},
                                             {'edge': MeanAggregator()}) if core_global_updater else None)

        dec_node_updater, dec_edge_updater, dec_global_updater = get_mlp_updaters(*dec_vertex_shape,
                                                                                  *dec_edge_shape,
                                                                                  *dec_global_shape,
                                                                                  independent=True)
        self.decoder = IndependentGraphNetwork(
            NodeBlock({'vertex': nn.Sequential(dec_node_updater, nn.Linear(dec_vertex_shape[-1], out_vertex_size))}),
            EdgeBlock({'edge': nn.Sequential(dec_edge_updater, nn.Linear(dec_edge_shape[-1], out_edge_size))},
                      independent=True),
            GlobalBlock({'context': nn.Sequential(dec_global_updater, nn.Linear(dec_global_shape[-1],
                                                                                out_global_size))}) if dec_global_updater else None,
        )

    def forward(self, input_data, output_all_steps=False):
        """Make a forward pass

        Encoder -> k Core iterations -> Decoder (or each decoder pass after each core pass)

        Parameters
        ----------
        input_data: dict
            dict of entities
        output_all_steps: bool
            if True, Decoder is called after each pass of the Core, else only on the last one

        Returns
        -------
            outputs: list
                if output_all_steps is False, it is a list with dict of entities per input graph as elements
                if output_all_steps if True, it is a list of lists, where each item is a time axis,
                sublists are the same as in the line above
        """

        # this data won't change during the core computation
        latents0_data = self.encoder([self._input_type(d) for d in input_data])

        latents_data = [el for el in latents0_data]
        outputs = []
        concat_topo = None

        for s in range(self._core_steps):
            latents0 = [self._input_type(d) for d in latents0_data]
            latents = [self._input_type(d) for d in latents_data]

            if concat_topo is None:
                concat_topo = [lg.get_graph_with_same_topology() for lg in latents0]

            concatenated = [l0.__class__.concat([l0, l], ct) for l0, l, ct in zip(latents0, latents, concat_topo)]
            latents_data = self.core(concatenated)

            if output_all_steps or s + 1 == self._core_steps:
                outputs.append(self.decoder([self._input_type(d) for d in latents_data]))

        if not output_all_steps:
            return outputs[-1]
        return outputs

    def process_batch(self, input_graphs, compute_grad=True):
        """This used to do batching which is moved to the GraphNetwork now. We have only eval->train mode here now"""
        # TODO this should probably go away
        # I think, just doing model.eval() from outside is enough

        if not compute_grad:
            self.encoder.eval()
            self.core.eval()
            self.decoder.eval()

        outs = self(input_graphs, output_all_steps=True)

        if not compute_grad:
            self.encoder.train()
            self.core.train()
            self.decoder.train()

        return outs

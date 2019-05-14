import torch.nn as nn

from pgn.aggregators import MeanAggregator
from pgn.blocks import NodeBlock, EdgeBlock, GlobalBlock, GraphNetwork
import pgn.graph as pg

def get_mlp(input_size, units, activation=nn.ReLU):
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

    def __init__(self, core_steps, enc_vertex_shape, core_vertex_shape, dec_vertex_shape, out_vertex_size,
                 enc_edge_shape, core_edge_shape, dec_edge_shape, out_edge_size,
                 enc_global_shape=(None, None), core_global_shape=(None, None), dec_global_shape=(None, None),
                 out_global_size=None,
                 input_type=pg.DirectedGraphWithContext):
        super().__init__()
        self._input_type = input_type

        self._core_steps = core_steps
        enc_node_updater, enc_edge_updater, enc_global_updater = get_mlp_updaters(*enc_vertex_shape,
                                                                                  *enc_edge_shape,
                                                                                  *enc_global_shape,
                                                                                  independent=True)

        self.encoder = GraphNetwork(NodeBlock({'vertex': enc_node_updater}),
                                    EdgeBlock({'edge': enc_edge_updater}, independent=True),
                                    GlobalBlock({'context': enc_global_updater}) if enc_global_updater else None)

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
        self.decoder = GraphNetwork(
            NodeBlock({'vertex': nn.Sequential(dec_node_updater, nn.Linear(dec_vertex_shape[-1], out_vertex_size))}),
            EdgeBlock({'edge': nn.Sequential(dec_edge_updater, nn.Linear(dec_edge_shape[-1], out_edge_size))},
                      independent=True),
            GlobalBlock({'context': nn.Sequential(dec_global_updater, nn.Linear(dec_global_shape[-1],
                                                                                out_global_size))}) if dec_global_updater else None,
            )

    def forward(self, input_data, output_all_steps=False):

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


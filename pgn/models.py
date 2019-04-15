import torch.nn as nn

def get_mlp(input_size, units, activation=nn.ReLU):
    arch = []
    inpt_size = input_size
    for l in units:
        arch.append(nn.Linear(inpt_size, l))
        arch.append(activation())
        inpt_size = l
    nn.LayerNorm(inpt_size)
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
        global_updater = get_mlp(input_global_size + output_edge_size + output_node_size, [16, output_global_size])  \
            if with_global else None
    return node_updater, edge_updater, global_updater
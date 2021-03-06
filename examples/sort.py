"""
My implementation of sorting with graph networks (GNs) in pytorch.
Original tf implementation here: https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/sort.ipynb
"""

import argparse
import itertools
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pgn.models import EncoderCoreDecoder
from pgn.utils import batch_data

torch.set_num_threads(4)


def graph_data_from_list(input_list):
    """Takes a list with the data, generates a fully connected graph with values of the list as nodes

    Parameters
    ----------
    input_list: list
        list of the numbers to sort

    Returns
    -------
    graph_data: dict
        Dict of entities for the list provided.
    """
    connectivity = torch.tensor(
        [el for el in itertools.product(range(len(input_list)), repeat=2)],
        dtype=torch.long,
    ).t()
    vdata = torch.tensor([[v] for v in input_list])
    edata = torch.zeros(connectivity.shape[1], 1)
    return (vdata, edata, connectivity)


def edge_id_by_sender_and_receiver(connectivity, sid, rid):
    """Get edge id from the information about its sender and its receiver.

    Parameters
    ----------
    metadata: list
        list of pgn.graph.Edge objects
    sid: int
        sender id
    rid: int
        receiver id

    Returns
    -------

    """
    return (connectivity[0, :] == sid).mul(connectivity[1, :] == rid).nonzero().item()


def create_target_data(vdata, edata, connectivity):
    """ Generate target data for training

    Parameters
    ----------
    input_data: list
        list of data to sort

    Returns
    -------
    res: dict
        dict of target graph entities
    """
    # two nodes might have true since they might have similar values
    min_val = vdata.min()

    # [prob_true, prob_false]
    target_vertex_data = torch.Tensor(
        [[1.0, 0.0] if v == min_val else [0.0, 1.0] for v in vdata]
    )

    sorted_ids = vdata.argsort(dim=0).flatten()
    target_edge_data = torch.zeros(edata.shape[0], 2)
    for sidx, sid in enumerate(sorted_ids):
        for ridx, rid in enumerate(sorted_ids):
            eid = edge_id_by_sender_and_receiver(connectivity, sid, rid)
            # we look for exact comparison here since we sort
            if sidx < len(sorted_ids) - 1 and ridx == sidx + 1:
                target_edge_data[eid][0] = 1.0
            else:
                target_edge_data[eid][1] = 1.0

    return target_vertex_data, target_edge_data


def generate_graph_batch(n_examples, sample_length):
    """ generate all of the training data

    Parameters
    ----------
    n_examples: int
        Num of the samples
    sample_length: int
        Length of the samples.
        # TODO we should implement samples of different lens as in the DeepMind example.
    Returns
    -------
    res: tuple
        (input_data, target_data), each of the elements is a list of entities dicts
    """

    input_data = [
        graph_data_from_list(np.random.uniform(size=sample_length))
        for _ in range(n_examples)
    ]
    target_data = [create_target_data(v, e, conn) for v, e, conn in input_data]

    return input_data, target_data


def batch_loss(outs, targets, criterion, batch_size, core_steps):
    """get the loss for the network outputs

    Parameters
    ----------
    outs: list
        list of lists of the graph network output, time is 0-th dimension, batch is 1-th dimension
    targets: list
        list of the graph entities for the expected output
    criterion: torch._Loss object
        loss to use
    Returns
    -------
    loss: float
        Shows how good your mode is.
    """
    loss = 0
    vsize = targets[0].shape[0] // batch_size
    esize = targets[1].shape[0] // batch_size
    for out in outs:
        for i in range(batch_size):
            loss += criterion(
                out[0][i * vsize : (i + 1) * vsize],
                targets[0][i * vsize : (i + 1) * vsize],
            )
        for i in range(batch_size):
            loss += criterion(
                out[1]["default"][i * esize : (i + 1) * esize],
                targets[1][i * esize : (i + 1) * esize],
            )

    return loss / core_steps / batch_size


def plot_test(model, length, cuda=False):

    unsorted = np.random.uniform(size=length)
    test_g = graph_data_from_list(unsorted)

    test_g = list(batch_data([test_g]))
    if cuda and torch.cuda.is_available():
        test_g[0] = test_g[0].to("cuda")
        for k in test_g[1]:
            test_g[1][k] = test_g[1][k].to("cuda")
            test_g[2][k] = test_g[2][k].to("cuda")

    g = model(*test_g)[1]["default"]
    conn = test_g[2]["default"]

    # evaluate and plot
    mx = np.zeros((len(unsorted), len(unsorted)))
    for eid in range(g.shape[0]):
        mx[conn[0, eid].item()][conn[1, eid].item()] = g[eid, 0].item()

    sort_indices = np.argsort(unsorted)
    plt.matshow(mx[sort_indices][:, sort_indices], cmap="viridis")
    plt.grid(False)
    plt.savefig("pgn_sorting_output.png")


def run():
    """Run an experiment and plot the results on a randomly sampled example not seen during training."""

    parser = argparse.ArgumentParser(description="Sorting with graph networks")
    parser.add_argument(
        "--num-train", type=int, default=32, help="number of training examples"
    )
    parser.add_argument(
        "--num-eval", type=int, default=32, help="number of evaluation examples"
    )
    parser.add_argument(
        "--epochs", type=int, default=5000, help="number of training epochs"
    )
    parser.add_argument(
        "--core-steps", type=int, default=10, help="number of core processing steps"
    )
    parser.add_argument(
        "--sample-length",
        type=int,
        default=10,
        help="number of elements in the list to sort",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=100, help="Evaluation/logging frequency"
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Use a GPU if the system has it."
    )
    parser.add_argument("--verbose", action="store_true", help="Print diagnostircs.")
    parser.add_argument(
        "--plot_graph_sample", action="store_true", help="Plot one of the input graphs"
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")

    model = EncoderCoreDecoder(
        args.core_steps,
        enc_vertex_shape=(1, 16),
        core_vertex_shape=(32, 16),
        dec_vertex_shape=(16, 16),
        out_vertex_size=2,
        enc_edge_shape=(1, 16),
        core_edge_shape=(32, 16),
        dec_edge_shape=(16, 16),
        out_edge_size=2,
        device=device,
    )

    optimiser = torch.optim.Adam(lr=0.001, params=model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    train_input, train_target = generate_graph_batch(
        args.num_train, sample_length=args.sample_length
    )

    eval_input, eval_target = generate_graph_batch(
        args.num_train, sample_length=args.sample_length
    )
    train_input = list(batch_data(train_input))
    train_target = [
        torch.cat([el[0] for el in train_target]),
        torch.cat([el[1] for el in train_target]),
    ]
    eval_target = [
        torch.cat([el[0] for el in eval_target]),
        torch.cat([el[1] for el in eval_target]),
    ]
    eval_input = list(batch_data(eval_input))
    if args.cuda and torch.cuda.is_available():
        train_input[0] = train_input[0].to("cuda")
        for k in train_input[1]:
            train_input[1][k] = train_input[1][k].to("cuda")
            train_input[2][k] = train_input[2][k].to("cuda")

        eval_input[0] = eval_input[0].to("cuda")
        for k in eval_input[1]:
            eval_input[1][k] = eval_input[1][k].to("cuda")
            eval_input[2][k] = eval_input[2][k].to("cuda")
        train_target[0] = train_target[0].to("cuda")
        train_target[1] = train_target[1].to("cuda")
        eval_target[0] = eval_target[0].to("cuda")
        eval_target[1] = eval_target[1].to("cuda")
        model.to("cuda")

    for e in range(args.epochs):

        st_time = time.time()
        train_outs = model(*train_input, output_all_steps=True)
        train_loss = batch_loss(
            train_outs, train_target, criterion, args.num_train, args.core_steps
        )
        optimiser.zero_grad()
        train_loss.backward()

        optimiser.step()

        end_time = time.time()
        if args.verbose:
            print("Epoch {} is done. {:.2f} sec spent.".format(e, end_time - st_time))

        if e % args.eval_freq == 0 or e == args.epochs - 1:
            model.eval()
            eval_outs = model(*eval_input, output_all_steps=True)
            eval_loss = batch_loss(
                eval_outs, eval_target, criterion, args.num_eval, args.core_steps
            )
            print(
                "Epoch %d, mean training loss: %f, mean evaluation loss: %f."
                % (
                    e,
                    train_loss.item() / args.num_train,
                    eval_loss.item() / args.num_eval,
                )
            )

            plot_test(model, args.sample_length, args.cuda)
            model.train()


if __name__ == "__main__":
    run()

"""
My implementation of sorting with graph networks (GNs) in pytorch.
Original tf implementation here: https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/sort.ipynb
"""

import argparse
import itertools

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pgn.graph import Vertex, DirectedEdge, Context, DirectedGraphWithContext
from pgn.models import EncoderCoreDecoder
from pgn.utils import pgn2nx, plot_graph


def graph_data_from_list(input_list):
    """
    Takes a list with the data, generates a fully connected graph with values of the list as nodes
    Parameters
    ----------
    input_list: list
    List with the data

    Returns
    -------
    pgn.Graph with the input_list values as nodes
    """
    connectivity = [el for el in itertools.product(range(len(input_list)), repeat=2)]
    vertices = [Vertex(i) for i in range(len(input_list))]

    graph_data = {'vertex': {'data': torch.Tensor([[v] for v in input_list]), 'info': vertices},
                  'edge': {'data': torch.zeros(len(connectivity), 1),
                           'info': [DirectedEdge(i, vertices[connectivity[i][0]], vertices[connectivity[i][1]]) for i in
                                range(len(connectivity))]},
                  'context': {'data': torch.Tensor([[0]]), 'info': [Context(0)]}
                  }
    add_connectivity(graph_data)
    return graph_data

def edge_id_by_sender_and_receiver(metadata, sid, rid):

    if sid > len(metadata) - 1 or rid > len(metadata) - 1:
        return -1

    for e in metadata:
        if e.sender.id == sid and e.receiver.id == rid:
            return e.id

    return -1

def create_target_data(input_data):
    # two nodes might have true since they might have similar values
    min_val = input_data['vertex']['data'].min()

    # [prob_true, prob_false]
    target_vertex_data = torch.Tensor([[1.0, 0.0] if v == min_val else [0.0, 1.0] for v in input_data['vertex']['data']])

    sorted_ids = input_data['vertex']['data'].argsort(dim=0).flatten()
    target_edge_data = torch.zeros(input_data['edge']['data'].shape[0], 2)

    for sidx, sid in enumerate(sorted_ids):
        for ridx, rid in enumerate(sorted_ids):
            eid = edge_id_by_sender_and_receiver(input_data['edge']['info'], sid, rid)
            # we look for exact comparison here since we sort
            if (sidx < len(sorted_ids) - 1 and ridx == sidx + 1):
                target_edge_data[eid][0] = 1.0
            else:
                target_edge_data[eid][1] = 1.0

    return {'vertex': target_vertex_data,
            'edge': target_edge_data}

def add_connectivity(entities):
    for v in entities['vertex']['info']:
        incoming = {'edge': [e for e in entities['edge']['info'] if e.receiver.id == v.id]}
        outgoing = {'edge': [e for e in entities['edge']['info'] if e.sender.id == v.id]}
        v.incoming_edges = incoming
        v.outgoing_edges = outgoing

def generate_graph_batch(n_examples, sample_length, target=True):
    input_data = [graph_data_from_list(np.random.uniform(size=sample_length)) for _ in range(n_examples)]
    if not target:
        return input_data

    target_data = [create_target_data(g) for g in input_data]
    return input_data, target_data


def batch_loss(outs, targets, criterion):
        loss = 0
        for out in outs:
            loss += sum([criterion(g['vertex']['vertex']['data'], t['vertex']) for g, t in zip(out, targets)])
            loss += sum([criterion(g['edge']['edge']['data'], t['edge']) for g, t in zip(out, targets)])
        return loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sorting with graph networks')
    parser.add_argument('--num-train', type=int, default=32, help='number of training examples')
    parser.add_argument('--num-eval', type=int, default=32, help='number of evaluation examples')
    parser.add_argument('--epochs', type=int, default=5000, help='number of training epochs')
    parser.add_argument('--core-steps', type=int, default=10, help='number of core processing steps')
    parser.add_argument('--sample-length', type=int, default=10, help='number of elements in the list to sort')
    parser.add_argument('--eval-freq', type=int, default=100, help='Evaluation/logging frequency')
    parser.add_argument('--cuda', action="store_true", help='Use a GPU if the system has it.')
    parser.add_argument('--verbose', action="store_true", help='Print diagnostircs.')
    parser.add_argument('--plot_graph_sample', action="store_true", help='Plot one of the input graphs')
    args = parser.parse_args()

    train_input, train_target = generate_graph_batch(args.num_train, sample_length=args.sample_length)
    eval_input, eval_target = generate_graph_batch(args.num_train, sample_length=args.sample_length)

    model = EncoderCoreDecoder(args.core_steps,
                               enc_vertex_shape=(1, 16),
                               core_vertex_shape=(32, 16),
                               dec_vertex_shape=(16, 16),
                               out_vertex_size=2,
                               enc_edge_shape=(1, 16),
                               core_edge_shape=(32, 16),
                               dec_edge_shape=(16, 16),
                               out_edge_size=2,
                               enc_global_shape=(1, 16),
                               core_global_shape=(32, 16),
                               dec_global_shape=(16, 16),
                               out_global_size=2,
                               )

    # plot one of the input graphs
    if args.plot_graph_sample:
        ng = pgn2nx(train_input[0])
        plot_graph(ng, fname='input_graph.pdf')
    
    if args.cuda and torch.cuda.is_available():
        for el in train_input + eval_input:
            for d in el.values():
                d['data'] = d['data'].to('cuda')
        for el in train_target + eval_target:
            for k in el.keys():
                el[k] = el[k].to('cuda')
        model.to('cuda')

    # the library expects the data to be in the form of
    # {'entity': {'etype1': {'data': [], 'info'}}}
    # the first level is the entity (e.g. vertex or edge)
    # the second level is the entity type
    # the last level is data and info
    for el in train_input + eval_input:
        el['vertex'] = {'vertex': el['vertex']}
        el['edge'] = {'edge': el['edge']}
        if 'context' in el:
            el['context'] = {'context': el['context']}


    optimiser = torch.optim.Adam(lr=0.001, params=model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    for e in range(args.epochs):
        st_time = time.time()

        train_outs = model.process_batch(train_input)
        train_loss = batch_loss(train_outs, train_target, criterion)
        optimiser.zero_grad()
        train_loss.backward()
        optimiser.step()

        end_time = time.time()
        if args.verbose:
            print('Epoch {} is done. {:.2f} sec spent.'.format(e, end_time - st_time))
        if e % args.eval_freq == 0 or e == args.epochs - 1:
            eval_outs = model.process_batch(eval_input, compute_grad=False)
            eval_loss = batch_loss(eval_outs, eval_target, criterion)
            print("Epoch %d, mean training loss: %f, mean evaluation loss: %f."
                  % (e, train_loss.item() / args.num_train, eval_loss.item() / args.num_train))

    unsorted = np.random.uniform(size=args.sample_length)
    test_g = graph_data_from_list(unsorted)
    test_g['vertex'] = {'vertex': test_g['vertex']}
    test_g['edge'] = {'edge': test_g['edge']}
    test_g['context'] = {'context': test_g['context']}

    if args.cuda and torch.cuda.is_available():
        test_g.to('cuda')

    g = model.forward([test_g])[0]
    g = DirectedGraphWithContext(g)

    # evaluate and plot
    mx = np.zeros((len(unsorted), len(unsorted)))
    for eid in range(g.num_edges('edge')):
        mx[g.senders(eid, 'edge').id][g.receivers(eid, 'edge').id] = g.edge_data('edge')[eid, 0]
    sort_indices = np.argsort(unsorted)
    plt.matshow(mx[sort_indices][:, sort_indices], cmap="viridis")
    plt.grid(False)
    try:
        plt.show()
    except:
        print("Wasn't able to show the plot. But I'll save it for sure.")
    finally:
        plt.savefig('pgn_sorting_output.png')

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from torch_scatter import scatter_mean

# TODO ideally, we would like all our inputs to be in the form [batch, time, data_shape]
# however I'm not sure if it's needed for aggregators since they are are not spread in time: [batch, data_shape]

class Aggregator(nn.Module):
    """
        A base class for aggregation functions.
        The main property is that they should operate on sets,
        i.e. be input permutation and # of inputs invariant.
    """

    def __init__(self, type=None):
        super().__init__()
        self._type = type

    @property
    def type(self):
        return self._type


class MeanAggregator(Aggregator):
    def forward(self, X, indices):
        # We can't simply batch this since the sublists can be of unequal length.
        # We either need to do this in a for loop or pad in a smart way
        # (simple padding will affect the average results, for instance).
        # TODO this thinks, that all graphs in a batch have the same topology
        # flatten X
        #gsize = len(X)
        #vsize = len(X[0])

        # lens = []
        # for g in X:
        #     lens.append([[el.shape[0]] for el in g])
        # lens = torch.tensor(lens, device=g[0].device, dtype=torch.float).detach()
        # X = [el for sl in X for el in sl]
        # ret = pad_sequence(X, batch_first=True).view(gsize, vsize, -1, fsize)
        # ret = ret.sum(dim=2)/lens
        #ret[ret.detach()!=ret.detach()] = 0.0

        # 0th dim is graph
        # 1st dim is entity id
        # 2st dim is num of entities to aggregate
        # 3st is their feature dim
        return scatter_mean(X, indices, dim=0)
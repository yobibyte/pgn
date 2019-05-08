import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

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
    def forward(self, X):
        # We can't simply batch this since the sublists can be of unequal length.
        # We either need to do this in a for loop or pad in a smart way
        # (simple padding will affect the average results, for instance).

        # flatten X
        bsize = len(X)
        vsize = len(X[0])
        lens = []
        for g in X:
            lens.append([[el.shape[1]] for el in g])
        lens = torch.tensor(lens, device=g[0].device, dtype=torch.float)
        X = [el for sl in X for el in sl]
        ret = pad_sequence(X, batch_first=True).view(bsize, vsize, -1, X[0].shape[1])
        ret = ret.sum(dim=2)
        ret /= lens
        ret[ret!=ret] = 0.0

        # 0th dim is graph
        # 1st dim is entity id
        return ret

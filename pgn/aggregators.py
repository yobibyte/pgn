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
        ret = pad_sequence(X, batch_first=True).sum(dim=1)/torch.Tensor([[el.shape[0]] for el in X])
        ret[ret != ret] = 0.0
        return ret

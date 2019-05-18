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
    def forward(self, X, indices, dim_size=None):
        # 0st dim is entity id
        # 1st is their feature dim
        return scatter_mean(X, indices, dim=0, dim_size=dim_size)

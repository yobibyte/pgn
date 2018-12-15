import torch
import torch.nn as nn

#TODO ideally, we would like all our inputs to be in the form [batch, time, data_shape]
# however I'm not sure if it's needed for aggregators since they are are not spread in time: [batch, data_shape]

class Aggregator(nn.Module):
    """
        A base class for aggregation functions.
        The main property is that they should operate on sets,
        i.e. be input permutation and # of inputs invariant.
    """

    def forward(self, X):
        if type(X) == list:
            return torch.stack(X)
        return X

class MeanAggregator(Aggregator):
    def forward(self, X):
        X = super().forward(X)
        return X.mean(dim=1)



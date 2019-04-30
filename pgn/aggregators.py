import torch
import torch.nn as nn


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

    def forward(self, X):
        """

        Parameters
        ----------
        X should be of shape [# vertex, # elements, # element features]

        Returns
        -------

        """
        if type(X) == list:
            raise ValueError('WTF, man, everything should be tensors here!')
        return X

    @property
    def type(self):
        return self._type


class MeanAggregator(Aggregator):
    def forward(self, X):
        # We can't simply batch this since the sublists can be of unequal length.
        # We either need to do this in a for loop or pad in a smart way
        # (simple padding will affect the average results, for instance).
        ret = [(super(MeanAggregator, self).forward(el)).mean(dim=0) for el in X]
        for el in ret:
            # if the entity has nothing to aggregate, it will be an empty list and will turn into nan after np.mean.
            # let's make it 0
            el[el != el] = 0.0
        return torch.stack(ret)

import torch.nn as nn
from torch_scatter import scatter_mean

"""Module to keep all the aggregation functions

Aggregation is batched within one graph only. 
I am not sure that batching along batch dimension (for different graphs) 
is better here. I tried that and padding took a lot of time. 
Moreover, I had to do splitting which made backward step muuuuuch slower.
 
"""


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

    def forward(self, *input):
        raise NotImplementedError("I am a simple man. All the implementation should be in my children.")


class MeanAggregator(Aggregator):
    """Average along the 0-th dimension (per entity)"""

    def forward(self, x, indices, dim_size=None):
        """

        Parameters
        ----------
        x: torch.Tensor
        indices: torch.LongTensor
        dim_size: How many elements should be in the output?
        We need that in case if an entity has not items to aggregate.
        Without it, the indices will be messed up.

        0-th dimension is entity id, e.g. edge id
        1-st dimension is the feature dim

        Returns
        -------

        """

        return scatter_mean(x, indices, dim=0, dim_size=dim_size)

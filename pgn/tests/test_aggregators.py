import unittest

import pgn.aggregators as aggregators
import torch

from torch_scatter import scatter_mean

class TestAggregators(unittest.TestCase):

    def test_mean(self):
        ag = aggregators.MeanAggregator()
        torch.testing.assert_allclose([[1.5]], ag(torch.Tensor([[1],[2]]), torch.LongTensor([0, 0])))
        torch.testing.assert_allclose([[3.5]], ag(torch.Tensor([[3], [4]]), torch.LongTensor([0, 0])))

    def test_scatter(self):
        ag = aggregators.MeanAggregator()
        a = torch.Tensor([[1], [2], [3], [3], [4]])
        torch.testing.assert_allclose(ag.forward(a, torch.LongTensor([0, 0, 0, 1, 1])), [[2], [3.5]])

    def test_multidim_scatter(self):
        ag = aggregators.MeanAggregator()
        a = torch.Tensor([[[1], [2], [3], [3], [4]], [[1], [2], [3], [3], [4]]])
        indices = torch.LongTensor([[[0], [0], [0], [1], [1]], [[1], [1], [1], [0], [0]]])
        res = torch.tensor([[[2], [3.5]], [[3.5], [2]]])

        torch.testing.assert_allclose(ag.forward(a, indices,dim=1), res)


if __name__ == '__main__':
    unittest.main()

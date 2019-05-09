import unittest

import pgn.aggregators as aggregators
import torch


class TestAggregators(unittest.TestCase):

    def test_mean(self):
        a = [[torch.Tensor([[1],[2]]), torch.Tensor([[3],[4]])]]
        ag = aggregators.MeanAggregator()
        torch.testing.assert_allclose(ag.forward(a, fsize=1), [[1.5], [3.5]])

    def test_concatenation(self):
        ''' What happens when the graph is not fully connected?
                    torch.Tensor won't have same dimensions for e2n aggregator input!  '''
        ag = aggregators.MeanAggregator()
        a = [[torch.Tensor([[1], [2], [3]]), torch.Tensor([[3], [4]])]]
        torch.testing.assert_allclose(ag.forward(a, fsize=1), [[2], [3.5]])


if __name__ == '__main__':
    unittest.main()

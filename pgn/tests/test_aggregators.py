import unittest

import torch
import pgn.aggregators as aggregators

class TestAggregators(unittest.TestCase):

    def test_mean(self):
        a = torch.Tensor([[[1],[2]],[[3],[4]]])
        ag = aggregators.MeanAggregator()
        torch.testing.assert_allclose(ag.forward(a), [[1.5], [3.5]])

    def test_concatenation(self):
        '''
            What happens when the graph is not fully connected?
            torch.Tensor won't have same dimensions for e2n aggregator input!
        '''
        ag = aggregators.MeanAggregator()
        a = [[[1], [2], [3]], [[3], [4]]]
        torch.testing.assert_allclose(ag.forward(a), [[2], [3.5]])

if __name__ == '__main__':
    unittest.main()
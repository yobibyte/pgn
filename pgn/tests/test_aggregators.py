import unittest

import torch
import pgn.aggregators as aggregators

class TestAggregators(unittest.TestCase):

    def test_mean(self):
        a = torch.Tensor([[1,2],[3,4]])
        ag = aggregators.MeanAggregator()
        torch.testing.assert_allclose(ag.forward(a), [1.5, 3.5])


if __name__ == '__main__':
    unittest.main()
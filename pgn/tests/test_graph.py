import unittest

from pgn import graph
import torch

class TestGraph(unittest.TestCase):

    def setUp(self):
        node_data = torch.Tensor([[0], [1]])
        edges_data = torch.Tensor([[0], [1], [2]])
        connectivity = [(0, 1), (0, 1), (1, 0)]
        self.g = graph.Graph(node_data, edges_data, connectivity=connectivity)

    def test_graph_build(self):

        self.assertEqual(self.g.nodes_data[0], 0)
        self.assertEqual(self.g.nodes_data[1], 1)

        self.assertEqual(self.g.edges_data[0], 0)
        self.assertEqual(self.g.edges_data[1], 1)
        self.assertEqual(self.g.edges_data[2], 2)

        self.assertEqual(set(self.g.incoming[0]), {2})
        self.assertEqual(set(self.g.outgoing[0]), {0, 1})

        self.assertEqual(set(self.g.incoming[1]), {0, 1})
        self.assertEqual(set(self.g.outgoing[1]), {2})

        self.assertEqual(self.g.senders[0], 0)
        self.assertEqual(self.g.receivers[0], 1)
        self.assertEqual(self.g.senders[1], 0)
        self.assertEqual(self.g.receivers[1], 1)
        self.assertEqual(self.g.senders[2], 1)
        self.assertEqual(self.g.receivers[2], 0)

    def test_graph_summary(self):
        self.g._graph_summary()


if __name__ == '__main__':
    unittest.main()
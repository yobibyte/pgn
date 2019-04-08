import unittest

from pgn import graph
import torch

class TestGraph(unittest.TestCase):

    def setUp(self):
        vertex_data = torch.Tensor([[0], [1]])
        edge_data = torch.Tensor([[0], [1], [2]])
        conn = [(0, 1), (0, 1), (1, 0)]
        entities = [
                    {'data': vertex_data,
                              'info': [graph.Vertex(el) for el in range(vertex_data.shape[0])]},
                    {'data': edge_data,
                               'info': [graph.DirectedEdge(el, conn[el][0], conn[el][1]) for el in range(edge_data.shape[0])]}
                    ]

        self.g = graph.DirectedGraph(entities)

    def test_graph_build(self):

        self.assertEqual(self.g.vertex_data()[0], 0)
        self.assertEqual(self.g.vertex_data()[1], 1)

        self.assertEqual(self.g.edge_data()[0], 0)
        self.assertEqual(self.g.edge_data()[1], 1)
        self.assertEqual(self.g.edge_data()[2], 2)

        # test with id output
        self.assertEqual(set(self.g.incoming_edges(0)), {2})
        self.assertEqual(set(self.g.outgoing_edges(0)), {0, 1})

        # test with list output
        self.assertEqual(set(self.g.incoming_edges()[1]), {0, 1})
        self.assertEqual(set(self.g.outgoing_edges()[1]), {2})

        # test with id output
        self.assertEqual(self.g.senders(1), 0)
        self.assertEqual(self.g.receivers(1), 1)
        self.assertEqual(self.g.senders(2), 1)
        self.assertEqual(self.g.receivers(2), 0)

        # test with the list output
        self.assertEqual(self.g.senders()[0], 0)
        self.assertEqual(self.g.receivers()[0], 1)

    # def test_graph_summary(self):
    #     self.g._graph_summary()


if __name__ == '__main__':
    unittest.main()
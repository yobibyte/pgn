import unittest

import torch
from pgn import graph
from pgn.graph import Vertex, DirectedEdge


class TestGraph(unittest.TestCase):
    def setUp(self):
        vinfo = [Vertex(id=i) for i in range(2)]
        vertices = {"vertex": {"data": torch.Tensor([[0], [1]]), "info": vinfo}}

        edges = {
            "edge": {
                "data": torch.Tensor([[0], [1], [2]]),
                "info": [
                    DirectedEdge(0, vinfo[0], vinfo[1]),
                    DirectedEdge(1, vinfo[0], vinfo[1]),
                    DirectedEdge(2, vinfo[1], vinfo[0]),
                ],
            }
        }
        for v in vertices["vertex"]["info"]:
            incoming = {
                "edge": [e for e in edges["edge"]["info"] if e.receiver.id == v.id]
            }
            outgoing = {
                "edge": [e for e in edges["edge"]["info"] if e.sender.id == v.id]
            }
            v.incoming_edges = incoming
            v.outgoing_edges = outgoing
        self.g = graph.DirectedGraph({"vertex": vertices, "edge": edges})

    def test_graph_build(self):
        self.assertEqual(self.g.vertex_data()["vertex"][0], 0)
        self.assertEqual(self.g.vertex_data()["vertex"][1], 1)

        self.assertEqual(self.g.edge_data()["edge"][0], 0)
        self.assertEqual(self.g.edge_data()["edge"][1], 1)
        self.assertEqual(self.g.edge_data()["edge"][2], 2)

        # test with id output
        self.assertEqual(
            set(el.id for el in self.g.incoming_edges(0, "vertex", "edge")), {2}
        )
        self.assertEqual(
            set(el.id for el in self.g.outgoing_edges(0, "vertex", "edge")), {0, 1}
        )

        self.assertEqual(
            set(el.id for el in self.g.incoming_edges(1, "vertex", "edge")), {0, 1}
        )
        self.assertEqual(
            set(el.id for el in self.g.outgoing_edges(1, "vertex", "edge")), {2}
        )

        # test with id output
        self.assertEqual(self.g.senders(1).id, 0)
        self.assertEqual(self.g.receivers(1).id, 1)
        self.assertEqual(self.g.senders(2).id, 1)
        self.assertEqual(self.g.receivers(2).id, 0)

        # test with the list output
        self.assertEqual(self.g.senders()[0].id, 0)
        self.assertEqual(self.g.receivers()[0].id, 1)

    def test_graph_summary(self):
        self.g._graph_summary()


if __name__ == "__main__":
    unittest.main()

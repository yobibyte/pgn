import unittest

from pgn import graph

class TestGraph(unittest.TestCase):

    def test_graph_build(self):
        g = graph.Graph()
        nid1 = g.add_node('First')
        nid2 = g.add_node('Second')
        eid1 = g.add_edge(nid1, nid2, 'first connector')
        eid2 = g.add_edge(nid1, nid2, 'second connector')
        eid3 = g.add_edge(nid2, nid1, 'third connector')

        n1 = g.get_node_by_id(nid1)
        n2 = g.get_node_by_id(nid2)

        e1 = g.get_edge_by_id(eid1)
        e2 = g.get_edge_by_id(eid2)
        e3 = g.get_edge_by_id(eid3)

        self.assertEqual(n1.data, 'First')
        self.assertEqual(n2.data, 'Second')

        self.assertEqual(e1.data, 'first connector')
        self.assertEqual(e2.data, 'second connector')
        self.assertEqual(e3.data, 'third connector')

        self.assertEqual(set(n1.incoming_edges.values()), {e3})
        self.assertEqual(set(n1.outcoming_edges.values()), {e1, e2})

        self.assertEqual(set(n2.incoming_edges.values()), {e1, e2})
        self.assertEqual(set(n2.outcoming_edges.values()), {e3})

        self.assertEqual(e1.sender.id, nid1)
        self.assertEqual(e1.receiver.id, nid2)
        self.assertEqual(e2.sender.id, nid1)
        self.assertEqual(e2.receiver.id, nid2)
        self.assertEqual(e3.sender.id, nid2)
        self.assertEqual(e3.receiver.id, nid1)

        g._graph_summary()


if __name__ == '__main__':
    unittest.main()
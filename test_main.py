import unittest
from unittest import TestCase
import numpy as np

import main


class MyTestCase(unittest.TestCase):
    def test_is_acyclic(self):
        # has cycle 1 -> 2 -> 3 -> 1
        partially_directed_cycle = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 0]
        ])
        self.assertFalse(main.is_acyclic(partially_directed_cycle))

        # only has 2 nodes, so this is considered acyclic despite the edges
        too_small = np.array([
            [0, 1],  # 1 <-> 2
            [1, 0]  # 2 <-> 1
        ])
        self.assertTrue(main.is_acyclic(too_small))

        # shaped like a triangle w/ undirected edges
        undirected_cycle = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        self.assertFalse(main.is_acyclic(undirected_cycle))

        # has same edges as undirected_cycle, but they are partially directed so that the graph no longer forms a cycle
        partially_directed_no_cycle = np.array([
            [0, 1, 1], # 1 -> 2, 1 <-> 3
            [0, 0, 0],
            [1, 1, 0] # 3 -> 2, 3 <-> 1
        ])
        self.assertTrue(main.is_acyclic(partially_directed_no_cycle))

if __name__ == '__main__':
    unittest.main()

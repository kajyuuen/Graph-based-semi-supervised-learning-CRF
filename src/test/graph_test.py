import sys
sys.path.append('..')
import unittest
import numpy as np
from graph import Graph

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class GraphTest(unittest.TestCase):
    def setUp(self):
        self.trigrams = ["a:a:a", "b:a:b", "c:c:c", "h:h:h", "a:j:l", "g:g:g"]
        self.labels = ["a", None, "c", "s", None, None]
        self.pmi_vectors = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0.9, 1, 1, 0.9],
            [1, 1, 10, 1, 10, 1, 10, 1, 10],
            [1, 3, 10, 1, 3, 1, 0, 1, 10],
            [10, 1, 10, 1, 5, 0, 10, 1, 10],
            [10, 1, 14, 1, 4, 100, 0, 1, 1]
        ]

    def test_sim_matrix(self):
        # k-nearest
        k = 3
        graph = Graph(self.trigrams, self.labels, self.pmi_vectors)
        first_sim_column = graph.sim_matrix[0]
        self.assertEqual(len(first_sim_column[first_sim_column > 0]), k+1)

if __name__ == '__main__':
    unittest.main()

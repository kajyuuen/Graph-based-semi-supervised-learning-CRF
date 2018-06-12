import sys
sys.path.append('..')
import unittest
from token_map import token_to_type, marginal_prob_add

class TokenMapTest(unittest.TestCase):
    def setUp(self):
        # Section1: x1 x2 x3 x4 x5
        # Section2: x7 x2 x3 x4 x8
        self.trigrams = [["x1:x2:x3", "x2:x3:x4", "x3:x4:x5"], ["x7:x2:x3", "x2:x3:x4", "x3:x4:x8"]]
        self.marginal_prob = [
            [{'a': 0.5, 'b': 0.5}, {'a': 0.4, 'b': 0.6}, {'a': 0.3, 'b': 0.7}, {'a': 0.2, 'b': 0.8}, {'a': 0.1, 'b': 0.9}],
            [{'a': 0.5, 'b': 0.5}, {'a': 0, 'b': 1}, {'a': 0.5, 'b': 0.5}, {'a': 0.8, 'b': 0.2}, {'a': 0.5, 'b': 0.5}]
        ]

    def test_token_to_type(self):
        q = token_to_type(self.trigrams, self.marginal_prob)
        sum_prob = marginal_prob_add(self.marginal_prob[0][2], self.marginal_prob[1][2])
        self.assertEqual(q['x2:x3:x4']['a'], sum_prob['a'] / 2)
        self.assertEqual(q['x2:x3:x4']['b'], sum_prob['b'] / 2)

if __name__ == '__main__':
    unittest.main()

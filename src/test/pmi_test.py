import sys
sys.path.append('..')
import unittest
from pmi import PMI

class PMITest(unittest.TestCase):
    def setUp(self):
        self.labels = ["x1x2x3", "x1x2x3", "x4x5x6"]
        self.features_list = [{'f1': "alpha", "f2": "beta"}, {'f1': "alpha", "f2": "gamma"}, {'f1': "alpha", "f2": "beta"}]
        self.pmi = PMI(self.labels, self.features_list)

    def test_pmi(self):
        self.assertEqual(self.pmi.pmi("x1x2x3", "alpha", "f1"), 0)
        self.assertTrue(self.pmi.pmi("x1x2x3", "beta", "f2") < 0)
        self.assertTrue(self.pmi.pmi("x1x2x3", "gamma", "f2") > 0)

    def test_pmi_vector(self):
        return 0
        #pmi_vec = self.pmi.pmi_vector('a', self.features_list[0])
        #self.assertEqual(len(pmi_vec), 2)

if __name__ == '__main__':
    unittest.main()

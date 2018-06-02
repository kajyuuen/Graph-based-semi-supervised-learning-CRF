import sys
sys.path.append('..')
import unittest
from pmi import PMI

class PMITest(unittest.TestCase):
    def setUp(self):
        self.x_list = ["a", "b", "b"]
        self.features_list = [{'f1': "book", "f2": "note"}, {'f1': "book", "f2": "pen"}, {'f1': "cat", "f2": "pen"}]
        self.pmi = PMI(self.x_list, self.features_list)

    def test_pmi(self):
        self.assertEqual(self.pmi.pmi("c", "apple", "f1"), 0)
        # 出現頻度が高い方がPMIが高くなる
        self.assertTrue(self.pmi.pmi("b", "pen", "f2") > self.pmi.pmi("b", "book", "f1"))

    def test_pmi_vector(self):
        pmi_vec = self.pmi.pmi_vector('a', self.features_list[0])
        self.assertEqual(len(pmi_vec), 2)


if __name__ == '__main__':
    unittest.main()

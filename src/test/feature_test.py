import sys
sys.path.append('..')
import unittest
from features import word2contextualfeature, sent2contextualfeature

class GraphFeatureTest(unittest.TestCase):
    def setUp(self):
        self.sent = [("x1", "X1", "X"), ("x2", "X2", "X"), ("x3", "X3", "X"), ("x4", "X4", "X"), ("x5", "X5", "X"), ("x6", "X6", "X")]

    def test_word2contextualfeature(self):
        features = word2contextualfeature(self.sent, 2)
        self.assertEqual(features['trigram+context'], 'x1:x2:x3:x4:x5')
        self.assertEqual(features['trigram'], 'x2:x3:x4')
        self.assertEqual(features['left_context'], 'x1:x2')
        self.assertEqual(features['right_context'], 'x4:x5')
        self.assertEqual(features['center_word'], 'x3')
        self.assertEqual(features['trigram-centerword'], 'x2:x4')
        self.assertEqual(features['left_word-right_context'], 'x2:x4:x5')
        self.assertEqual(features['left_context-right_word'], 'x1:x2:x4')
        self.assertFalse(features['suffix'])

    def test_sent2contextualfeature(self):
        features_list = sent2contextualfeature(self.sent)
        self.assertEqual(len(features_list), 2)


if __name__ == '__main__':
    unittest.main()

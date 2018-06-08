from collections import Counter
from math import log

import numpy as np
class PMI:
    def __init__(self, labels, features_list):
        self.label_counter = Counter(labels)
        self.sum_label = len(labels)

        feature_dict = {}
        label_feature_dict = {}
        for label, features in zip(labels, features_list):
            for feature_name, feature in features.items():
                if feature_name not in feature_dict:
                    feature_dict[feature_name] = []
                if feature_name not in label_feature_dict:
                    label_feature_dict[feature_name] = []
                feature_dict[feature_name].append(feature)
                label_feature_dict[feature_name].append((label, feature))

        self.feature_counters = {}
        self.label_feature_counters = {}
        for feature_name, feature in feature_dict.items():
            self.feature_counters[feature_name] = Counter(feature)
        for feature_name, label_feature in label_feature_dict.items():
            self.label_feature_counters[feature_name] = Counter(label_feature)

    def pmi(self, label, feature, feature_name):
        cnt_label = self.label_counter[label] if self.label_counter[label] is not None else 0
        cnt_feature = self.feature_counters[feature_name][feature] if self.feature_counters[feature_name][feature] is not None else 0
        cnt_label_feature = self.label_feature_counters[feature_name][(label, feature)] if self.label_feature_counters[feature_name][(label, feature)] is not None else 0

        if cnt_label == 0 or cnt_feature == 0 or cnt_label_feature == 0:
            score = 0
        else:
            score = log((cnt_label_feature * self.sum_label)/(cnt_label*cnt_feature))

        return score

    def pmi_vector(self, label, features_dict):
        return np.array([self.pmi(label, feature, feature_name) for feature_name, feature_set in features_dict.items() for feature in feature_set])

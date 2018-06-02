from collections import Counter
from math import log

class PMI:
    def __init__(self, x_list, features_list):
        self.x_counter = Counter(x_list)
        self.sum_x = len(x_list)

        xy_dict = {}
        y_dict = {}
        for x, features in zip(x_list, features_list):
            for feature_name, feature in features.items():
                if feature_name not in y_dict:
                     y_dict[feature_name] = []
                if feature_name not in xy_dict:
                     xy_dict[feature_name] = []
                y_dict[feature_name].append(feature)
                xy_dict[feature_name].append((x, feature))

        self.y_counters = {}
        self.xy_counters = {}
        for feature_name, y in y_dict.items():
            self.y_counters[feature_name] = Counter(y)
        for feature_name, xy in xy_dict.items():
            self.xy_counters[feature_name] = Counter(xy)

    def pmi(self, x, y, feature_name):
        cnt_x = self.x_counter[x] if self.x_counter[x] is not None else 0
        cnt_y = self.y_counters[feature_name][y] if self.y_counters[feature_name][y] is not None else 0
        cnt_xy = self.xy_counters[feature_name][(x, y)] if self.xy_counters[feature_name][(x, y)] is not None else 0
        if cnt_x == 0 or cnt_y == 0:
            score = 0
        else:
            score = log(cnt_xy * self.sum_x/(cnt_x*cnt_y))
        return score

    def pmi_vector(self, x, features):
        return [self.pmi(x, feature, feature_name) for feature_name, feature in features.items()]

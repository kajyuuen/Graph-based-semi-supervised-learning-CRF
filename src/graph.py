import numpy as np
import networkx as nx

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

class Graph:
    def __init__(self, trigrams, pmi_vectors):
        # k-nearest
        k = 3
        pmi_matrix = np.array(pmi_vectors)
        cos_sim_matrix = 1 - pairwise_distances(pmi_matrix, metric="cosine")
        # HACK: numpyだけで何とかしたい
        sim_matrix = cos_sim_matrix.copy()
        for c_ind, arr in enumerate(np.argsort(cos_sim_matrix, axis=1)):
            for r_ind in arr[0:len(arr)-k-1]:
                sim_matrix[c_ind][r_ind] = 0
        sim_matrix = sim_matrix - np.diag(np.ones(sim_matrix.shape[0]))
        # グラフのノード番号とtrigramsのindexは一致している
        self.G = nx.from_numpy_matrix(sim_matrix, create_using=nx.DiGraph())
        self.trigram = trigrams

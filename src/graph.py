import numpy as np
import networkx as nx
import copy

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from token_map import marginal_prob_add, marginal_prob_division, marginal_prob_times, marginal_prob_scala_add

class Graph:
    def __init__(self, trigrams, pmi_vectors, u):
        # k-nearest
        k = 3
        pmi_matrix = np.array(pmi_vectors)
        cos_sim_matrix = 1 - pairwise_distances(pmi_matrix, metric="cosine")
        # HACK: numpyだけで何とかしたい
        sim_matrix = cos_sim_matrix.copy()
        for c_ind, arr in enumerate(np.argsort(cos_sim_matrix, axis=1)):
            for r_ind in arr[0:len(arr)-k-1]:
                sim_matrix[c_ind][r_ind] = 0
        self.w = sim_matrix - np.diag(np.ones(sim_matrix.shape[0]))
        self.l = len(self.w) - u
        self.u = u
        self.trigrams = trigrams

    def setting_marginal_prob(self, marginal_prob):
        self.marginal_prob = marginal_prob

    def graph_propagations(self, r, q_0, mu, nu, y_size, marginal_prob_type, count):
        print(sum([ v for v in q_0[10].values()]))
        q = self.graph_propagation(r, q_0, mu, nu, y_size, marginal_prob_type)
        print("1")
        print(sum([ v for v in q[10].values()]))
        for i in range(count - 1):
            q = self.graph_propagation(r, q, mu, nu, y_size, marginal_prob_type)
            print(i+2)
            print(sum([ v for v in q[10].values()]))
        return q

    def graph_propagation(self, r, q, mu, nu, y_size, marginal_prob_type):
        # 正規化の式が違う
        # qの合計が1にならない
        q_next = []
        for u_ind in range(self.l + self.u):
            kappa = 0
            gamma = copy.deepcopy(marginal_prob_type)
            trigram = self.trigrams[u_ind]
            for v_ind, w_uv in enumerate(self.w[u_ind]):
                gamma = marginal_prob_add(gamma, marginal_prob_times(q[v_ind], w_uv))
                kappa =+ mu * w_uv
            kappa += nu
            if trigram in r:
                delta = r[trigram]
                kappa += 1
            else:
                delta = marginal_prob_type
            gamma = marginal_prob_add(gamma, delta)
            gamma = marginal_prob_scala_add(gamma, nu / y_size)
            p_q = marginal_prob_division(gamma, kappa)
            q_next.append(p_q)
        return q_next

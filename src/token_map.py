from collections import Counter
import copy

def marginal_prob_add(current_marginal_prob, marginal_prob):
    r_marginal_prob = copy.deepcopy(current_marginal_prob)
    for key, prob in current_marginal_prob.items():
        r_marginal_prob[key] += marginal_prob[key]
    return r_marginal_prob

def marginal_prob_scala_add(current_marginal_prob, scala):
    for key, prob in current_marginal_prob.items():
        current_marginal_prob[key] += scala
    return current_marginal_prob


def marginal_prob_division(marginal_prob, div):
    r_marginal_prob = copy.deepcopy(marginal_prob)
    for key, prob in marginal_prob.items():
        r_marginal_prob[key] = marginal_prob[key] / div
    return r_marginal_prob

def marginal_prob_times(marginal_prob, times):
    r_marginal_prob = copy.deepcopy(marginal_prob)
    for key, prob in marginal_prob.items():
        r_marginal_prob[key] = marginal_prob[key] * times
    return r_marginal_prob

def token_to_type(ngrams, marginal_probs):
    cq = {}
    q_trigrams = []
    ngram_type_counter = Counter()
    for ngram, marginal_prob in zip(ngrams, marginal_probs):
        q_trigrams.append(ngram)
        if ngram not in cq:
            cq[ngram] = marginal_prob
        else:
            cq[ngram] = marginal_prob_add(cq[ngram], marginal_prob)
    for ngram_type, count in ngram_type_counter.most_common():
        if count == 1:
            break
        cq[ngram] = marginal_prob_division(cq[ngram], count)
    q = []
    for ngram in q_trigrams:
        q.append(cq[ngram])
    return q

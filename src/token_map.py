from collections import Counter
import copy

def marginal_prob_add(current_marginal_prob, marginal_prob):
    r_marginal_prob = copy.deepcopy(current_marginal_prob)
    for key, prob in current_marginal_prob.items():
        r_marginal_prob[key] += marginal_prob[key]
    return r_marginal_prob

def marginal_prob_division(marginal_prob, div):
    r_marginal_prob = copy.deepcopy(marginal_prob)
    for key, prob in marginal_prob.items():
        r_marginal_prob[key] = marginal_prob[key]/ div
    return r_marginal_prob

def token_to_type(ngrams_list, marginal_prob):
    q = {}
    ngram_type_counter = Counter()
    for i, ngrams in enumerate(ngrams_list):
        for j, ngram in enumerate(ngrams[1:-1]):
            ngram_type_counter[ngram] += 1
            if ngram not in q:
                q[ngram] = marginal_prob[i][j+2]
            else:
                q[ngram] = marginal_prob_add(q[ngram], marginal_prob[i][j])
    for ngram_type, count in ngram_type_counter.most_common():
        if count == 1:
            break
        q[ngram] = marginal_prob_division(q[ngram], count)
    return q

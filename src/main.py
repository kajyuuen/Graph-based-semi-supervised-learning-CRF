import pickle
import argparse
import nltk
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
import numpy as np
import random
from collections import Counter
from itertools import chain

import features as f
from features import sent2features, sent2labels, sent2contextualfeature, sent2trigrams, contextualfeatureslist2dict, sent2pos
from pmi import PMI
from graph import Graph
from token_map import token_to_type
from annotation_data_split import annotation_data_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    saving_group = parser.add_mutually_exclusive_group()
    saving_group.add_argument('--save', action='store_true')
    saving_group.add_argument('--load', action='store_true')
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--crf', action='store_true', help='accuracy with supervised CRF')
    model_group.add_argument('--graph', action='store_true', help='accuracy with Graph based semi-supervised learning CRF')
    args = parser.parse_args()

    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))[0:10]
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))[0:10]

    if(args.crf):
        X_train = [sent2features(s) for s in train_sents]
        y_train = [sent2pos(s) for s in train_sents]
        X_test = [sent2features(s) for s in test_sents]
        y_test = [sent2pos(s) for s in test_sents]
        if(args.load):
            crf = joblib.load('crf.pkl')
        else:
            crf = sklearn_crfsuite.CRF(
                algorithm='l2sgd',
                c2=0.01,
                max_iterations=100,
                all_possible_transitions=True
            )
            crf.fit(X_train, y_train)
            if(args.save):
                joblib.dump(crf, 'crf.pkl')
        y_pred = crf.predict(X_test)
        score = metrics.flat_f1_score(y_test, y_pred, average='weighted')
    elif(args.graph):
        # Create label and unlabel data
        labeled_sents, partial_sents, missing_sents = annotation_data_split(train_sents, 0, 0, 0.5, 42)
        unlabeled_sents = random.sample(partial_sents + missing_sents, len(partial_sents) + len(missing_sents))
        all_sents = labeled_sents + unlabeled_sents
        X = [sent2features(s) for s in all_sents]
        unlabel_X = [sent2features(s) for s in unlabeled_sents]
        label_X = [sent2features(s) for s in labeled_sents]
        label_y = [sent2pos(s) for s in labeled_sents]

        # Training CRF
        crf = sklearn_crfsuite.CRF(
            algorithm='l2sgd',
            c2=0.01,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(label_X, label_y)

        # Create graph
        # ラベルありなしデータのサイズ
        label_ngram_list = [[sent2trigrams(s)] for s in labeled_sents]
        all_label_ngram_list = [ngram for ngrams in label_ngram_list for ngram in ngrams]
        unlabel_ngram_list = [[sent2trigrams(s)] for s in unlabeled_sents]
        all_unlabel_ngram_list = [ngram for ngrams in unlabel_ngram_list for ngram in ngrams]
        # グラフの構築
        contextualfeatures_list = [sent2contextualfeature(s) for s in all_sents]
        ngrams_list = []
        ngrams_list.append(all_label_ngram_list)
        ngrams_list.append(all_unlabel_ngram_list)
        all_ngrams = [ngram for ngrams in ngrams_list for ngram in ngrams]
        flat_ngrams = list(chain.from_iterable(list(chain.from_iterable(ngrams_list))))
        all_ngram_counter = Counter(flat_ngrams)
        all_features = [contextualfeature for contextualfeatures in contextualfeatures_list for contextualfeature in contextualfeatures]
        pmi = PMI(flat_ngrams, all_features)
        all_features_dict = contextualfeatureslist2dict(contextualfeatures_list)
        pmi_vectors = np.array([pmi.pmi_vector(ngram, all_features_dict) for ngram in flat_ngrams])
        if(args.load):
            with open('graph.dat', 'rb') as fp:
                graph = pickle.load(fp)
        else:
            graph = Graph(flat_ngrams, pmi_vectors, len(all_unlabel_ngram_list))
            if(args.save):
                with open('graph.dat', 'wb') as fp:
                    pickle.dump(graph, fp)

        # 4.1 Posterior Decoding
        marginal_prob = crf.predict_marginals(X)
        flatten_marginal = list(chain.from_iterable(marginal_prob))
        flatten_ngram = list(chain.from_iterable(list(chain.from_iterable(ngrams_list))))
        # 4.2 Token-to-Type Mapping
        q_0 = token_to_type(flatten_ngram, flatten_marginal)
        # 4.3 Graph Propagation
        marginal_prob_type = { label: 0 for label in marginal_prob[0][0].keys()}
        postags = list(filter(lambda str:str != '', [ p for labeled_sent in labeled_sents for (_, p, _) in labeled_sent]))
        labeled_flatten_ngram = list(chain.from_iterable(list(chain.from_iterable(label_ngram_list))))
        count_r, ngram_type_counter = f.ngramlist_and_sents2cr(labeled_flatten_ngram, postags, marginal_prob_type)
        r = f.count_r2r(count_r, ngram_type_counter)
        mu = 0.5
        nu = 0.1
        q = graph.graph_propagations(r, q_0, mu, nu, len(marginal_prob_type), marginal_prob_type, 10)
        score = 0

    print("Accuracy: {}".format(score))

import pickle
import argparse
import nltk
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
import numpy as np

from features import sent2features, sent2labels, sent2contextualfeature, sent2trigrams
from pmi import PMI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    saving_group = parser.add_mutually_exclusive_group()
    saving_group.add_argument('--save', action='store_true')
    saving_group.add_argument('--load', action='store_true')
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--crf', action='store_true', help='accuracy with supervised CRF')
    model_group.add_argument('--graph', action='store_true', help='accuracy with Graph based semi-supervised learning CRF')

    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    args = parser.parse_args()
    if(args.crf):
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
        labels = list(crf.classes_)
        labels.remove('O')
        y_pred = crf.predict(X_test)
        score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    elif(args.graph):
        if(args.load):
            with open('pmi_vectors.dat', 'rb') as fp:
                pmi_vectors = pickle.load(fp)
        else:
            contextualfeatures_list = [sent2contextualfeature(s) for s in train_sents]
            ngrams_list = [sent2trigrams(s) for s in train_sents]
            all_ngrams = [ngram for ngrams in ngrams_list for ngram in ngrams]
            all_features = [contextualfeature for contextualfeatures in contextualfeatures_list for contextualfeature in contextualfeatures]
            pmi = PMI(all_ngrams, all_features)
            pmi_vectors = [pmi.pmi_vector(ngram, features) for ngram, features in zip(all_ngrams, all_features)]
        if(args.save):
            with open('pmi_vectors.dat', 'wb') as fp:
                pickle.dump(pmi_vectors, fp)
        score = 0

    print("Accuracy: {}".format(score))


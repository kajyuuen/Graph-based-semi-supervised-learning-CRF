import argparse
import nltk
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.externals import joblib

from features import sent2features, sent2labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    saving_group = parser.add_mutually_exclusive_group()
    saving_group.add_argument('--save', action='store_true')
    saving_group.add_argument('--load', action='store_true')
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--crf', action='store_true', help='accuracy with supervised CRF')
    model_group.add_argument('--graph-based-crf', action='store_true', help='accuracy with Graph based semi-supervised learning CRF')

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
    elif(args.graph-based-crf):
        # TODO:
        score = 0

    print("Accuracy: {}".format(score))


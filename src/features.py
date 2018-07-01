from collections import Counter, defaultdict
import copy

def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),

    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper()
        })
    else:
        features['EOS'] = True

    return features

def word2features_usepostag(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def has_suffix(word):
    suffixes = ["ed", "ing"]
    for suffix in suffixes:
        if word.endswith(suffix):
            return True
    return False

def word2contextualfeature(sent, i):
    # trigram, 5-nearest neighbor
    words = [ word for (word, _, _) in sent[i-2:i+3]]

    features = {
        'trigram+context': ':'.join(words),
        'trigram': ':'.join(words[1:4]),
        'left_context': ':'.join(words[0:2]),
        'right_context': ':'.join(words[3:5]),
        'center_word': words[2],
        'trigram-centerword': words[1] +':'+ words[3],
        'left_word-right_context': words[1] + ':' + ':'.join(words[3:5]),
        'left_context-right_word': ':'.join(words[0:2]) + ':' + words[3],
        'suffix': has_suffix(words[2])
    }

    return features

def sent2contextualfeature(sent):
    return [word2contextualfeature(sent, i+2) for i in range(len(sent)-4)]

def sent2trigrams(sent):
    trigrams = []
    sent.insert(0, ("<BOS>", '', ''))
    sent.append(("<EOS>" , '', ''))
    for i in range(len(sent)-2):
        trigram = [word for (word, _, _) in (sent[i:i+3])]
        trigrams.append(':'.join(trigram))
    return trigrams

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def sent2pos(sent):
    return [postag for token, postag, label in sent]

def ngramlist_and_sents2cr(ngram_list, postags, marginal_prob_type):
    cr = {}
    ngram_type_counter = Counter()
    for ngram, pos_tag in zip(ngram_list, postags):
        ngram_type_counter[pos_tag] += 1
        if ngram not in cr:
            cr[ngram] = copy.deepcopy(marginal_prob_type)
        cr[ngram][pos_tag] += 1
    return cr, ngram_type_counter

def count_r2r(count_r, ngram_type_counter):
    r = {}
    for ngram, pos_cnt in count_r.items():
        all_cnt = sum([cnt for cnt in pos_cnt.values()])
        r[ngram] = { pos: cnt/all_cnt for pos, cnt in pos_cnt.items()}
    return r

def contextualfeatureslist2dict(contextualfeatures_list):
    features_dict = {}
    for contextualfeatures in contextualfeatures_list:
        for contextualfeature in contextualfeatures:
            for feature_name, feature in contextualfeature.items():
                if feature_name not in features_dict:
                    features_dict[feature_name] = set()
                features_dict[feature_name].add(feature)
    return features_dict

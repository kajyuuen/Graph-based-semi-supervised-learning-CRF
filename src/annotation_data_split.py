import random
from sklearn.model_selection import train_test_split

def annotation_data_split(sents, missing_rate=0.5, partial_label_size = 0.33, nonlabel_size=0.33, random_state=42):
    full_sents, deficit_sents = train_test_split(sents, test_size=(partial_label_size+nonlabel_size), random_state=random_state)
    tmp_partial_sents, missing_sents = train_test_split(deficit_sents, test_size=partial_label_size/(partial_label_size+nonlabel_size), random_state=random_state)

    partial_sents = []
    for sent in tmp_partial_sents:
        for (token, postag, label) in sent:
            if random.random() <= missing_rate:
                partial_sents.append((token, postag, label))
            else:
                partial_sents.append((token, None, None))
    missing_sents = [(token, None, None) for sents in missing_sents for (token, _, _) in sents]

    return full_sents, partial_sents, missing_sents

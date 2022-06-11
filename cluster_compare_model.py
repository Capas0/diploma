import pickle
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

WORKDIR = 'path/'

CORPUS_FILE = WORKDIR + 'corpus_sep.bin'
CORPUS_GEN_FILE = WORKDIR + 'corpus_gpt2_sep.bin'
CLUSTER_MODEL_FILE = WORKDIR + 'model.bin'

MODEL_FILE = WORKDIR + 'result_model.bin'


def get_matrix(model, data):
    result = np.zeros((len(data), 3), dtype=int)
    for i, text in enumerate(data):
        labels = model.best_estimator_.predict(text) if text else []
        for val, cnt in Counter(labels).items():
            result[i, val] = cnt
    return result


if __name__ == '__main__':
    with open(CORPUS_FILE, 'rb') as f:
        corpus = pickle.load(f)

    with open(CORPUS_GEN_FILE, 'rb') as f:
        corpus1 = pickle.load(f)

    with open(CLUSTER_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    matrix = get_matrix(model, corpus)
    matrix1 = get_matrix(model, corpus1)

    data = np.vstack((matrix, matrix1))
    labels = np.concatenate((np.full(len(matrix), 0), np.full(len(matrix1), 1)))
    print(data.shape, labels.shape)

    base = LogisticRegression(n_jobs=-1, random_state=0)
    params = {
        'C': [0.01, 0.05, 0.1, 0.5, 1]
    }
    clf = GridSearchCV(base, params, scoring='f1_weighted', verbose=3)
    clf.fit(data, labels)

    print(clf.best_params_, clf.best_score_)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf.best_estimator_, f)

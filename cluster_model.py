import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

WORKDIR = 'path/'

CORPUS_FILE = WORKDIR + 'corpus_unique.bin'
LABELS_FILE = WORKDIR + 'labels.npy'

CORPUS_GEN_FILE = WORKDIR + 'corpus_gpt2_unique.bin'
LABELS_GEN_FILE = WORKDIR + 'labels_gpt2.npy'

MODEL_FILE = WORKDIR + 'model.bin'

if __name__ == '__main__':
    with open(CORPUS_FILE, 'rb') as f:
        data1 = pickle.load(f)
    labels1 = np.load(LABELS_FILE)
    labels1[labels1 != -1] = 1
    print(len(data1), len(labels1))

    with open(CORPUS_GEN_FILE, 'rb') as f:
        data2 = pickle.load(f)
    labels2 = np.load(LABELS_GEN_FILE)
    labels2[labels2 != -1] = 2
    print(len(data2), len(labels2))

    data11 = data1[labels1 != -1]
    data22 = data2[labels2 != -1]

    first = set([item.tobytes() for item, label in zip(data11, labels1) if label > 0])
    second = set([item.tobytes() for item, label in zip(data22, labels2) if label > 0])

    data = np.vstack((data1, data2))

    labels = []
    for item in data:
        val = item.tobytes()
        if val in first:
            labels.append(1)
        elif val in second:
            labels.append(2)
        else:
            labels.append(0)
    labels = np.array(labels)

    model = RandomForestClassifier(n_jobs=-1, random_state=0)
    params1 = {
        'min_samples_leaf': [3, 5, 10, 20],
        'min_samples_split': [5, 10, 20, 50, 100]
    }

    clf = GridSearchCV(model, params1, scoring='f1_weighted', verbose=3).fit(data, labels)
    print(clf.best_params_, clf.best_score_)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf.best_estimator_, f)

import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from entropy_complexity import entropy_complexity
from entropy_complexity_borders import get_borders

WORKDIR = 'path/'

CORPUS_FILE = WORKDIR + 'corpus_sep.bin'
CORPUS_GEN_FILE = WORKDIR + 'corpus_gpt2_sep.bin'

MODEL_FILE = WORKDIR + 'model.bin'

N = 2
M = 8

if __name__ == '__main__':
    min_ec, max_ec = get_borders(n=N, m=M)
    with open(CORPUS_FILE, 'rb') as f:
        corpus = pickle.load(f)

    with open(CORPUS_GEN_FILE, 'rb') as f:
        corpus1 = pickle.load(f)

    dots = np.asarray([entropy_complexity([word[-M:] for word in text], n=N, m=M) for text in corpus])
    dots1 = np.asarray([entropy_complexity([word[-M:] for word in text], n=N, m=M) for text in corpus1])

    data = np.vstack((dots, dots1))
    labels = np.concatenate((np.full(len(dots), 0), np.full(len(dots1), 1)))
    print(data.shape, labels.shape)

    indx = np.arange(len(data))
    np.random.shuffle(indx)
    data_ = data[indx]
    labels_ = np.asarray(list(map(lambda x: 'gpt-2' if x else 'literature', labels[indx])))

    plt.figure(figsize=(10, 7))
    plt.title('Serbian')
    plt.plot(min_ec[:, 0], min_ec[:, 1], color='r')
    plt.plot(max_ec[:, 0], max_ec[:, 1], color='r')
    sns.scatterplot(x=data_[:, 0], y=data_[:, 1], hue=labels_, alpha=.5, zorder=2)
    plt.xlabel('entropy')
    plt.ylabel('complexity')
    plt.legend()
    plt.show()

    model = RandomForestClassifier(n_jobs=-1, random_state=0)
    params1 = {
        'min_samples_leaf': [3, 5, 10, 20],
        'min_samples_split': [5, 10, 20, 50, 100]
    }

    clf = GridSearchCV(model, params1, scoring='f1_weighted', verbose=3)
    clf.fit(data, labels)

    print(clf.best_params_, clf.best_score_)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf.best_estimator_, f)

import pickle
import random
import sys
import traceback
from time import time

import hdbscan
import numpy as np

WORKDIR = 'path/'
CORPUS_UNIQUE_FILE = WORKDIR + 'corpus_unique.bin'


def get_params(params, index=0, temp=None):
    if temp is None:
        temp = {}
    try:
        key = list(params)[index]
    except IndexError:
        yield temp
        return
    for item in params.get(key, []):
        yield from get_params(params, index + 1, {**temp, key: item})


if __name__ == '__main__':
    random.seed(0)

    with open(CORPUS_UNIQUE_FILE, 'rb') as f:
        data = pickle.load(f)

    param_dist = {
        'min_samples': [10, 20, 50],
        'min_cluster_size': [1000, 2000, 3000],
    }

    best_score = -10
    best_params = None
    best_model = None

    for params in get_params(param_dist):
        try:
            print(f'Params: {params}')
            start = time()
            model = hdbscan.HDBSCAN(
                gen_min_span_tree=True,
                core_dist_n_jobs=-1,
                **params
            ).fit(data)

            duration = time() - start
            score = model.relative_validity_
            print(f'Estimated: {duration}')
            print(f'Fast score: {score}')
            print(f'{np.unique(model.labels_).shape[0]} clusters')
            print()

            if best_score < score:
                best_score = score
                best_params = params.copy()
                best_model = model
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)

    print()
    print(f'Best params: {best_params}')
    print(f'Best fast score:{best_score}')

    np.save('labels.npy', best_model.labels_)

import os
import pickle
from itertools import chain

import numpy as np

WORKDIR = 'path/'

LEMMATIZED_DIR = WORKDIR + 'lemmatized'
CORPUS_FILE = WORKDIR + 'corpus_sep.bin'
CORPUS_UNIQUE_FILE = WORKDIR + 'corpus_unique.bin'
DICT_FILE = WORKDIR + 'sr_dict.bin'

VECTOR_LEN = 8
NGRAM = 2


def text2ngrams(text, dictionary):
    window = []
    for token in text.split():
        vec = dictionary.get(token, [])[:VECTOR_LEN]
        window.append(vec)
        window = window[-NGRAM:]

        if [] in window or len(window) < NGRAM:
            continue

        yield np.array(window).flatten()


if __name__ == '__main__':
    with open(DICT_FILE, 'rb') as f:
        dictionary = pickle.load(f)

    texts = sorted(os.listdir(LEMMATIZED_DIR))

    result = []
    for filename in texts:
        with open('{}/{}'.format(LEMMATIZED_DIR, filename), 'r', encoding='utf-8') as fin:
            text = fin.read()
        result.append(list(text2ngrams(text, dictionary)))

    with open(CORPUS_FILE, 'wb') as fout:
        pickle.dump(result, fout)
    with open(CORPUS_UNIQUE_FILE, 'wb') as fout:
        unq = np.unique(list(chain.from_iterable(result)), axis=0)
        pickle.dump(unq, fout)
    print(f'{unq.shape[0]} unique n-grams')

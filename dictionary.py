import re
import glob
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import numpy as np

WORKDIR = 'path/'
LEMMATIZED_DIR = WORKDIR + 'lemmatized'
DICT_FILE = WORKDIR + 'sr_dict.bin'

DICT_SHAPE = 16

ALPHABET = 'AaBbVvGgDdĐđEeŽžZzIiJjKkLlLjljMmNnNjnjOoPpRrSsTtĆćUuFfHhCcČčDždžŠš'


def make_table_and_dict(corpus, min_df, max_df, token_pattern=None, use_idf=True):
    if token_pattern:
        vectorizer = TfidfVectorizer(analyzer='word', min_df=min_df, max_df=max_df, token_pattern=token_pattern,
                                     use_idf=use_idf)
    else:
        vectorizer = TfidfVectorizer(analyzer='word', min_df=min_df, max_df=max_df)
    data_vectorized = vectorizer.fit_transform(corpus)
    return data_vectorized, vectorizer.get_feature_names_out(), vectorizer.idf_


if __name__ == '__main__':
    text_files = glob.glob('{}/*.txt'.format(LEMMATIZED_DIR))

    corpus = []
    for file in text_files:
        with open(file, encoding='utf-8') as f:
            text = re.sub(r'\s+', ' ', f.read())
            corpus.append(text)

    data_vectorized, dictionary, idfs = make_table_and_dict(corpus, 5, 0.8, '[{}]+'.format(ALPHABET))

    u, sigma, vt = svds(data_vectorized, DICT_SHAPE)

    n = len(sigma)
    # reverse the n first columns of u
    u[:, :n] = u[:, n - 1::-1]
    # reverse sigma
    sigma = sigma[::-1]
    # reverse the n first rows of vt
    vt[:n, :] = vt[n - 1::-1, :]

    vecs = np.dot(np.diag(sigma), vt).T
    sr_dict = dict(zip(dictionary, vecs))

    with open(DICT_FILE, 'wb') as f:
        pickle.dump(sr_dict, f)


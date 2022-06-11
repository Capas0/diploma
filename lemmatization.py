import os
from itertools import chain

import stanza

SOURCE_DIR = 'path/to/source'
LEMMATIZED_DIR = 'path/to/result'


def preprocess(nlp, text):
    doc = nlp(text)
    for token in chain(*doc.to_dict()):
        if token['upos'] == 'PUNCT':
            continue
        if token['upos'] in ['PROPN', 'NUM', 'X']:
            yield token['upos']
        else:
            yield token['lemma']


if __name__ == '__main__':
    sources = sorted(os.listdir(SOURCE_DIR))
    nlp = stanza.Pipeline('sr', processors='tokenize,pos,lemma')

    for filename in sources:
        with open('{}/{}'.format(SOURCE_DIR, filename), 'r', encoding='utf-8') as fin:
            text = fin.read()
        with open('{}/{}'.format(LEMMATIZED_DIR, filename), 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(preprocess(nlp, text)))


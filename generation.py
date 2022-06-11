import os
import pickle

from srtools import latin_to_cyrillic
from transformers import pipeline, set_seed

WORKDIR = 'path/'

SOURCE_DIR = WORKDIR + 'source'
GENERATED_TEXTS = WORKDIR + 'texts_gpt2.bin'

if __name__ == '__main__':
    sources = sorted(os.listdir(SOURCE_DIR))
    generator = pipeline('text-generation', model='macedonizer/sr-gpt2', device=0)
    set_seed(42)

    samples = []
    for filename in sources:
        with open(f'{SOURCE_DIR}/{filename}', 'r', encoding='utf-8') as fin:
            text = fin.read().split()

        if len(text) < 1000:
            continue

        sample = latin_to_cyrillic(' '.join(text[:500]))[:2000]
        samples.append(sample[:sample.rfind(' ')])

    print(f'Samples: {len(samples)}')
    result = [
        gen[0]['generated_text'][len(sample):].strip()
        for gen, sample in
        zip(generator(samples, do_sample=True, max_length=1000), samples)
    ]

    with open(GENERATED_TEXTS, 'wb') as f:
        pickle.dump(result, f)

    print(len(result))

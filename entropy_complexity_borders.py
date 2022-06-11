import math

import numpy as np


def s_b(N, p, k):
    if p != 0:
        return -(k * p * math.log(p) + (1 - p * k) * math.log((1 - p * k) / (N - k)))
    else:
        return math.log(N - k)


def s_b_1(N, p, k):
    return -(
            k * (p + 1 / N) / 2 * math.log((p + 1 / N) / 2) +
            (N - k) * ((1 - p * k) / (N - k) + 1 / N) / 2 * math.log((((1 - p * k) / (N - k)) + 1 / N) / 2))


def entropy_b(N, p, k):
    return s_b(N, p, k) / math.log(N)


def q_0_b(N):
    return 1 / (s_b(N, 1 / (2 * N), N - 1) - math.log(N) / 2)


def q_j_b(N, p, k):
    return q_0_b(N) * (s_b_1(N, p, k) - s_b(N, p, k) / 2 - math.log(N) / 2)


def complexity_b(N, p, k, ent=None):
    if ent is None:
        ent = entropy_b(N, p, k)
    return q_j_b(N, p, k) * ent


def entropy_complexity_b(N, p, k):
    e_b = entropy_b(N, p, k)
    c_b = complexity_b(N, p, k, e_b)
    return e_b, c_b


def get_borders(n, m):
    N = np.math.factorial(n) ** m
    i = 1
    entropy = []
    complexity = []
    while i * 100 < N:
        for k in range(N - i * 100, N - 1, i):
            e_b, c_b = entropy_complexity_b(N, 0, k)
            entropy.append(e_b)
            complexity.append(c_b)
        i *= 2
    for k in range(0, N - 1, i):
        e_b, c_b = entropy_complexity_b(N, 0, k)
        entropy.append(e_b)
        complexity.append(c_b)
    idx = np.argsort(entropy)
    entropy = np.array(entropy)[idx]
    complexity = np.array(complexity)[idx]
    max_ec = np.vstack([entropy, complexity]).T
    max_ec = np.vstack([[0, 0], max_ec, [1, 0]])

    entropy = []
    complexity = []
    for p in np.arange(0.01, 0.99, 0.01):
        e_b, c_b = entropy_complexity_b(N, p, 1)
        entropy.append(e_b)
        complexity.append(c_b)
    idx = np.argsort(entropy)
    entropy = np.array(entropy)[idx]
    complexity = np.array(complexity)[idx]
    min_ec = np.vstack([entropy, complexity]).T
    min_ec = np.vstack([[0, 0], min_ec, [1, 0]])
    del entropy
    del complexity
    return min_ec, max_ec

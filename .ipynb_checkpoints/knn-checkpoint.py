import numpy as np
from collections import Counter


def knn(data: np.array, target: np.array, x, k):
    assert k >= 1 and k <= data.shape[0], \
        "k must be valid interger"
    assert x.shape[0] == data.shape[1], "dimensions of input data must be equal to training data"

    distances = np.sum((data - x)**2, axis=1)
    res = target[np.argsort(distances)[:k]]
    res = Counter(res)
    return res.most_common(1)[0][0]

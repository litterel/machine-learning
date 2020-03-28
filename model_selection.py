import numpy as np


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0]
    assert test_ratio > 0 and test_ratio < 1
    
    counts = X.shape[0]
    test_size = int(counts*test_ratio)
    np.random.seed(seed)
    random_inx = np.random.permutation(np.arange(counts))
    test_inx = random_inx[:int(test_size)]
    train_inx = random_inx[int(test_size):]
    return X[train_inx], X[test_inx], y[train_inx],  y[test_inx]

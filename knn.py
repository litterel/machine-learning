import numpy as np
from collections import Counter
from metrics import accuracy_score


def knn(data: np.array, target: np.array, x, k):
    assert k >= 1 and k <= data.shape[0], \
        "k must be valid interger"
    assert x.shape[0] == data.shape[
        1], "dimensions of input data must be equal to training data"

    distances = np.sum((data - x)**2, axis=1)
    res = target[np.argsort(distances)[:k]]
    res = Counter(res)
    return res.most_common(1)[0][0]


class KNNclassifier():
    def __init__(self, k):
        k = int(k)
        assert k >= 1, "k must be a valid interger"
        self.train_data: np.ndarray = None
        self.train_target: np.ndarray = None
        self.k = k

    def _predict(self, x):
        distances = np.sum((self.train_data-x)**2, axis=1)
        nearest = self.train_target[np.argsort(distances)[:self.k]]
        res = Counter(nearest)
        return res.most_common(1)[0][0]

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "size of training data must be equal to training target."
        assert self.k <= X.shape[0], "size of training data must be no less than k."
        self.train_data = X
        self.train_target = y
        return self

    def predict(self, X: np.ndarray):
        assert self.train_data is not None and self.train_target is not None, "model must be fit before predicting"
        assert type(X) is np.ndarray or type(
            X) is np.matrix, "input data must be np.ndarray"
        assert X.shape[1] == self.train_data.shape[1]

        y = [self._predict(x) for x in X]
        return np.array(y)

    def score(self, test_x, test_y):
        prediction = self.predict(test_x)
        return accuracy_score(test_y, prediction)

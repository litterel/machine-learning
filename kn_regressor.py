import numpy as np
from metrics import r2_score


class KNRegressor:
    def __init__(self, k, weights="uniform"):
        k = int(k)
        assert k >= 1, "k must be a valid intreger."
        assert weights in [
            "uniform",
            "distance",
        ], "weights can only be 'uniform' or 'distance'"
        self.weights = weights
        self.train_data: np.ndarray = None
        self.train_target: np.ndarray = None
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert (X.shape[0] == y.shape[0]
                ), "size of training data must be equal to training target."
        assert self.k <= X.shape[
            0], "size of training data must be no less than k."
        self.train_data = X
        self.train_target = y
        return self

    def _predict(self, x):
        distances = np.sum((self.train_data - x)**2, axis=1)
        k_inx = np.argsort(distances)[:self.k]

        if self.weights == "uniform":
            k_weights = np.ones(self.k)
        if self.weights == "distance":
            k_weights = 1 / distances[k_inx]

        k_targets = self.train_target[k_inx]
        return np.average(k_targets, weights=k_weights)

    def predict(self, X: np.ndarray):
        assert (self.train_data is not None and self.train_target is not None
                ), "model must be fit before predicting"
        assert (type(X) is np.ndarray
                or type(X) is np.matrix), "input data must be np.ndarray"
        assert X.shape[1] == self.train_data.shape[1]
        y = [self._predict(x) for x in X]
        return np.array(y)

    def score(self, X, y):
        prediction = self.predict(X)
        return r2_score(y, prediction)

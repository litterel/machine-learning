import numpy as np


class MinMaxScaler():
    def __init__(self):
        self.max_ = None
        self.min_ = None

    def fit(self, X: np.ndarray):
        assert X.ndim == 2, "train data should be 2-dimensional."
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

        return self

    def transfrom(self, X: np.ndarray):
        assert self.max_ is not None and self.min_ is not None, "Fit should be done before transforming"
        assert X.shape[1] == len(
            self.min_), "Test data should have same coloumns as train data."
        assert X.ndim == 2

        return (X-self.min_)/(self.max_-self.min_)


class StandardScaler():
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        assert X.ndim == 2, "train data should be 2-dimensional."

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        return self

    def transform(self, X: np.ndarray):
        assert self.mean_ is not None and self.scale_ is not None, "Fit should be done before transforming"
        assert X.shape[1] == len(
            self.mean_), "Test data should have same coloumns as train data."
        assert X.ndim == 2
        return (X-self.mean_)/self.scale_

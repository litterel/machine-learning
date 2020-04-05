import numpy as np


class PCA:
    def __init__(self, n):
        n = int(n)
        assert n >= 1, "numbers of features must be at least 1."
        self._n = n
        self.compenents_: np.ndarray = None

    def _demean(self, X: np.ndarray):
        return X - X.mean(axis=0)

    def _normalize(self, w: np.ndarray):
        return w / np.linalg.norm(w)

    def _f(self, w: np.ndarray, X: np.ndarray):
        return np.sum((X.dot(w))**2) / len(X)

    def _df(self, w: np.ndarray, X: np.ndarray):

        return X.T.dot(X.dot(w)) * 2 / len(X)

    def _components(self,
                    w: np.ndarray,
                    X: np.ndarray,
                    max_iter=1e4,
                    epsilon=1e-8,
                    eta=0.01):
        w = np.ones(X.shape[1])
        for _ in np.arange(max_iter):
            learning_rate = 5 / (50 + iter)
            last_f = self._f(w, X)
            gradient = self._df(w, X)
            w = w + learning_rate * gradient
            w = self._normalize(w)
            if np.abs(last_f - self._f(w, X)) < epsilon:
                break
        return w

    def fit(self, X: np.ndarray):
        assert X.ndim == 2, "X must be 2 dimensional."
        assert X.shape[1] >= self._n
        X_p = self._demean(X)

        W = []
        for _ in np.arange(self._n):
            w = np.ones([X.shape[1]])
            w = self._components(w, X_p)
            W.append(w)
            k_component = X_p.dot(w)
            X_p = X_p - k_component.reshape(-1, 1) * w
        self.compenents_ = np.array(W)
        return self

    def transform(self, X: np.ndarray):
        assert self.compenents_ is not None, "Fit must be done before transforming"
        return X.dot(self.compenents_.T)

    def inverse_transform(self, X_p: np.ndarray):
        assert self.compenents_ is not None, "Fit must be done before restoring"
        assert X_p.shape[1] == self.compenents_.shape[0]
        return X_p.dot(self.compenents_)
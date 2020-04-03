import numpy as np


class PCA:
    def __init__(self, n):
        n = int(n)
        assert n >= 1, "numbers of features must be at least 1."
        self._n = n
        self.compenents_ = None

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
                    max_iter=20,
                    epsilon=1e-6,
                    eta=0.01):
        w = np.ones(X.shape[1])
        for _ in np.arange(max_iter):
            last_f = self._f(w, X)
            gradient = self._df(w, X)
            w = w + eta + gradient
            w = self._normalize(w)
            if np.abs(last_f - self._f(w, X)) < epsilon:
                break
        return w

    def fit(self, X: np.ndarray):
        assert X.ndim == 2, "X must be 2 dimensional."
        assert X.shape[1] >= self._n
        X_p = X.copy()
        X_p = self._demean(X_p)
        W = []
        #res = []
        for _ in np.arange(self._n):
            w = np.ones([X.shape[1]])
            w = self._components(w, X_p)
            W.append(w)
            k_component = X_p.dot(w)
            X_p = self._demean(X_p - k_component.reshape(-1, 1) * w)
        self.compenents_ = np.array(W)

    def transform(self, X: np.ndarray):
        assert self.compenents_ is not None
        return X.dot(self.compenents_.T)

    def restore(self, X_p: np.ndarray):
        assert self.compenents_ is not None
        assert X_p.shape[1] == self.compenents_.shape[0]
        return X_p.dot(self.compenents_)
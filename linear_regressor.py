import numpy as np
from .metrics import r2_score


class LinearRegressor():
    def __init__(self):
        #self.train_X = None
        #self.train_y = None
        self.coef_ = None
        self.b_ = None
        self._theta = None

    def fit_normal(self, X_train: np.ndarray, y_train: np.ndarray):
        assert X_train.shape[0] == y_train.shape[0]

        X_b: np.ndarray = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.b_ = self._theta[0]
        self.coef_ = self._theta[1:]

    def predict(self, X: np.ndarray):
        assert self._theta is not None
        assert X.shape[1] == len(self.coef_)
        X_b = np.hstack([np.ones(len(X), 1), X])
        return X_b.dot(self._theta)

    def score(self, X, y):
        prediction = self.predict(X)
        return r2_score(y, prediction)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "size of training data must be equal to training target."

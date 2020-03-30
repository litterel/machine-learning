import numpy as np
from metrics import r2_score


class LinearRegressor():
    def __init__(self):
        #self.train_X = None
        #self.train_y = None
        self.coef_ = None
        self.b_ = None
        self._theta = None

    def _J(self, X_b: np.ndarray, theta: np.ndarray, y: np.ndarray):
        assert X_b.shape[1] == len(
            theta), "X must have the same features as theta"
        assert X_b.shape[0] == len(y), "X must have the same size as y"
        try:
            return np.mean((y - X_b.dot(theta))**2)
        except:
            return np.inf

    def _J_partial(self, X_b: np.ndarray, theta: np.ndarray, y: np.ndarray):
        return 2 * X_b.T.dot(X_b.dot(theta) - y) / len(y)

    def fit_gd(self,
               X_train: np.ndarray,
               y_train: np.ndarray,
               eta=0.01,
               max_iter=1e4,
               epsilon=1e-5):
        '''data must be normalized before fitting, otherwise you might encounter overflow.'''

        assert X_train.shape[0] == y_train.shape[
            0], "X must have the same size as y"

        X_b: np.ndarray = np.hstack([np.ones((len(X_train), 1)), X_train])

        self._theta = np.ones(X_b.shape[1])

        for _ in np.arange(max_iter):
            last_J = self._J(X_b, self._theta, y_train)
            self._theta -= eta * self._J_partial(X_b, self._theta, y_train)
            if np.abs(1-self._J(X_b, self._theta, y_train)/last_J) < epsilon:
                break
        self.b_ = self._theta[0]
        self.coef_ = self._theta[1:]

    def fit_normal(self, X_train: np.ndarray, y_train: np.ndarray):
        assert X_train.shape[0] == y_train.shape[
            0], "X must have the same size as y"

        X_b: np.ndarray = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.b_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X: np.ndarray):
        assert self._theta is not None
        assert X.ndim == 2, "X should be 2 dimensional."
        assert X.shape[1] == len(self.coef_)
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return X_b.dot(self._theta)

    def score(self, X, y):
        prediction = self.predict(X)
        return r2_score(y, prediction)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[
            0], "size of training data must be equal to training target."

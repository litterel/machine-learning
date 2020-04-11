import numpy as np

from metric import accuracy_score

log = np.math.log


class LogisticRegressor():
    def __init__(self):
        #self.train_X = None
        #self.train_y = None
        self.coef_ = None
        self.b_ = None
        self._theta = None

    def _sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def _J(self, X_b: np.ndarray, theta: np.ndarray, y: np.ndarray):
        assert X_b.shape[1] == len(
            theta), "X must have the same features as theta"
        assert X_b.shape[0] == len(y), "X must have the same size as y"
        sig = self._sigmoid(X_b.dot(theta))
        try:
            return (np.sum(-y * log(sig) - (1 - y) * log(1 - sig))) / len(y)
        except:
            return np.inf

    def _J_partial(self, X_b: np.ndarray, theta: np.ndarray, y: np.ndarray):
        return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)

    def fit_gd(self,
               X_train: np.ndarray,
               y_train: np.ndarray,
               eta=0.01,
               max_iter=1e3,
               epsilon=1e-5):
        '''data must be normalized before fitting, otherwise you might encounter overflow.'''

        assert X_train.shape[0] == y_train.shape[
            0], "X must have the same size as y"

        X_b: np.ndarray = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.ones(X_b.shape[1])

        for iter in np.arange(max_iter):
            eta = 10 / (iter + 50)
            #last_J = self._J(X_b, self._theta, y_train)
            gradient = self._J_partial(X_b, self._theta, y_train)
            self._theta -= eta * gradient

        self.b_ = self._theta[0]
        self.coef_ = self._theta[1:]

    def fit(self, X, y):
        self.fit_gd(X, y)

    def _predict_prob(self, X: np.ndarray):
        assert self._theta is not None
        assert X.ndim == 2, "X should be 2 dimensional."
        assert X.shape[1] == len(self.coef_)
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return self._sigmoid(X_b.dot(self._theta))

    def decision_function(self, X: np.ndarray):
        return X.dot(self._theta) + 1

    def predict(self, X: np.ndarray):
        probability = self._predict_prob(X)

        return np.array(probability >= 0.5, dtype='int')

    def score(self, X, y):
        prediction = self.predict(X)
        return accuracy_score(y, prediction)
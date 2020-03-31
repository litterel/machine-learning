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

    def _J_partial_sgd(self, x_b: np.ndarray, theta: np.ndarray, y):
        return 2 * x_b * (np.sum(x_b * theta) - y)

    def fit_sgd(self,
                X_train: np.ndarray,
                y_train: np.ndarray,
                max_iter=10,
                t0=1,
                t1=50):
        '''data must be normalized before fitting, otherwise you might encounter overflow.
            You should choose t0 and t1 carefully.
        '''

        assert X_train.shape[0] == y_train.shape[
            0], "X must have the same size as y"

        X_b: np.ndarray = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.ones(X_b.shape[1])
        m = len(X_b)
        for iter in np.arange(max_iter):
            random_inx = np.random.permutation(m)
            for i in np.arange(m):
                learning_rate = t0 / (t1 + iter * m)
                last_J = self._J(X_b, self._theta, y_train)
                gradient_s = self._J_partial_sgd(X_b[random_inx[i]],
                                                 self._theta,
                                                 y_train[random_inx[i]])
                self._theta -= learning_rate * gradient_s
            '''
    
            if np.abs(X_b, self._theta, y_train - last_J) < epsilon:
                break
            '''

        self.b_ = self._theta[0]
        self.coef_ = self._theta[1:]

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

        for iter in np.arange(max_iter):
            eta = 10 / (iter + 50)
            last_J = self._J(X_b, self._theta, y_train)
            gradient = self._J_partial(X_b, self._theta, y_train)
            self._theta -= eta * gradient
            if np.abs(self._J(X_b, self._theta, y_train) - last_J) < epsilon:
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

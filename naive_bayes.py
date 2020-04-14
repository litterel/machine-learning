import numpy as np
import re
from metric import accuracy_score
from nltk.corpus import stopwords


class NaiveBayesClassifier():
    def __init__(self):
        self._vocab_list = None
        self._category_list = None
        
        self._p_c = None
        self._p_c_log = None
        self._p_word_mat = None
        self._p_word_mat_log = None
        

    def _creat_vocab_list(self, X: np.ndarray):
        self._vocab_list = set()
        for item in X:
            self._vocab_list = self._vocab_list.union(set(item))
        stopwords_set = set(stopwords.words('english'))
        self._vocab_list = self._vocab_list - stopwords_set
        self._vocab_list = list(self._vocab_list)

    def _word2vec(self, x):
        input_set = set(x)
        return_vec = np.zeros(len(self._vocab_list))
        for i in range(len(self._vocab_list)):
            if self._vocab_list[i] in input_set:
                return_vec[i] = 1
        return return_vec

    def _word2vec_batch(self, X):
        return_mat = []
        for item in X:
            return_mat.append(self._word2vec(item))
        return np.array(return_mat)

    '''
    def _word2vec(self, X: np.ndarray):
        return_mat = []
        for item in X:
            return_mat.append(self._word2vec_single(item))
        return np.array(return_mat)
    '''

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2, "trainning data must be 2 dimensional array."
        assert (X.shape[0] == y.shape[0]
                ), "size of training data must be equal to training target."

        self._creat_vocab_list(X)
        self._category_list = np.unique(y)
        self._p_c = []
        self._p_word_mat = []
        for item in self._category_list:
            target_X = self._word2vec_batch(X[y == item])
            target_y = y[y == item]
            self._p_c.append(len(target_y) / len(y))
            self._p_word_mat.append(
                (np.sum(target_X, axis=0) + 1) / (len(target_y) + 1))

        self._p_c = np.array(self._p_c)
        self._p_word_mat = np.array(self._p_word_mat)
        self._p_c_log = np.log(self._p_c)
        self._p_word_mat_log = np.log(self._p_word_mat)

    def _predict(self, x):
        x_vec = self._word2vec(x)
        return self._category_list[np.argmax(
            self._p_c_log + np.sum(self._p_word_mat_log * x_vec, axis=1))]

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict(x))
        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

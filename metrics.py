import numpy as np


def accuracy_score(target: np.ndarray, prediction: np.ndarray):
    assert target.shape[0] == prediction.shape[
        0], "the size of target should be equal to the prediction"
    return np.sum(target == prediction) / len(target)


def MSE_error(target: np.ndarray, prediction: np.ndarray):
    assert target.shape[0] == prediction.shape[
        0], "the size of target should be equal to the prediction"
    return np.mean((target - prediction)**2)


def r2_score(target: np.ndarray, prediction: np.ndarray):
    assert target.shape[0] == prediction.shape[
        0], "the size of target should be equal to the prediction"
    return 1 - MSE_error(target, prediction) / np.var(target)


def confusion_matrix(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.array([[
        np.sum((y_true == 0) & (y_predict == 0)),
        np.sum((y_true == 0) & (y_predict == 1))
    ],
                     [
                         np.sum((y_true == 1) & (y_predict == 0)),
                         np.sum((y_true == 1) & (y_predict == 1))
                     ]])

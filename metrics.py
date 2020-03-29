import numpy as np


def accuracy_score(target: np.ndarray, prediction: np.ndarray):
    assert target.shape[0] == prediction.shape[0], "the size of target should be equal to the prediction"
    return np.sum(target == prediction)/len(target)


def MSE_error(target: np.ndarray, prediction: np.ndarray):
    assert target.shape[0] == prediction.shape[0], "the size of target should be equal to the prediction"
    return np.mean((target-prediction)**2)


def r2_score(target: np.ndarray, prediction: np.ndarray):
    assert target.shape[0] == prediction.shape[0], "the size of target should be equal to the prediction"
    return 1 - MSE_error(target, prediction)/np.var(target)


def euler():
    pass

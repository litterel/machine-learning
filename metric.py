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


def precision_score(y_true, y_predict):
    tp = np.sum((y_true == 1) & (y_predict == 1))
    fp = np.sum((y_true == 0) & (y_predict == 1))
    try:
        return tp / (tp + fp)
    except:
        return 0


def recall_score(y_true, y_predict):
    tp = np.sum((y_true == 1) & (y_predict == 1))
    fn = np.sum((y_true == 1) & (y_predict == 0))
    return tp / (tp + fn)


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


def f1_score(y_true, y_predict):
    '''
    精准率和召回率的调和平均数
    调和平均数的好处在于，如果一个值特别大一个值特别小，那么平均值会很小
    '''
    ps = precision_score(y_true, y_predict)
    rs = recall_score(y_true, y_predict)

    try:
        return 2 * ps * rs / (ps + rs)
    except:
        return 0


def precision_recall_curve():
    pass
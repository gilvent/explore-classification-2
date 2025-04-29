import numpy as np


def accuracy_score(actual_Y, pred_Y):
    return np.sum(actual_Y == pred_Y, axis=0) / len(actual_Y)


def precision_score(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def recall_score(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def f1_score(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def macro_f1_score(conf_matrix: np.ndarray):
    class_count = conf_matrix.shape[0]
    total_f1 = 0

    for i in range(0, class_count):
        tp = conf_matrix[i][i]
        fn = np.sum(conf_matrix[i]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        precision = precision_score(tp=tp, fp=fp)
        recall = recall_score(tp=tp, fn=fn)
        total_f1 += f1_score(precision=precision, recall=recall)

    return total_f1 / class_count

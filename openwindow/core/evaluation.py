import numpy as np
import pandas as pd
import sklearn.metrics as metrics


def evaluate(y_true, y_pred, label_set, threshold=-1):
    class_scores = compute_scores(y_true, y_pred).T
    macro_scores = np.mean(class_scores, axis=0, keepdims=True)
    micro_scores = compute_scores(y_true, y_pred, average='micro')

    # Create DataFrame for scores
    data = np.concatenate((class_scores, macro_scores, micro_scores[None, :]))
    index = label_set + ['Macro Average', 'Micro Average']
    columns = ['Accuracy', 'Average Precision']
    df = pd.DataFrame(data, pd.Index(index, name='Class'), columns)
    return df


def compute_scores(y_true, y_pred, average=None):
    acc = accuracy(y_true, y_pred, average)
    ap = average_precision(y_true, y_pred, average)
    return np.array([acc, ap])


def accuracy(y_true, y_pred, average=None):
    if len(y_pred.shape) > 1:
        n_classes = y_pred.shape[1]
        y_pred = y_pred.argmax(axis=1)

    if average is None:
        return [accuracy(y_true[y_true == k],
                         y_pred[y_true == k],
                         average='micro')
                for k in range(n_classes)]
    if average == 'macro':
        return np.mean(accuracy(y_true, y_pred, average=None), axis=0)
    if average == 'micro':
        return metrics.accuracy_score(y_true, y_pred)

    raise ValueError(f'Unrecognized value for `average` parameter: {average}')


def average_precision(y_true, y_pred, average=None):
    y_true = np.eye(y_pred.shape[1])[y_true]
    return metrics.average_precision_score(y_true, y_pred, average=average)


def confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred.argmax(axis=1))

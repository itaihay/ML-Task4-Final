import numpy as np
from sklearn import metrics


def auc_pr(y_true, y_pred, num_classes):
    precision = dict()
    recall = dict()
    auc = list()
    for i in range(num_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true[:, i],
                                                                    y_pred[:, i])
        auc.append(metrics.auc(recall[i], precision[i]))

    return np.average(auc)

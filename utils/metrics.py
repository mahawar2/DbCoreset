import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    aucs = []
    for i in range(y_true.shape[1]):
        try:
            aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        except:
            continue

    macro_f1 = f1_score(y_true, y_pred > 0.5, average='macro')
    micro_f1 = f1_score(y_true, y_pred > 0.5, average='micro')

    return {
        'mean_auc': np.mean(aucs),
        'macro_f1': macro_f1,
        'micro_f1': micro_f1
    }

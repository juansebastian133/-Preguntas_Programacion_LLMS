import numpy as np

def evaluar_modelo_avanzado(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    total = tn + fp + fn + tp

    accuracy = (tn + tp) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)


def detection(y, score, thr=0.5):
    pred = (score >= thr).astype(int)
    return {
        "acc": accuracy_score(y, pred) * 100,
        "prec": precision_score(y, pred, zero_division=0) * 100,
        "rec": recall_score(y, pred, zero_division=0) * 100,
        "f1": f1_score(y, pred, average="macro", zero_division=0) * 100,
        "auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else float("nan"),
        "ap": average_precision_score(y, score) if len(np.unique(y)) > 1 else float("nan"),
    }


def attribution(y, pred, n_classes=5):
    per = f1_score(y, pred, labels=list(range(n_classes)), average=None, zero_division=0) * 100
    return {"per_class": per.tolist(), "macro": float(per.mean()), "acc": accuracy_score(y, pred) * 100}


def confusion(y, pred, n_classes=5):
    cm = confusion_matrix(y, pred, labels=list(range(n_classes))).astype(float)
    return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0) * 100


def roc(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    return fpr, tpr


def pr(y, score):
    p, r, _ = precision_recall_curve(y, score)
    return r, p

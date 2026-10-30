import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from satforensics.experiment import aggregate


def prepare(train, val, test):
    X = {k: aggregate(p) for k, p in (("train", train), ("val", val), ("test", test))}
    sc = StandardScaler().fit(X["train"])
    return {k: sc.transform(v) for k, v in X.items()}


def fit_search(kind, X, train, val, grid, seed):
    from sklearn.metrics import f1_score
    best, best_cfg = -1.0, None
    am = train["attr"] >= 0
    avm = val["attr"] >= 0
    for cfg in grid:
        if kind == "svm":
            det = SVC(kernel="rbf", C=cfg["C"], gamma=cfg["gamma"], probability=False)
            att = SVC(kernel="rbf", C=cfg["C"], gamma=cfg["gamma"])
        else:
            det = RandomForestClassifier(200, max_depth=cfg["depth"], min_samples_leaf=cfg["leaf"], n_jobs=2, random_state=seed)
            att = RandomForestClassifier(200, max_depth=cfg["depth"], min_samples_leaf=cfg["leaf"], n_jobs=2, random_state=seed)
        det.fit(X["train"], train["det"])
        f1 = f1_score(val["det"], det.predict(X["val"]), average="macro")
        if f1 > best:
            best, best_cfg = f1, cfg
    if kind == "svm":
        det = SVC(kernel="rbf", C=best_cfg["C"], gamma=best_cfg["gamma"])
        att = SVC(kernel="rbf", C=best_cfg["C"], gamma=best_cfg["gamma"])
    else:
        det = RandomForestClassifier(200, max_depth=best_cfg["depth"], min_samples_leaf=best_cfg["leaf"], n_jobs=2, random_state=seed)
        att = RandomForestClassifier(200, max_depth=best_cfg["depth"], min_samples_leaf=best_cfg["leaf"], n_jobs=2, random_state=seed)
    det.fit(X["train"], train["det"])
    att.fit(X["train"][am], train["attr"][am])
    return det, att, best_cfg, best


def scores(det, att, X):
    if hasattr(det, "predict_proba"):
        s = det.predict_proba(X)[:, 1]
    else:
        d = det.decision_function(X)
        s = 1.0 / (1.0 + np.exp(-d))
    return s, att.predict(X)


SVM_GRID = [{"C": c, "gamma": g} for c in (1.0, 10.0, 100.0) for g in ("scale", 0.001, 0.01)]
RF_GRID = [{"depth": d, "leaf": l} for d in (None, 12, 24) for l in (1, 3)]

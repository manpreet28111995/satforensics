import json
import os
import sys

import numpy as np

from satforensics import experiment as ex

data_path, out = sys.argv[1], sys.argv[2]
data = ex.load(data_path)
idx = ex.group_split(data["norad"], 0)
parts = [ex.build_samples(data, ix, 100 + 17 * k) for k, ix in enumerate(idx)]
ARG, DOP = 8, 5


def stats(part):
    sel = np.where(part["attr"] >= 0)[0]
    arg = part["rf"][sel][:, :, ARG]
    dop = data["geo"][part["idx"][sel]][:, :, DOP]
    sd = arg.std(axis=1)
    cor = np.array([abs(np.corrcoef(a, d)[0, 1]) if a.std() > 1e-6 and d.std() > 1e-6 else 0.0 for a, d in zip(arg, dop)])
    return sel, sd, cor, part["attr"][sel]


res = {}
_, sd_v, cor_v, y_v = stats(parts[1])
_, sd_t, cor_t, y_t = stats(parts[2])
CW, SW, SP = 1, 2, 4
m = np.isin(y_v, [CW, SW])
yy = (y_v[m] == SW).astype(int)
best, thr = 0.0, 0.0
for t in np.unique(np.quantile(sd_v[m], np.linspace(0.02, 0.98, 97))):
    acc = ((sd_v[m] > t).astype(int) == yy).mean()
    if acc > best:
        best, thr = acc, t
mt = np.isin(y_t, [CW, SW])
acc_t = (((sd_t[mt] > thr).astype(int)) == (y_t[mt] == SW).astype(int)).mean()
res["cw_vs_sw"] = {"rule": "std of per-row position of the maximum", "threshold": float(thr), "val_acc": float(best), "test_acc": float(acc_t),
                   "median_cw": float(np.median(sd_t[y_t == CW])), "median_sw": float(np.median(sd_t[y_t == SW])), "n_test": int(mt.sum())}
m = np.isin(y_v, [CW, SP])
yy = (y_v[m] == SP).astype(int)
best, thr = 0.0, 0.0
for t in np.unique(np.quantile(cor_v[m], np.linspace(0.02, 0.98, 97))):
    acc = ((cor_v[m] > t).astype(int) == yy).mean()
    if acc > best:
        best, thr = acc, t
mt = np.isin(y_t, [CW, SP])
acc_t = (((cor_t[mt] > thr).astype(int)) == (y_t[mt] == SP).astype(int)).mean()
res["cw_vs_sp"] = {"rule": "absolute correlation of the position of the maximum with the predicted Doppler curve", "threshold": float(thr),
                   "val_acc": float(best), "test_acc": float(acc_t),
                   "median_cw": float(np.median(cor_t[y_t == CW])), "median_sp": float(np.median(cor_t[y_t == SP])), "n_test": int(mt.sum())}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))

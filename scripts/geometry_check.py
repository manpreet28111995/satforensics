import json
import sys

import numpy as np

from satforensics.orbit import track

ds = np.load(sys.argv[1])
meta = {}
with open(sys.argv[2]) as fh:
    for line in fh:
        r = json.loads(line)
        meta[r["id"]] = r
diffs, keep = [], []
for i, rid in enumerate(ds["ids"].tolist()):
    r = meta[rid]
    if r.get("max_altitude") is None:
        continue
    trk = track(r, steps=256)
    diffs.append(abs(float(trk[:, 0].max()) - float(r["max_altitude"])))
    keep.append(i)
d = np.array(diffs)
nat = ds["natural"][np.array(keep)]
over = d > 5
out = {
    "n": int(len(d)), "mae": float(d.mean()), "median": float(np.median(d)), "p95": float(np.quantile(d, 0.95)),
    "max": float(d.max()), "share_over_5": float(over.mean() * 100), "n_over_5": int(over.sum()),
    "bad_rate_over_5": float(nat[over].mean() * 100), "bad_rate_rest": float(nat[~over].mean() * 100),
}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(out)

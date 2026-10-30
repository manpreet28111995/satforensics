import argparse
import glob
import json
import os

import numpy as np

from satforensics.features import context_vector, geo_features
from satforensics.orbit import frequency_hz, parse_time, track


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--wf", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse()
    meta = {}
    with open(a.meta) as fh:
        for line in fh:
            r = json.loads(line)
            meta[r["id"]] = r
    mats = {}
    for name in sorted(glob.glob(os.path.join(a.wf, "shard_*.npz"))):
        z = np.load(name)
        for i, m in zip(z["ids"].tolist(), z["mats"]):
            mats[i] = m
    stats = {"metadata_records": len(meta), "with_waterfall_matrix": len(mats)}
    rows, seen = [], set()
    dropped = {"duplicate_key": 0, "no_tle_or_station": 0, "propagation_error": 0, "bad_geometry": 0, "not_good_or_bad": 0}
    for rid in sorted(mats):
        r = meta[rid]
        if r["status"] not in ("good", "bad"):
            dropped["not_good_or_bad"] += 1
            continue
        key = (r["ground_station"], r["norad_cat_id"], r["start"])
        if key in seen:
            dropped["duplicate_key"] += 1
            continue
        seen.add(key)
        if not (r["tle1"] and r["tle2"] and r["station_lat"] is not None and r["station_lng"] is not None):
            dropped["no_tle_or_station"] += 1
            continue
        try:
            trk = track(r)
        except Exception:
            trk = None
        if trk is None:
            dropped["propagation_error"] += 1
            continue
        freq = frequency_hz(r)
        if freq <= 0 or not np.isfinite(trk).all():
            dropped["bad_geometry"] += 1
            continue
        geo, dshape = geo_features(trk, freq)
        span = (parse_time(r["end"]) - parse_time(r["start"])).total_seconds()
        rows.append((rid, r, mats[rid], geo, dshape, context_vector(r, span)))
    n = len(rows)
    out = {
        "ids": np.array([x[0] for x in rows]),
        "norad": np.array([x[1]["norad_cat_id"] for x in rows]),
        "station": np.array([x[1]["ground_station"] for x in rows]),
        "start": np.array([parse_time(x[1]["start"]).timestamp() for x in rows]),
        "natural": np.array([1 if x[1]["status"] == "bad" else 0 for x in rows]),
        "level": np.stack([x[2] for x in rows]),
        "geo": np.stack([x[3] for x in rows]),
        "dshape": np.stack([x[4] for x in rows]),
        "ctx": np.stack([x[5] for x in rows]),
    }
    np.savez_compressed(a.out, **out)
    stats.update({"kept": n, "dropped": dropped, "bad": int(out["natural"].sum()), "good": int(n - out["natural"].sum()),
                  "satellites": int(len(np.unique(out["norad"]))), "stations": int(len(np.unique(out["station"])))})
    print(json.dumps(stats, indent=1))
    with open(a.out.replace(".npz", "_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)


if __name__ == "__main__":
    main()

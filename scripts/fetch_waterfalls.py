import argparse
import json
import os
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from satforensics.waterfall import extract


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard", type=int, default=500)
    return p.parse_args()


def download(url, dest, retries=4):
    delay = 3.0
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "satforensics-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
                fh.write(resp.read())
            return True
        except Exception:
            time.sleep(delay)
            delay *= 2
    return False


def work(rec, tmpdir):
    path = os.path.join(tmpdir, "%d.png" % rec["id"])
    try:
        if not download(rec["waterfall"], path):
            return rec["id"], None
        return rec["id"], extract(path)
    except Exception:
        return rec["id"], None
    finally:
        if os.path.exists(path):
            os.remove(path)


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    recs = []
    with open(a.meta) as fh:
        for line in fh:
            r = json.loads(line)
            if r["status"] in ("good", "bad") and r["waterfall"]:
                recs.append(r)
    done = set()
    for name in os.listdir(a.out):
        if name.endswith(".npz"):
            done.update(np.load(os.path.join(a.out, name))["ids"].tolist())
    todo = [r for r in recs if r["id"] not in done]
    print("eligible", len(recs), "todo", len(todo), flush=True)
    shard_idx = len([n for n in os.listdir(a.out) if n.endswith(".npz")])
    ids, mats, failed = [], [], 0
    with tempfile.TemporaryDirectory() as tmpdir, ThreadPoolExecutor(a.workers) as ex:
        for n, (rid, mat) in enumerate(ex.map(lambda r: work(r, tmpdir), todo), 1):
            if mat is None:
                failed += 1
            else:
                ids.append(rid)
                mats.append(mat)
            if len(ids) >= a.shard:
                np.savez_compressed(os.path.join(a.out, "shard_%04d.npz" % shard_idx), ids=np.array(ids), mats=np.stack(mats))
                shard_idx += 1
                ids, mats = [], []
            if n % 200 == 0:
                print(n, len(todo), "failed", failed, flush=True)
    if ids:
        np.savez_compressed(os.path.join(a.out, "shard_%04d.npz" % shard_idx), ids=np.array(ids), mats=np.stack(mats))
    print("failed", failed)


if __name__ == "__main__":
    main()

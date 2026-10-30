import argparse
import datetime as dt
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from satforensics.api import query_window, slim


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--first", default="2025-03-01")
    p.add_argument("--last", default="2025-09-30")
    p.add_argument("--windows", type=int, default=700)
    p.add_argument("--minutes", type=int, default=15)
    p.add_argument("--pages", type=int, default=2)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main():
    a = parse()
    rng = random.Random(a.seed)
    t0 = dt.datetime.fromisoformat(a.first).replace(tzinfo=dt.timezone.utc)
    t1 = dt.datetime.fromisoformat(a.last).replace(tzinfo=dt.timezone.utc)
    span = int((t1 - t0).total_seconds() // 60) - a.minutes
    starts = sorted(rng.sample(range(span), a.windows))
    wins = []
    for m in starts:
        s = t0 + dt.timedelta(minutes=m)
        e = s + dt.timedelta(minutes=a.minutes)
        wins.append((s.strftime("%Y-%m-%dT%H:%M:%SZ"), e.strftime("%Y-%m-%dT%H:%M:%SZ")))
    seen = set()
    done = 0
    with open(a.out, "w") as fh, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(query_window, s, e, a.pages): (s, e) for s, e in wins}
        for f in as_completed(futs):
            done += 1
            try:
                rows = f.result()
            except Exception:
                rows = []
            for r in rows:
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                fh.write(json.dumps(slim(r)) + "\n")
            fh.flush()
            if done % 25 == 0:
                print(done, len(wins), len(seen), flush=True)
                with open(a.out + ".windows", "w") as wf:
                    wf.write(str(done))
    print("records", len(seen))
    with open(a.out + ".windows", "w") as fh:
        fh.write(str(done))


if __name__ == "__main__":
    main()

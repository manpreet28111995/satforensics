import collections
import datetime as dt
import json
import sys

import numpy as np

meta_path, ds_path, stats_path, geo_path, out = sys.argv[1:6]
status = collections.Counter()
windows_records = 0
eligible = 0
with open(meta_path) as fh:
    for line in fh:
        r = json.loads(line)
        windows_records += 1
        status[r["status"]] += 1
        if r["status"] in ("good", "bad") and r["waterfall"]:
            eligible += 1
z = np.load(ds_path)
stats = json.load(open(stats_path))
geo = json.load(open(geo_path))
start = z["start"]
first = dt.datetime.fromtimestamp(float(start.min()), dt.timezone.utc)
last = dt.datetime.fromtimestamp(float(start.max()), dt.timezone.utc)
per_sat = collections.Counter(z["norad"].tolist())
per_station = collections.Counter(z["station"].tolist())


def num(v):
    return "{:,}".format(int(v)).replace(",", "{,}")


m = {
    "NStatGood": num(status["good"]), "NStatBad": num(status["bad"]), "NStatUnknown": num(status["unknown"]),
    "NStatFailed": num(status["failed"]), "NStatOther": num(windows_records - status["good"] - status["bad"] - status["unknown"] - status["failed"]),
    "NEligible": num(eligible), "NMatrix": num(stats["with_waterfall_matrix"]),
    "NWfLost": num(eligible - stats["with_waterfall_matrix"]),
    "NDup": num(stats["dropped"]["duplicate_key"]), "NNoTLE": num(stats["dropped"]["no_tle_or_station"]),
    "NPropErr": num(stats["dropped"]["propagation_error"]), "NBadGeo": num(stats["dropped"]["bad_geometry"]),
    "SpanFirst": first.strftime("%d %B %Y"), "SpanLast": last.strftime("%d %B %Y"),
    "SatMedianObs": "%d" % int(np.median(list(per_sat.values()))), "SatMaxObs": "%d" % max(per_sat.values()),
    "StationMedianObs": "%d" % int(np.median(list(per_station.values()))),
    "GeoMed": "%.2f" % geo["median"], "GeoPNF": "%.2f" % geo["p95"], "GeoOver": "%.1f" % geo["share_over_5"], "GeoOverN": "%d" % geo["n_over_5"],
    "GeoOverBad": "%.1f" % geo["bad_rate_over_5"], "GeoRestBad": "%.1f" % geo["bad_rate_rest"],
    "NBadBySat": "%d" % sum(1 for s in per_sat if z["natural"][z["norad"] == s].any()),
}
import os
if os.path.exists(meta_path + ".windows"):
    m["NWindowsDone"] = open(meta_path + ".windows").read().strip()
with open(out, "w") as fh:
    for k, v in sorted(m.items()):
        fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(json.dumps(m, indent=1))

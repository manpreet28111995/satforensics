import json
import os
import sys

import numpy as np
from scipy import stats as st

res, data_stats, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
main = json.load(open(os.path.join(res, "main_results.json")))
cfgs = json.load(open(os.path.join(res, "configs.json")))
ARCH = ["BJ", "CW", "SW", "PB", "SP"]
NAMES = {"sf": "SatForensics", "sf_base": "SF-Base", "trans_s": "Trans.-S", "cnn": "CNN", "lstm": "LSTM", "svm": "SVM", "rf": "Random Forest",
         "no_gate": "SF w/o gating", "no_mi": "SF w/o MI loss", "no_geo": "SF w/o geometry", "no_ctx": "SF w/o context", "no_pe": "SF w/o pos.\\ enc."}
ORDER = ["svm", "rf", "cnn", "lstm", "trans_s", "sf_base", "sf"]


def vals(model, getter):
    return np.array([getter(r["models"][model]) for r in main], dtype=float)


def ms(model, getter, nd=1):
    v = vals(model, getter)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return v.mean(), sd


def fmt(m, s, nd=1, bold=False):
    t = ("%." + str(nd) + "f") % m
    u = ("%." + str(nd) + "f") % s
    body = "%s$\\pm$%s" % (t, u)
    return "\\textbf{%s}" % body if bold else body


def best_flags(models, getter, higher=True):
    means = [ms(m, getter)[0] for m in models]
    top = max(means) if higher else min(means)
    return [abs(x - top) < 1e-9 for x in means]


def column_table(models, cols, caption_cols=None):
    rows = []
    flags = {c: best_flags(models, g) for c, (_, g, _) in cols.items()}
    for i, m in enumerate(models):
        cells = []
        for c, (_, g, nd) in cols.items():
            a, b = ms(m, g)
            cells.append(fmt(a, b, nd, flags[c][i]))
        rows.append("%s & %s \\\\" % (NAMES[m], " & ".join(cells)))
    return rows


macros = {}
n_runs = len(main)
macros["NRuns"] = str(n_runs)
sets = [r["n"] for r in main]
macros["NTrainMean"] = "%d" % round(np.mean([s[0] for s in sets]))
macros["NValMean"] = "%d" % round(np.mean([s[1] for s in sets]))
macros["NTestMean"] = "%d" % round(np.mean([s[2] for s in sets]))

ds = json.load(open(data_stats))
macros["NRecords"] = "{:,}".format(ds["metadata_records"]).replace(",", "{,}")
macros["NKept"] = "{:,}".format(ds["kept"]).replace(",", "{,}")
macros["NGood"] = "{:,}".format(ds["good"]).replace(",", "{,}")
macros["NBad"] = "{:,}".format(ds["bad"]).replace(",", "{,}")
macros["NSats"] = str(ds["satellites"])
macros["NStations"] = str(ds["stations"])
macros["BadRate"] = "%.1f" % (100.0 * ds["bad"] / ds["kept"])

det_cols = {
    "acc": ("Acc.", lambda r: r["all"]["acc"], 1),
    "prec": ("Prec.", lambda r: r["all"]["prec"], 1),
    "rec": ("Rec.", lambda r: r["all"]["rec"], 1),
    "f1": ("F1", lambda r: r["all"]["f1"], 1),
    "auc": ("AUC", lambda r: r["all"]["auc"], 3),
}
rows = column_table(ORDER, det_cols)
open(os.path.join(out, "tab_detection.tex"), "w").write("\n".join(rows))

split_cols = {
    "nf": ("F1 nat", lambda r: r["natural"]["f1"], 1),
    "na": ("AUC nat", lambda r: r["natural"]["auc"], 3),
    "ap": ("AP nat", lambda r: r["natural"]["ap"], 3),
    "if": ("F1 inj", lambda r: r["injected"]["f1"], 1),
    "ia": ("AUC inj", lambda r: r["injected"]["auc"], 3),
}
open(os.path.join(out, "tab_detection_split.tex"), "w").write("\n".join(column_table(ORDER, split_cols)))

attr_models = ["svm", "rf", "cnn", "lstm", "trans_s", "sf_base", "sf"]
attr_cols = {a: (a, (lambda r, k=k: r["attr"]["per_class"][k]), 1) for k, a in enumerate(ARCH)}
attr_cols["macro"] = ("Macro", lambda r: r["attr"]["macro"], 1)
open(os.path.join(out, "tab_attribution.tex"), "w").write("\n".join(column_table(attr_models, attr_cols)))

abl = [("sf", "Full SatForensics"), ("no_pe", "w/o positional encoding"), ("no_geo", "w/o orbital geometry"),
       ("no_ctx", "w/o link context"), ("no_gate", "w/o gated fusion (concatenation)"), ("no_mi", "w/o MI alignment loss"),
       ("sf_base", "w/o gating and MI (SF-Base)"), ("heads1", "$H=1$ attention head"), ("layers1", "$L=1$ encoder layer"),
       ("trans_s", "RF modality only (Trans.-S)")]
rows = []
ref = vals("sf", lambda r: r["all"]["f1"])
refa = vals("sf", lambda r: r["attr"]["macro"])
fl_d = max(ms(k, lambda r: r["all"]["f1"])[0] for k, _ in abl)
fl_a = max(ms(k, lambda r: r["attr"]["macro"])[0] for k, _ in abl)
for k, lab in abl:
    d = ms(k, lambda r: r["all"]["f1"])
    n = ms(k, lambda r: r["natural"]["f1"])
    a = ms(k, lambda r: r["attr"]["macro"])
    rows.append("%s & %s & %s & %s \\\\" % (lab, fmt(d[0], d[1], 1, abs(d[0] - fl_d) < 1e-9), fmt(n[0], n[1], 1), fmt(a[0], a[1], 1, abs(a[0] - fl_a) < 1e-9)))
open(os.path.join(out, "tab_ablation.tex"), "w").write("\n".join(rows))

sig_rows = []
for k in ("svm", "rf", "cnn", "lstm", "trans_s", "sf_base", "no_gate", "no_mi", "no_geo", "no_ctx", "no_pe"):
    line = [NAMES.get(k, k)]
    for getter in (lambda r: r["all"]["f1"], lambda r: r["natural"]["f1"], lambda r: r["attr"]["macro"]):
        a = vals("sf", getter)
        b = vals(k, getter)
        diff = a - b
        if len(diff) > 1 and diff.std(ddof=1) > 0:
            t = st.ttest_rel(a, b)
            ci = st.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            line.append("%+.1f$\\pm$%.1f (%s)" % (diff.mean(), ci, ("%.3f" % t.pvalue) if t.pvalue >= 0.001 else "<0.001"))
        else:
            line.append("%+.1f" % diff.mean())
    sig_rows.append(line)
open(os.path.join(out, "tab_significance.tex"), "w").write("\n".join(" & ".join(r) + " \\\\" for r in sig_rows))

mask_rows = []
for k, lab in (("sf", "SatForensics (gated)"), ("no_gate", "Concatenation fusion")):
    cells = [lab]
    full = ms(k, lambda r: r["all"]["f1"])
    cells.append(fmt(*full, 1))
    for key in ("geo", "ctx", "geo_ctx", "rf"):
        a = ms(k, lambda r, key=key: r["masked"][key] if "masked" in r else float("nan"))
        cells.append(fmt(*a, 1))
    mask_rows.append(" & ".join(cells) + " \\\\")
open(os.path.join(out, "tab_masking.tex"), "w").write("\n".join(mask_rows))

lat = json.load(open(os.path.join(res, "latency.json")))
rows = []
for k in ORDER:
    v = lat[k]
    p = "--" if v["params"] is None else "%.1f" % (v["params"] / 1000.0)
    f = "--" if v["mflops"] is None else "%.2f" % v["mflops"]
    rows.append("%s & %s & %s & %.2f & %.2f \\\\" % (NAMES[k], p, f, v["latency_ms"], v["size_mb"]))
open(os.path.join(out, "tab_complexity.tex"), "w").write("\n".join(rows))
macros["SFLatency"] = "%.2f" % lat["sf"]["latency_ms"]
macros["SFParamsK"] = "%.1f" % (lat["sf"]["params"] / 1000.0)
macros["SFSizeMB"] = "%.2f" % lat["sf"]["size_mb"]
macros["SFMFLOPs"] = "%.2f" % lat["sf"]["mflops"]

led = json.load(open(os.path.join(res, "ledger.json")))
b = led["benchmark"]
rows = []
for r in led["rates"]:
    rows.append("%d & %.3f & %.3f & %.1f \\\\" % (r["rate"], r["median_ms"], r["p95_ms"], r["cpu_pct"]))
open(os.path.join(out, "tab_ledger.tex"), "w").write("\n".join(rows))
macros["LedgerMedianUs"] = "%.1f" % (b["median_ms"] * 1000)
macros["LedgerPNinetyFiveUs"] = "%.1f" % (b["p95_ms"] * 1000)
macros["LedgerBaseUs"] = "%.1f" % (b["baseline_median_ms"] * 1000)
macros["LedgerThroughput"] = "{:,}".format(int(round(b["max_throughput_eps"]))).replace(",", "{,}")
macros["LedgerVerifyMs"] = "%.0f" % b["verify_all_ms"]
macros["LedgerN"] = "{:,}".format(b["n"]).replace(",", "{,}")
macros["TamperDetected"] = str(b["tamper_detected"])
macros["TamperTrials"] = str(b["tamper_trials"])

for k in ("sf", "sf_base", "trans_s", "cnn", "lstm", "svm", "rf"):
    tag = {"sf": "SF", "sf_base": "SFBase", "trans_s": "TransS", "cnn": "CNN", "lstm": "LSTM", "svm": "SVM", "rf": "RF"}[k]
    a = ms(k, lambda r: r["all"]["f1"])
    n = ms(k, lambda r: r["natural"]["f1"])
    at = ms(k, lambda r: r["attr"]["macro"])
    au = ms(k, lambda r: r["all"]["auc"])
    na = ms(k, lambda r: r["natural"]["auc"])
    macros[tag + "DetF"] = fmt(a[0], a[1], 1)
    macros[tag + "NatF"] = fmt(n[0], n[1], 1)
    macros[tag + "Attr"] = fmt(at[0], at[1], 1)
    macros[tag + "AUC"] = fmt(au[0], au[1], 3)
    macros[tag + "NatAUC"] = fmt(na[0], na[1], 3)
    macros[tag + "DetFm"] = "%.1f" % a[0]
    macros[tag + "NatFm"] = "%.1f" % n[0]
    macros[tag + "Attrm"] = "%.1f" % at[0]

cross_path = os.path.join(res, "cross_satellite.json")
if os.path.exists(cross_path):
    macros["NLOSO"] = str(len(json.load(open(cross_path))))
best = cfgs["best"]
macros["Epochs"] = str(cfgs["epochs"])
macros["SFd"] = str(best["sf"]["d_model"])
macros["SFL"] = str(best["sf"]["layers"])
macros["SFH"] = str(best["sf"]["heads"])
macros["SFff"] = str(best["sf"]["d_ff"])
macros["SFlr"] = "%g" % best["sf"]["lr"]
macros["TSd"] = str(best["trans_s"]["d_model"])
macros["TSL"] = str(best["trans_s"]["layers"])
macros["TSlr"] = "%g" % best["trans_s"]["lr"]

cal = json.load(open(os.path.join(res, "calibration.json")))


def cs(getter):
    v = np.array([getter(r) for r in cal], dtype=float)
    return np.nanmean(v), (np.nanstd(v, ddof=1) if len(v) > 1 else 0.0)


rows = []
for t in ("0.05", "0.02", "0.01"):
    cells = ["%d\\%%" % round(float(t) * 100)]
    for key, scale, nd in (("tau_margin", 1.0, 2), ("tau_entropy", 1.0, 2), ("flagged_fraction", 100.0, 1),
                           ("error_unflagged", 100.0, 1), ("error_flagged", 100.0, 1), ("true_in_top2_flagged", 100.0, 1)):
        a, b = cs(lambda r, key=key, t=t: (np.nan if r["targets"][t][key] is None else r["targets"][t][key] * scale))
        valid = int(sum(1 for r in cal if r["targets"][t][key] is not None))
        if valid == 0:
            cells.append("--")
        elif valid == 1:
            cells.append(("%." + str(nd) + "f (1 run)") % a)
        else:
            cells.append(fmt(a, b, nd))
    rows.append(" & ".join(cells) + " \\\\")
open(os.path.join(out, "tab_calibration.tex"), "w").write("\n".join(rows))
a, b = cs(lambda r: r["temperature"])
macros["CalTemp"] = fmt(a, b, 2)
a, b = cs(lambda r: r["ece_before"] * 100)
macros["CalEceBefore"] = fmt(a, b, 2)
a, b = cs(lambda r: r["ece_after"] * 100)
macros["CalEceAfter"] = fmt(a, b, 2)
a, b = cs(lambda r: r["error_all"] * 100)
macros["CalErrAll"] = fmt(a, b, 1)
a, b = cs(lambda r: r["true_in_top2_all"] * 100)
macros["CalTopTwo"] = fmt(a, b, 1)
cm = np.mean([np.array(r["models"]["sf"]["confusion"]) for r in main], axis=0)
for nm, (i, j) in {"CWtoSW": (1, 2), "SWtoCW": (2, 1), "SPtoCW": (4, 1), "CWtoSP": (1, 4), "SWtoBJ": (2, 0), "BJtoPB": (0, 3), "PBtoBJ": (3, 0)}.items():
    macros["Conf" + nm] = "%.1f" % cm[i, j]

temporal = json.load(open(os.path.join(res, "temporal.json")))
station = json.load(open(os.path.join(res, "station_split.json")))
for tag, d in (("Temporal", temporal), ("Station", station)):
    for k, nm in (("sf", "SF"), ("trans_s", "TransS"), ("lstm", "LSTM")):
        macros[tag + nm + "F"] = "%.1f" % d[k]["all"]["f1"]
        macros[tag + nm + "Nat"] = "%.1f" % d[k]["natural"]["f1"]
        macros[tag + nm + "Attr"] = "%.1f" % d[k]["attr"]["macro"]
macros["StationNTest"] = str(station["n"][2])
macros["StationNStations"] = "%d / %d / %d" % tuple(station["stations"])
cr = json.load(open(cross_path))
f1s = np.array([r["f1"] for r in cr])
macros["LosoMean"] = "%.1f" % f1s.mean()
macros["LosoMin"] = "%.1f" % f1s.min()
macros["LosoMax"] = "%.1f" % f1s.max()
macros["LosoNatMean"] = "%.1f" % np.mean([r["natural_f1"] for r in cr])
macros["LosoAttrMean"] = "%.1f" % np.mean([r["attr"] for r in cr])
macros["LosoBelowSeventyFive"] = str(int((f1s < 75).sum()))

sweep = json.load(open(os.path.join(res, "jnr_sweep.json")))
by = {r["jnr"]: r for r in sweep}
macros["JnrTwoRecall"] = "%.1f" % by[2]["detect_recall"]
macros["JnrFourRecall"] = "%.1f" % by[4]["detect_recall"]
macros["JnrSixRecall"] = "%.1f" % by[6]["detect_recall"]
macros["JnrEightRecall"] = "%.1f" % by[8]["detect_recall"]
macros["JnrTwoAttr"] = "%.1f" % by[2]["attr_macro"]
macros["JnrFourAttr"] = "%.1f" % by[4]["attr_macro"]
macros["JnrEightAttr"] = "%.1f" % by[8]["attr_macro"]
macros["JnrTwoSP"] = "%.0f" % by[2]["per_class_recall"]["SP"]
macros["JnrTwoCW"] = "%.0f" % by[2]["per_class_recall"]["CW"]
macros["JnrTwoSW"] = "%.0f" % by[2]["per_class_recall"]["SW"]
macros["JnrFourSP"] = "%.0f" % by[4]["per_class_recall"]["SP"]
macros["JnrTwoBJ"] = "%.0f" % by[2]["per_class_recall"]["BJ"]
macros["JnrTwoPB"] = "%.0f" % by[2]["per_class_recall"]["PB"]

for key, tag in (("geo", "Geo"), ("ctx", "Ctx"), ("geo_ctx", "GeoCtx"), ("rf", "RF")):
    a = vals("sf", lambda r, key=key: r["masked"][key])
    b = vals("no_gate", lambda r, key=key: r["masked"][key])
    macros["MaskGap" + tag] = "%+.1f" % (a - b).mean()
    macros["MaskP" + tag] = "%.2f" % st.ttest_rel(a, b).pvalue
g = np.mean([np.load(os.path.join(res, "sf_run%d_preds.npz" % i))["gates"].mean(0) for i in range(n_runs)], axis=0)
macros["GateRF"], macros["GateGeo"], macros["GateCtx"] = ["%.2f" % x for x in g]

tt = "0.01"
macros["CalFlagOne"] = "%.1f" % cs(lambda r: r["targets"][tt]["flagged_fraction"] * 100)[0]
macros["CalUnflOne"] = "%.1f" % cs(lambda r: r["targets"][tt]["error_unflagged"] * 100)[0]
macros["CalFlOne"] = "%.0f" % cs(lambda r: np.nan if r["targets"][tt]["error_flagged"] is None else r["targets"][tt]["error_flagged"] * 100)[0]
macros["CalTopTwoFlOne"] = "%.0f" % cs(lambda r: np.nan if r["targets"][tt]["true_in_top2_flagged"] is None else r["targets"][tt]["true_in_top2_flagged"] * 100)[0]
macros["LedgerRateMin"] = "%.3f" % min(r["median_ms"] for r in led["rates"])
macros["LedgerRateMax"] = "%.3f" % max(r["median_ms"] for r in led["rates"])
macros["LedgerRateP"] = "%.2f" % max(r["p95_ms"] for r in led["rates"])
for k, nm in (("sf", "SF"), ("trans_s", "TransS"), ("rf", "RF"), ("svm", "SVM"), ("cnn", "CNN"), ("lstm", "LSTM"), ("sf_base", "SFBase")):
    macros[nm + "InjF"] = "%.1f" % ms(k, lambda r: r["injected"]["f1"])[0]
a, b = ms("sf", lambda r: r["masked"]["geo_ctx"])
macros["MaskedGatedGeoCtx"] = "%.1f" % a
a, b = ms("no_gate", lambda r: r["masked"]["geo_ctx"])
macros["MaskedConcatGeoCtx"] = "%.1f" % a
macros["ConcatDetFm"] = "%.1f" % ms("no_gate", lambda r: r["all"]["f1"])[0]
hist = json.load(open(os.path.join(res, "run0_history.json")))
hv = np.array(hist["sf"])
macros["SFBestEpoch"] = str(int(hv[:, 1].argmin()) + 1)

tri = json.load(open(os.path.join(res, "triage_rules.json")))
macros["TriSWAcc"] = "%.1f" % (tri["cw_vs_sw"]["test_acc"] * 100)
macros["TriSWVal"] = "%.1f" % (tri["cw_vs_sw"]["val_acc"] * 100)
macros["TriSPVal"] = "%.1f" % (tri["cw_vs_sp"]["val_acc"] * 100)
macros["TriSWn"] = str(tri["cw_vs_sw"]["n_test"])
macros["TriSWThr"] = "%.2f" % tri["cw_vs_sw"]["threshold"]
macros["TriSWMedCW"] = "%.2f" % tri["cw_vs_sw"]["median_cw"]
macros["TriSWMedSW"] = "%.2f" % tri["cw_vs_sw"]["median_sw"]
macros["TriSPAcc"] = "%.1f" % (tri["cw_vs_sp"]["test_acc"] * 100)
macros["TriSPn"] = str(tri["cw_vs_sp"]["n_test"])
macros["TriSPThr"] = "%.2f" % tri["cw_vs_sp"]["threshold"]
macros["TriSPMedCW"] = "%.2f" % tri["cw_vs_sp"]["median_cw"]
macros["TriSPMedSP"] = "%.2f" % tri["cw_vs_sp"]["median_sp"]

with open(os.path.join(out, "macros_auto.tex"), "w") as fh:
    for k, v in sorted(macros.items()):
        fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(len(macros), "macros")

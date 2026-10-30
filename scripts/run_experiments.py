import argparse
import copy
import json
import os
import pickle
import time

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import f1_score

from satforensics import classical, experiment as ex, inject, metrics
from satforensics.ledger import benchmark

torch.set_num_threads(int(os.environ.get('SF_THREADS', '1')))

SF_GRID = [{"d_model": d, "layers": l, "heads": 4, "d_ff": 2 * d, "lr": lr}
           for d in (32, 64) for l in (1, 2, 3) for lr in (1e-3, 3e-3)]
NET_GRID = [{"width": w, "lr": lr} for w in (32, 64) for lr in (1e-3, 3e-3)]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--topk", type=int, default=20)
    return p.parse_args()


def make_parts(data, split_seed, inj_seed, kind="group"):
    if kind == "group":
        idx = ex.group_split(data["norad"], split_seed)
    else:
        idx = ex.temporal_split(data["start"])
    parts = [ex.build_samples(data, ix, inj_seed + 17 * k) for k, ix in enumerate(idx)]
    stats = ex.standardise(parts)
    return parts, stats


def subset_metrics(part, score, logits):
    det = part["det"]
    clean = (det == 0)
    out = {"all": metrics.detection(det, score)}
    nat = (part["natural"] == 1) | clean
    inj = (part["attr"] >= 0) | clean
    out["natural"] = metrics.detection(det[nat], score[nat])
    out["injected"] = metrics.detection(det[inj], score[inj])
    m = part["attr"] >= 0
    out["attr"] = metrics.attribution(part["attr"][m], logits[m].argmax(1))
    out["confusion"] = metrics.confusion(part["attr"][m], logits[m].argmax(1)).tolist()
    return out


def val_objective(res, val):
    s = 1.0 / (1.0 + np.exp(-res["val"][0]))
    det = f1_score(val["det"], (s >= 0.5).astype(int), average="macro")
    m = val["attr"] >= 0
    att = f1_score(val["attr"][m], res["val"][1][m].argmax(1), average="macro") if m.any() else 0.0
    return 0.5 * (det + att)


def stage_search(a, data):
    parts, _ = make_parts(data, 0, 100)
    train, val, test = parts
    best = {}
    log = []
    for name, modalities, grid in (("sf", ("rf", "geo", "ctx"), SF_GRID), ("trans_s", ("rf",), SF_GRID)):
        top, top_cfg = -1.0, None
        for cfg in grid:
            res = ex.run_model("sf", cfg, train, val, test, 0, epochs=a.epochs, modalities=modalities)
            obj = val_objective(res, val)
            log.append({"model": name, "cfg": cfg, "val_objective": obj, "params": res["params"], "seconds": res["seconds"]})
            print(name, cfg, round(obj, 4), flush=True)
            if obj > top:
                top, top_cfg = obj, cfg
        best[name] = top_cfg
    for name, kind in (("cnn", "cnn"), ("lstm", "lstm")):
        top, top_cfg = -1.0, None
        for cfg in NET_GRID:
            res = ex.run_model(kind, cfg, train, val, test, 0, epochs=a.epochs)
            obj = val_objective(res, val)
            log.append({"model": name, "cfg": cfg, "val_objective": obj, "params": res["params"], "seconds": res["seconds"]})
            print(name, cfg, round(obj, 4), flush=True)
            if obj > top:
                top, top_cfg = obj, cfg
        best[name] = top_cfg
    with open(os.path.join(a.out, "configs.json"), "w") as fh:
        json.dump({"best": best, "epochs": a.epochs, "log": log, "sf_grid": SF_GRID, "net_grid": NET_GRID,
                   "svm_grid": classical.SVM_GRID, "rf_grid": classical.RF_GRID}, fh, indent=1)


def eval_sf(res, parts):
    val, test = parts[1], parts[2]
    r = subset_metrics(test, res["test"][0], res["test"][1])
    r["det_score"] = res["test"][0].tolist()
    return r


def stage_main(a, data):
    cfgs = json.load(open(os.path.join(a.out, "configs.json")))["best"]
    sfc, tsc = cfgs["sf"], cfgs["trans_s"]
    results = []
    keep = {}
    for run in range(a.runs):
        parts, stats = make_parts(data, run, 100 + run)
        train, val, test = parts
        row = {"run": run, "n": [len(p["det"]) for p in parts], "models": {}}
        t0 = time.time()
        variants = {
            "sf": ("sf", sfc, dict(modalities=("rf", "geo", "ctx"))),
            "sf_base": ("sf", dict(sfc, gated=False, mi=False), dict(modalities=("rf", "geo", "ctx"))),
            "trans_s": ("sf", tsc, dict(modalities=("rf",))),
            "cnn": ("cnn", cfgs["cnn"], {}),
            "lstm": ("lstm", cfgs["lstm"], {}),
            "no_pe": ("sf", dict(sfc, pe=False), dict(modalities=("rf", "geo", "ctx"))),
            "no_geo": ("sf", sfc, dict(modalities=("rf", "ctx"))),
            "no_ctx": ("sf", sfc, dict(modalities=("rf", "geo"))),
            "no_gate": ("sf", dict(sfc, gated=False), dict(modalities=("rf", "geo", "ctx"))),
            "no_mi": ("sf", dict(sfc, mi=False), dict(modalities=("rf", "geo", "ctx"))),
            "heads1": ("sf", dict(sfc, heads=1), dict(modalities=("rf", "geo", "ctx"))),
            "layers1": ("sf", dict(sfc, layers=1), dict(modalities=("rf", "geo", "ctx"))),
        }
        for name, (kind, cfg, kw) in variants.items():
            res = ex.run_model(kind, cfg, train, val, test, run, epochs=a.epochs, **kw)
            sc = 1.0 / (1.0 + np.exp(-res["test"][0]))
            lg = res["test"][1]
            r = subset_metrics(test, sc, lg)
            r["params"] = res["params"]
            r["seconds"] = res["seconds"]
            row["models"][name] = r
            if run == 0:
                keep[name] = {"score": sc, "logits": lg, "history": res["history"]}
            if name in ("sf", "no_gate"):
                drops = {"geo": ("geo",), "ctx": ("ctx",), "geo_ctx": ("geo", "ctx"), "rf": ("rf",)}
                row["models"][name]["masked"] = {}
                for k, d in drops.items():
                    s2, l2 = ex.masked_scores(res, test, d)
                    row["models"][name]["masked"][k] = subset_metrics(test, s2, l2)["all"]["f1"]
            if name == "sf":
                row["models"][name]["val_logits_temperature"] = None
                keep_sf = res
                if run == 0:
                    pickle.dump({"stats": stats}, open(os.path.join(a.out, "stats_run0.pkl"), "wb"))
                    torch.save(res["model"].state_dict(), os.path.join(a.out, "sf_run0.pt"))
                np.savez(os.path.join(a.out, "sf_run%d_preds.npz" % run),
                         val_logits=res["val"][1], val_attr=val["attr"], val_det=val["det"], val_score=res["val"][0],
                         test_logits=res["test"][1], test_attr=test["attr"], test_det=test["det"],
                         test_score=res["test"][0], test_natural=test["natural"], test_jnr=test["jnr"],
                         test_norad=test["norad"], gates=res["test"][2] if res["test"][2] is not None else np.zeros(1))
            print(run, name, round(r["all"]["f1"], 2), round(r["attr"]["macro"], 2), flush=True)
        X = classical.prepare(train, val, test)
        for name, grid in (("svm", classical.SVM_GRID), ("rf", classical.RF_GRID)):
            det, att, cfg, _ = classical.fit_search(name, X, train, val, grid, run)
            sc, ap = classical.scores(det, att, X["test"])
            lg = np.eye(ex.N_CLASSES)[ap]
            r = subset_metrics(test, sc, lg)
            r["cfg"] = {k: str(v) for k, v in cfg.items()}
            row["models"][name] = r
            if run == 0:
                keep[name] = {"score": sc, "logits": lg}
            print(run, name, round(r["all"]["f1"], 2), round(r["attr"]["macro"], 2), flush=True)
        row["test_labels_run0"] = None
        row["seconds"] = time.time() - t0
        results.append(row)
        json.dump(results, open(os.path.join(a.out, "main_results.json"), "w"))
        if run == 0:
            np.savez(os.path.join(a.out, "run0_curves.npz"), det=test["det"], natural=test["natural"],
                     attr=test["attr"], jnr=test["jnr"],
                     **{"%s_score" % k: v["score"] for k, v in keep.items()},
                     **{"%s_logits" % k: v["logits"] for k, v in keep.items()})
            json.dump({k: v["history"] for k, v in keep.items() if "history" in v},
                      open(os.path.join(a.out, "run0_history.json"), "w"))


def stage_jnr(a, data):
    cfgs = json.load(open(os.path.join(a.out, "configs.json")))["best"]
    parts, stats = make_parts(data, 0, 100)
    train, val, test = parts
    res = ex.run_model("sf", cfgs["sf"], train, val, test, 0, epochs=a.epochs)
    idx = ex.group_split(data["norad"], 0)[2]
    out = []
    for jnr in (2, 4, 6, 8, 10, 12, 14, 16, 18):
        p = ex.build_samples(data, idx, 900 + jnr, fixed_jnr=float(jnr))
        ex.standardise([p], stats)
        s, l = ex.masked_scores(res, p)
        m = subset_metrics(p, s, l)
        inj = p["attr"] >= 0
        recall = float((s[inj] >= 0.5).mean() * 100)
        per = {}
        for k, name in enumerate(inject.ARCHETYPES):
            sel = p["attr"] == k
            per[name] = float((s[sel] >= 0.5).mean() * 100)
        out.append({"jnr": jnr, "detect_recall": recall, "attr_macro": m["attr"]["macro"], "f1": m["injected"]["f1"], "per_class_recall": per})
        print(jnr, out[-1], flush=True)
    json.dump(out, open(os.path.join(a.out, "jnr_sweep.json"), "w"))


def stage_cross(a, data):
    cfgs = json.load(open(os.path.join(a.out, "configs.json")))["best"]
    sats, counts = np.unique(data["norad"], return_counts=True)
    nat_by = {s: data["natural"][data["norad"] == s].mean() for s in sats}
    order = [s for s in sats[np.argsort(-counts)] if 0.0 < nat_by[s] < 1.0][:a.topk]
    out = []
    for k, sat in enumerate(order):
        test_idx = np.where(data["norad"] == sat)[0]
        rest = np.where(data["norad"] != sat)[0]
        sub = {key: (v[rest] if len(v) == len(data["norad"]) else v) for key, v in data.items()}
        tr_i, va_i, _ = ex.group_split(sub["norad"], 500 + k, fracs=(0.85, 0.15, 0.0))
        tr_i, va_i = rest[tr_i], rest[va_i]
        parts = [ex.build_samples(data, ix, 300 + k + 11 * j) for j, ix in enumerate((tr_i, va_i, test_idx))]
        ex.standardise(parts)
        res = ex.run_model("sf", cfgs["sf"], parts[0], parts[1], parts[2], k, epochs=a.epochs)
        sc = 1.0 / (1.0 + np.exp(-res["test"][0]))
        m = subset_metrics(parts[2], sc, res["test"][1])
        out.append({"norad": int(sat), "n": int(len(test_idx)), "f1": m["all"]["f1"], "attr": m["attr"]["macro"],
                    "natural_f1": m["natural"]["f1"]})
        print(out[-1], flush=True)
        json.dump(out, open(os.path.join(a.out, "cross_satellite.json"), "w"))


def stage_temporal(a, data):
    cfgs = json.load(open(os.path.join(a.out, "configs.json")))["best"]
    parts, _ = make_parts(data, 0, 100, kind="temporal")
    train, val, test = parts
    out = {}
    for name, kind, cfg, kw in (("sf", "sf", cfgs["sf"], {}), ("trans_s", "sf", cfgs["trans_s"], dict(modalities=("rf",))),
                                ("lstm", "lstm", cfgs["lstm"], {})):
        res = ex.run_model(kind, cfg, train, val, test, 0, epochs=a.epochs, **kw)
        sc = 1.0 / (1.0 + np.exp(-res["test"][0]))
        out[name] = subset_metrics(test, sc, res["test"][1])
        print(name, round(out[name]["all"]["f1"], 2), flush=True)
    json.dump(out, open(os.path.join(a.out, "temporal.json"), "w"))


def stage_station(a, data):
    cfgs = json.load(open(os.path.join(a.out, "configs.json")))["best"]
    idx = ex.group_split(data["station"], 0)
    parts = [ex.build_samples(data, ix, 100 + 17 * k) for k, ix in enumerate(idx)]
    ex.standardise(parts)
    train, val, test = parts
    out = {"n": [len(p["det"]) for p in parts], "stations": [int(len(np.unique(data["station"][ix]))) for ix in idx]}
    for name, kind, cfg, kw in (("sf", "sf", cfgs["sf"], {}), ("trans_s", "sf", cfgs["trans_s"], dict(modalities=("rf",))),
                                ("lstm", "lstm", cfgs["lstm"], {})):
        res = ex.run_model(kind, cfg, train, val, test, 0, epochs=a.epochs, **kw)
        sc = 1.0 / (1.0 + np.exp(-res["test"][0]))
        out[name] = subset_metrics(test, sc, res["test"][1])
        print(name, round(out[name]["all"]["f1"], 2), flush=True)
    json.dump(out, open(os.path.join(a.out, "station_split.json"), "w"))


def stage_calibration(a, data):
    rows = []
    for run in range(a.runs):
        z = np.load(os.path.join(a.out, "sf_run%d_preds.npz" % run))
        vm = z["val_attr"] >= 0
        tm = z["test_attr"] >= 0
        vl, tl = z["val_logits"][vm], z["test_logits"][tm]
        va, ta = z["val_attr"][vm], z["test_attr"][tm]
        best_t, best_nll = 1.0, 1e9
        for t in np.linspace(0.3, 4.0, 75):
            p = softmax(vl / t, axis=1)
            nll = -np.log(np.maximum(p[np.arange(len(va)), va], 1e-12)).mean()
            if nll < best_nll:
                best_t, best_nll = t, nll

        def ece(p, y):
            conf, pred = p.max(1), p.argmax(1)
            bins = np.linspace(0, 1, 11)
            e = 0.0
            for lo, hi in zip(bins[:-1], bins[1:]):
                s = (conf > lo) & (conf <= hi)
                if s.any():
                    e += s.mean() * abs((pred[s] == y[s]).mean() - conf[s].mean())
            return e

        def stats(p):
            srt = np.sort(p, axis=1)[:, ::-1]
            margin = srt[:, 0] - srt[:, 1]
            ent = -(p * np.log(np.maximum(p, 1e-12))).sum(1) / np.log(p.shape[1])
            return margin, ent

        pv, pt = softmax(vl / best_t, axis=1), softmax(tl / best_t, axis=1)
        mv, ev = stats(pv)
        mt, et = stats(pt)
        err_v = pv.argmax(1) != va
        err_t = pt.argmax(1) != ta
        top2 = np.argsort(-pt, axis=1)[:, :2]
        in_top2 = np.array([ta[i] in top2[i] for i in range(len(ta))])
        row = {"run": run, "temperature": float(best_t), "ece_before": float(ece(softmax(tl, axis=1), ta)),
               "ece_after": float(ece(pt, ta)), "error_all": float(err_t.mean()),
               "true_in_top2_all": float(in_top2.mean()), "targets": {}}
        for target in (0.05, 0.02, 0.01):
            chosen, best_cov = (1.0, 0.0), -1.0
            for tm_ in np.linspace(0.0, 0.99, 100):
                for te_ in np.linspace(1.0, 0.05, 20):
                    keep = (mv >= tm_) & (ev <= te_)
                    if keep.sum() >= 20 and err_v[keep].mean() <= target and keep.mean() > best_cov:
                        chosen, best_cov = (float(tm_), float(te_)), float(keep.mean())
            flag = ~((mt >= chosen[0]) & (et <= chosen[1]))
            row["targets"][str(target)] = {
                "tau_margin": chosen[0], "tau_entropy": chosen[1], "val_coverage": best_cov,
                "flagged_fraction": float(flag.mean()),
                "error_unflagged": float(err_t[~flag].mean()) if (~flag).any() else None,
                "error_flagged": float(err_t[flag].mean()) if flag.any() else None,
                "true_in_top2_flagged": float(in_top2[flag].mean()) if flag.any() else None,
            }
        rows.append(row)
    json.dump(rows, open(os.path.join(a.out, "calibration.json"), "w"), indent=1)
    print(json.dumps(rows[0], indent=1))


def stage_ledger(a, data):
    base = benchmark()
    rates = []
    from satforensics.ledger import ForensicLedger, make_event
    for rate in (1, 10, 100, 500):
        n = int(max(30, min(3 * rate, 1500)))
        led = ForensicLedger()
        lat = []
        c0, w0 = time.process_time(), time.perf_counter()
        interval = 1.0 / rate
        nxt = time.perf_counter()
        for i in range(n):
            nxt += interval
            while time.perf_counter() < nxt:
                time.sleep(max(0.0, min(0.0005, nxt - time.perf_counter())))
            ev = make_event(i)
            t = time.perf_counter()
            led.append(ev)
            lat.append((time.perf_counter() - t) * 1000.0)
        cpu = (time.process_time() - c0) / (time.perf_counter() - w0) * 100.0
        lat.sort()
        rates.append({"rate": rate, "median_ms": float(np.median(lat)), "p95_ms": lat[int(0.95 * len(lat))], "cpu_pct": cpu})
    json.dump({"benchmark": base, "rates": rates}, open(os.path.join(a.out, "ledger.json"), "w"), indent=1)
    print(json.dumps({"benchmark": base, "rates": rates}, indent=1))


def stage_latency(a, data):
    cfgs = json.load(open(os.path.join(a.out, "configs.json")))["best"]
    parts, _ = make_parts(data, 0, 100)
    train, val, test = parts
    out = {}
    from torch.utils.flop_counter import FlopCounterMode
    for name, kind, cfg, kw in (("sf", "sf", cfgs["sf"], {}), ("sf_base", "sf", dict(cfgs["sf"], gated=False, mi=False), {}),
                                ("trans_s", "sf", cfgs["trans_s"], dict(modalities=("rf",))),
                                ("cnn", "cnn", cfgs["cnn"], {}), ("lstm", "lstm", cfgs["lstm"], {})):
        res = ex.run_model(kind, cfg, train, val, test, 0, epochs=1, **kw)
        model, feed, tt = res["model"], res["feed"], res["tensors"]["test"]
        model.eval()
        one = torch.arange(1)
        with torch.no_grad():
            for _ in range(20):
                model(feed(tt, one))
            t0 = time.perf_counter()
            for i in range(300):
                model(feed(tt, torch.tensor([i % len(test["det"])])))
            ms = (time.perf_counter() - t0) / 300 * 1000.0
            fc = FlopCounterMode(display=False)
            with fc:
                model(feed(tt, one))
        flops = fc.get_total_flops()
        if kind == "lstm":
            h = cfg["width"]
            d_in = sum(tt[m].shape[-1] for m in ("rf", "geo", "ctx"))
            steps = tt["rf"].shape[1]
            flops += steps * 2 * 4 * h * (d_in + h + h + h) + 2 * h * (1 + ex.N_CLASSES)
        out[name] = {"params": res["params"], "latency_ms": ms, "mflops": flops / 1e6,
                     "size_mb": res["params"] * 4 / 1e6}
        print(name, out[name], flush=True)
    mpath = os.path.join(a.out, "main_results.json")
    if os.path.exists(mpath):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        mr = json.load(open(mpath))[0]["models"]
        X = classical.prepare(train, val, test)
        am = train["attr"] >= 0

        def num(v):
            return v if v == "scale" else float(v)

        for name in ("svm", "rf"):
            c = mr[name]["cfg"]
            if name == "svm":
                det = SVC(kernel="rbf", C=float(c["C"]), gamma=num(c["gamma"]))
                att = SVC(kernel="rbf", C=float(c["C"]), gamma=num(c["gamma"]))
            else:
                depth = None if c["depth"] == "None" else int(c["depth"])
                det = RandomForestClassifier(200, max_depth=depth, min_samples_leaf=int(c["leaf"]), n_jobs=1, random_state=0)
                att = RandomForestClassifier(200, max_depth=depth, min_samples_leaf=int(c["leaf"]), n_jobs=1, random_state=0)
            det.fit(X["train"], train["det"])
            att.fit(X["train"][am], train["attr"][am])
            row = X["test"][:1]
            for _ in range(10):
                det.predict(row)
                att.predict(row)
            t0 = time.perf_counter()
            for i in range(200):
                r = X["test"][i % len(X["test"])][None]
                det.predict(r)
                att.predict(r)
            ms = (time.perf_counter() - t0) / 200 * 1000.0
            size = len(pickle.dumps(det)) + len(pickle.dumps(att))
            out[name] = {"params": None, "latency_ms": ms, "mflops": None, "size_mb": size / 1e6}
            print(name, out[name], flush=True)
    json.dump(out, open(os.path.join(a.out, "latency.json"), "w"), indent=1)


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    data = ex.load(a.data)
    globals()["stage_" + a.stage](a, data)


if __name__ == "__main__":
    main()

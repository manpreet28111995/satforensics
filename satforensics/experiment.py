import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from satforensics import inject
from satforensics.features import rf_features
from satforensics.model import CNN1D, LSTMNet, SatForensics, info_nce

N_CLASSES = len(inject.ARCHETYPES)


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def group_split(norad, seed, fracs=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    sats, counts = np.unique(norad, return_counts=True)
    order = rng.permutation(len(sats))
    total = counts.sum()
    targets = np.cumsum(np.array(fracs)) * total
    assign = {}
    running = 0
    for i in order:
        part = int(np.searchsorted(targets, running + counts[i] / 2.0))
        assign[sats[i]] = min(part, 2)
        running += counts[i]
    part = np.array([assign[s] for s in norad])
    return [np.where(part == k)[0] for k in range(3)]


def temporal_split(start, fracs=(0.70, 0.15, 0.15)):
    order = np.argsort(start)
    n = len(order)
    a, b = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
    return [order[:a], order[a:b], order[b:]]


def build_samples(data, idx, seed, p_inject=0.5, jnr_range=(2.0, 18.0), fixed_jnr=None):
    rng = np.random.default_rng(seed)
    level = data["level"][idx].astype(np.float32).copy()
    natural = data["natural"][idx]
    n = len(idx)
    attr = np.full(n, -1, dtype=np.int64)
    jnr = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if natural[i] == 0 and rng.random() < p_inject:
            k = int(rng.integers(0, N_CLASSES))
            j = fixed_jnr if fixed_jnr is not None else inject.draw_jnr(rng, *jnr_range)
            level[i] = inject.apply(level[i], inject.ARCHETYPES[k], j, rng, data["dshape"][idx][i])
            attr[i] = k
            jnr[i] = j
    det = ((natural == 1) | (attr >= 0)).astype(np.int64)
    return {
        "rf": rf_features(level),
        "geo": data["geo"][idx].astype(np.float32),
        "ctx": data["ctx"][idx].astype(np.float32),
        "det": det, "attr": attr, "natural": natural.astype(np.int64), "jnr": jnr,
        "norad": data["norad"][idx], "idx": idx,
    }


def standardise(parts, stats=None):
    if stats is None:
        stats = {}
        tr = parts[0]
        for key in ("rf", "geo"):
            flat = tr[key].reshape(-1, tr[key].shape[-1])
            stats[key] = (flat.mean(0), flat.std(0) + 1e-6)
        stats["ctx"] = (tr["ctx"].mean(0), tr["ctx"].std(0) + 1e-6)
    for p in parts:
        for key in ("rf", "geo", "ctx"):
            p[key] = ((p[key] - stats[key][0]) / stats[key][1]).astype(np.float32)
    return stats


def tensors(part, device="cpu"):
    t = part["rf"].shape[1]
    ctx = np.repeat(part["ctx"][:, None, :], t, axis=1)
    return {
        "rf": torch.from_numpy(part["rf"]), "geo": torch.from_numpy(part["geo"]), "ctx": torch.from_numpy(ctx),
        "det": torch.from_numpy(part["det"]).float(), "attr": torch.from_numpy(part["attr"]),
    }


def select(tt, names):
    return [tt[n] for n in names]


def aggregate(part):
    seqs = np.concatenate([part["rf"], part["geo"]], axis=2)
    agg = np.concatenate([seqs.mean(1), seqs.std(1), seqs.min(1), seqs.max(1), part["ctx"]], axis=1)
    return agg


def batches(n, size, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield order[i:i + size]


def run_model(kind, cfg, train, val, test, seed, epochs=30, batch=128, modalities=("rf", "geo", "ctx"),
              lam1=0.5, lam2=0.1, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tt = {k: tensors(p) for k, p in (("train", train), ("val", val), ("test", test))}
    dims = {"rf": tt["train"]["rf"].shape[-1], "geo": tt["train"]["geo"].shape[-1], "ctx": tt["train"]["ctx"].shape[-1]}
    use_mi = cfg.get("mi", True)
    if kind == "sf":
        model = SatForensics([dims[m] for m in modalities], d_model=cfg["d_model"], layers=cfg["layers"],
                             heads=cfg["heads"], d_ff=cfg["d_ff"], dropout=cfg.get("dropout", 0.1),
                             gated=cfg.get("gated", True), use_pe=cfg.get("pe", True))
        feed = lambda d, ix: [d[m][ix] for m in modalities]
    else:
        d_in = sum(dims[m] for m in modalities)
        model = CNN1D(d_in, N_CLASSES, cfg["width"]) if kind == "cnn" else LSTMNet(d_in, N_CLASSES, cfg["width"])
        feed = lambda d, ix: torch.cat([d[m][ix] for m in modalities], dim=-1)
        use_mi = False
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_pos = float(tt["train"]["det"].sum())
    pos_w = torch.tensor((len(tt["train"]["det"]) - n_pos) / max(n_pos, 1.0))
    history, best, best_state = [], float("inf"), None

    def forward(d, ix):
        out = model(feed(d, ix))
        return out

    def predict(d):
        model.eval()
        scores, logits, gates = [], [], []
        with torch.no_grad():
            for ix in batches(len(d["det"]), 256):
                ix = torch.from_numpy(ix)
                out = forward(d, ix)
                scores.append(out[0])
                logits.append(out[1])
                if kind == "sf" and out[4] is not None:
                    gates.append(out[4].mean(1))
        g = torch.cat(gates).numpy() if gates else None
        return torch.cat(scores).numpy(), torch.cat(logits).numpy(), g

    def loss_on(d, ix, train_mode):
        out = forward(d, ix)
        det_l = F.binary_cross_entropy_with_logits(out[0], d["det"][ix], pos_weight=pos_w)
        a = d["attr"][ix]
        m = a >= 0
        attr_l = F.cross_entropy(out[1][m], a[m]) if m.any() else torch.zeros(())
        total = det_l + lam1 * attr_l
        mi = torch.zeros(())
        if kind == "sf" and use_mi and train_mode and len(ix) > 1:
            mi = info_nce(out[2], out[3])
            total = total + lam2 * mi
        return total, det_l, attr_l, mi

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for ix in batches(len(train["det"]), batch, rng):
            ix = torch.from_numpy(ix)
            opt.zero_grad()
            total, _, _, _ = loss_on(tt["train"], ix, True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(total) * len(ix)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for ix in batches(len(val["det"]), 256):
                ix = torch.from_numpy(ix)
                vt, _, _, _ = loss_on(tt["val"], ix, False)
                vl += float(vt) * len(ix)
        vl /= len(val["det"])
        history.append((tr_loss / len(train["det"]), vl))
        if vl < best:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(ep, round(history[-1][0], 4), round(vl, 4), flush=True)
    model.load_state_dict(best_state)
    fit_seconds = time.time() - t0
    res = {"val": predict(tt["val"]), "test": predict(tt["test"]), "history": history, "seconds": fit_seconds,
           "params": int(sum(p.numel() for p in model.parameters()))}
    res["model"] = model
    res["feed"] = feed
    res["tensors"] = tt
    return res


def masked_scores(res, part, drop=()):
    model = res["model"]
    feed = res["feed"]
    t = tensors(part)
    for name in drop:
        t[name] = torch.zeros_like(t[name])
    model.eval()
    scores, logits = [], []
    with torch.no_grad():
        for ix in batches(len(part["det"]), 256):
            ix = torch.from_numpy(ix)
            out = model(feed(t, ix))
            scores.append(out[0])
            scores_idx = out[1]
            logits.append(scores_idx)
    return torch.sigmoid(torch.cat(scores)).numpy(), torch.cat(logits).numpy()

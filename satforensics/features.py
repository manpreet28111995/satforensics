import json
import math

import numpy as np

from satforensics.inject import BAND_HI, BAND_LO, to_power
from satforensics.orbit import C_KM_S, frequency_hz

RF_DIM = 21
GEO_DIM = 8
MODES = ["FM", "FSK", "GFSK", "GMSK", "BPSK", "AFSK", "CW", "LoRa", "SSTV", "USB"]
TYPES = ["Transmitter", "Transceiver", "Transponder"]
CTX_DIM = 27


def rf_features(level):
    n, t, f = level.shape
    x = level[:, :, BAND_LO:BAND_HI] / 255.0
    p = to_power(level[:, :, BAND_LO:BAND_HI])
    med = np.median(x, axis=2)
    mean = x.mean(axis=2)
    std = x.std(axis=2)
    q = np.quantile(x, [0.10, 0.90, 0.99], axis=2)
    mx = x.max(axis=2)
    peak = mx - med
    argmax = x.argmax(axis=2) / float(x.shape[2] - 1)
    geo = np.exp(np.mean(np.log(np.maximum(p, 1e-30)), axis=2))
    flat = geo / np.maximum(p.mean(axis=2), 1e-30)
    occ10 = (x > (med[:, :, None] + 0.10)).mean(axis=2)
    occ20 = (x > (med[:, :, None] + 0.20)).mean(axis=2)
    bands = np.stack([b.mean(axis=2) for b in np.array_split(x, 8, axis=2)], axis=2)
    delta = np.concatenate([np.zeros((n, 1)), np.diff(med, axis=1)], axis=1)
    cols = [mean, med, std, q[0], q[1], q[2], mx, peak, argmax, flat, occ10, occ20]
    out = np.stack(cols, axis=2)
    out = np.concatenate([out, bands, delta[:, :, None]], axis=2)
    return out.astype(np.float32)


def geo_features(track, freq_hz):
    elev, saz, caz, rng, rr, tt = [track[:, i] for i in range(6)]
    doppler = -rr / C_KM_S * freq_hz / 1000.0
    scale = max(np.abs(doppler).max(), 1e-6)
    rate = np.gradient(doppler, tt) if tt[-1] > 0 else np.zeros_like(doppler)
    frac = tt / max(tt[-1], 1e-6)
    cols = [elev / 90.0, saz, caz, rng / 3000.0, rr / 8.0, doppler / scale, np.clip(rate / 0.5, -3, 3), frac]
    return np.stack(cols, axis=1).astype(np.float32), (doppler / scale).astype(np.float32)


def tle_elements(rec):
    l2 = rec["tle2"]
    inc = float(l2[8:16])
    ecc = float("0." + l2[26:33].strip())
    mm = float(l2[52:63])
    l1 = rec["tle1"]
    raw = l1[53:61].strip()
    try:
        mant = float(raw[:-2]) / 1e5
        exp = int(raw[-2:])
        bstar = mant * (10.0 ** exp)
    except ValueError:
        bstar = 0.0
    return inc, ecc, mm, bstar


def context_vector(rec, span_s):
    freq = frequency_hz(rec)
    baud = float(rec.get("transmitter_baud") or 0.0)
    mode = (rec.get("transmitter_mode") or "").strip()
    mode_oh = [1.0 if mode == m else 0.0 for m in MODES]
    other = 1.0 if mode not in MODES else 0.0
    ttype = rec.get("transmitter_type") or ""
    type_oh = [1.0 if ttype == k else 0.0 for k in TYPES]
    inc, ecc, mm, bstar = tle_elements(rec)
    lng = math.radians(rec["station_lng"])
    drift = 1.0 if rec.get("transmitter_downlink_drift") else 0.0
    vec = [
        math.log10(max(freq, 1.0)) / 10.0,
        math.log10(baud + 1.0) / 6.0,
        1.0 if rec.get("transmitter_invert") else 0.0,
        drift,
    ] + mode_oh + [other] + type_oh + [
        (rec["station_alt"] or 0.0) / 1000.0,
        rec["station_lat"] / 90.0,
        math.sin(lng),
        math.cos(lng),
        (rec["max_altitude"] or 0.0) / 90.0,
        span_s / 900.0,
        inc / 180.0,
        mm / 16.0,
        math.log10(abs(bstar) + 1e-9) / 9.0,
    ]
    return np.array(vec, dtype=np.float32)


def load_meta(path):
    with open(path) as fh:
        return {json.loads(l)["id"]: json.loads(l) for l in fh}

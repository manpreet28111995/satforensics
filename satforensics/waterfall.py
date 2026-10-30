import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from matplotlib import colormaps

LUT = (colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)
TREE = cKDTree(LUT)
T_BINS = 64
F_BINS = 128


def longest_run(flags):
    best, cur, start, best_start = 0, 0, 0, 0
    for i, f in enumerate(flags):
        if f:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best_start, best_start + best


def pool(arr, rows, cols):
    h, w = arr.shape
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    out = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        band = arr[ys[i]:max(ys[i + 1], ys[i] + 1)]
        for j in range(cols):
            out[i, j] = band[:, xs[j]:max(xs[j + 1], xs[j] + 1)].mean()
    return out


def extract(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    sub = rgb[::4, ::4].astype(np.float32)
    dist, _ = TREE.query(sub.reshape(-1, 3))
    mask = (dist < 10.0).reshape(sub.shape[:2])
    c0, c1 = longest_run(mask.mean(axis=0) > 0.6)
    if c1 - c0 < 40:
        return None
    r0, r1 = longest_run(mask[:, c0:c1].mean(axis=1) > 0.6)
    if r1 - r0 < 40:
        return None
    crop = rgb[r0 * 4:r1 * 4:2, c0 * 4:c1 * 4:2].astype(np.float32)
    _, idx = TREE.query(crop.reshape(-1, 3))
    level = idx.reshape(crop.shape[:2]).astype(np.float32)
    level = level[::-1]
    return pool(level, T_BINS, F_BINS).astype(np.uint8)

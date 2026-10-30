import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from style import setup

setup()
out = sys.argv[1]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 64)
ax.axis("off")


def box(x, y, w, h, text, fc, ec="#333333", size=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333333", shrinkA=0, shrinkB=0))


c_in, c_proc, c_fuse, c_enc, c_head, c_led = "#E3ECF7", "#D6EBE6", "#FCE8CF", "#F7D9D9", "#EDE3F3", "#DADFE5"
w, h = 27, 8.5
xs = [3, 36.5, 70]
ys_in = 54
for x, t in zip(xs, ["RF waterfall features\n21 per time step", "Orbital geometry (SGP4)\n8 per time step", "Link context\n27 static values"]):
    box(x, ys_in, w, h, t, c_in)
for x in xs:
    box(x, 41, w, 7, "LayerNorm + projection\nto $d_{\\mathrm{model}}$", c_proc, size=7.8)
    arrow(x + w / 2, ys_in, x + w / 2, 48)
box(10, 28.5, 80, 7.2, "Gated multi-modal fusion (per-step sigmoid gates)", c_fuse)
for x in xs:
    arrow(x + w / 2, 41, x + w / 2, 35.7)
box(10, 18.5, 80, 7.2, "CLS token + sinusoidal positional encoding + Transformer encoder", c_enc)
arrow(50, 28.5, 50, 25.7)
hx = [3, 36.5, 70]
names = ["Detection head\nanomaly score", "Attribution head\ncalibrated posterior,\nmargin, ambiguity flag", "Forensic logger\nSHA-256 hash chain"]
for x, t in zip(hx, names):
    box(x, 7.2, w, 8.5, t, c_head, size=7.8)
    arrow(x + w / 2 if x != 36.5 else 50, 18.5, x + w / 2, 15)
box(3, 0.4, 94, 4.2, "Tamper-evident evidence ledger", c_led, size=8.2, bold=True)
for x in hx:
    arrow(x + w / 2, 7.2, x + w / 2, 4.6)
fig.savefig(out, bbox_inches="tight", pad_inches=0.05)

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from style import ARCH, LABELS, PALETTE, setup

setup()
res, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
main = json.load(open(os.path.join(res, "main_results.json")))
curves = np.load(os.path.join(res, "run0_curves.npz"))
order = ["sf", "sf_base", "trans_s", "cnn", "lstm", "svm", "rf"]


def stat(model, getter):
    v = np.array([getter(r["models"][model]) for r in main], dtype=float)
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0


fig, ax = plt.subplots(figsize=(5.2, 3.2))
x = np.arange(len(order))
bw = 0.38
for k, (key, lab, col) in enumerate((("all", "All test samples", "#0B3C5D"), ("natural", "Natural labels only", "#D98E04"))):
    m = [stat(o, lambda r, key=key: r[key]["f1"])[0] for o in order]
    s = [stat(o, lambda r, key=key: r[key]["f1"])[1] for o in order]
    ax.bar(x + (k - 0.5) * bw, m, bw, yerr=s, capsize=2, color=col, label=lab, error_kw={"lw": 0.8})
ax.set_xticks(x)
ax.set_xticklabels([LABELS[o] for o in order], rotation=25, ha="right")
ax.set_ylabel("Detection macro-F1 (%)")
ax.set_ylim(0, 100)
ax.grid(axis="x", visible=False)
ax.legend(loc="lower center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.savefig(os.path.join(out, "Fig4_F1_Score_Comparison.pdf"), bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

det = curves["det"]
nat = (curves["natural"] == 1) | (det == 0)
for name, fn, xl, yl, title in (("Fig5_ROC_Curves.pdf", "roc", "False positive rate", "True positive rate", "roc"),
                                ("Fig6_Precision_Recall_Curves.pdf", "pr", "Recall", "Precision", "pr")):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, mask, ttl in ((axs[0], np.ones_like(det, dtype=bool), "All test samples"), (axs[1], nat, "Natural labels only")):
        for o in order:
            sc = curves["%s_score" % o]
            if fn == "roc":
                f, t, _ = roc_curve(det[mask], sc[mask])
                ax.plot(f, t, color=PALETTE[o], lw=1.4 if o == "sf" else 1.0, label=LABELS[o])
            else:
                p, r, _ = precision_recall_curve(det[mask], sc[mask])
                ax.plot(r, p, color=PALETTE[o], lw=1.4 if o == "sf" else 1.0, label=LABELS[o])
        if fn == "roc":
            ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
        ax.set_xlabel(xl)
        ax.set_title(ttl)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    axs[0].set_ylabel(yl)
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=7, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(out, name), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 3.2))
models = ["sf", "sf_base", "trans_s", "cnn", "lstm", "svm", "rf"]
bw = 0.8 / len(models)
x = np.arange(len(ARCH))
for k, o in enumerate(models):
    m = [stat(o, lambda r, c=c: r["attr"]["per_class"][c])[0] for c in range(len(ARCH))]
    s = [stat(o, lambda r, c=c: r["attr"]["per_class"][c])[1] for c in range(len(ARCH))]
    ax.bar(x + (k - len(models) / 2 + 0.5) * bw, m, bw, yerr=s, capsize=1.5, color=PALETTE[o], label=LABELS[o], error_kw={"lw": 0.7})
ax.set_xticks(x)
ax.set_xticklabels(ARCH)
ax.set_ylabel("Attribution F1 (%)")
ax.set_ylim(0, 105)
ax.grid(axis="x", visible=False)
ax.legend(loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
fig.savefig(os.path.join(out, "Fig7_Attribution_Per_Archetype.pdf"), bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

cm = np.mean([np.array(r["models"]["sf"]["confusion"]) for r in main], axis=0)
fig, ax = plt.subplots(figsize=(3.9, 3.5))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(ARCH)
ax.set_yticklabels(ARCH)
ax.set_xlabel("Predicted archetype")
ax.set_ylabel("True archetype")
ax.grid(False)
for i in range(5):
    for j in range(5):
        ax.text(j, i, "%.1f" % cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 55 else "#222222")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(os.path.join(out, "Fig8_Attribution_Confusion_Matrix.pdf"), bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

hist = json.load(open(os.path.join(res, "run0_history.json")))
fig, ax = plt.subplots(figsize=(5.2, 3.1))
for key, ls in (("sf", "-"), ("sf_base", "--"), ("trans_s", ":")):
    h = np.array(hist[key])
    ep = np.arange(1, len(h) + 1)
    ax.plot(ep, h[:, 0], color=PALETTE[key], ls=ls, lw=1.2, label="%s train" % LABELS[key])
    ax.plot(ep, h[:, 1], color=PALETTE[key], ls=ls, lw=1.2, alpha=0.55, label="%s validation" % LABELS[key])
ax.set_xlabel("Epoch")
ax.set_ylabel("Total training objective")
ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))
fig.savefig(os.path.join(out, "Fig9_Training_Convergence.pdf"), bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

cross = json.load(open(os.path.join(res, "cross_satellite.json")))
cross = sorted(cross, key=lambda r: r["f1"])
fig, ax = plt.subplots(figsize=(5.6, 0.22 * len(cross) + 1.0))
y = np.arange(len(cross))
ax.barh(y, [r["f1"] for r in cross], color="#3E7CB1", height=0.7)
ax.set_yticks(y)
ax.set_yticklabels(["NORAD %d (n=%d)" % (r["norad"], r["n"]) for r in cross], fontsize=7)
mean = np.mean([r["f1"] for r in cross])
ax.axvline(mean, color="#A23B72", lw=1.2, ls="--")
ax.set_xlabel("Detection macro-F1 on the held-out satellite (%)")
ax.set_xlim(0, 100)
ax.grid(axis="y", visible=False)
ax.text(mean + 1.0, -0.9, "mean %.1f" % mean, color="#A23B72", fontsize=7.5, va="center")
ax.set_ylim(-1.5, len(cross) - 0.4)
fig.savefig(os.path.join(out, "Fig10_Cross_Satellite_Generalization.pdf"), bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

sweep = json.load(open(os.path.join(res, "jnr_sweep.json")))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
j = [r["jnr"] for r in sweep]
cols = ["#0B3C5D", "#D98E04", "#A23B72", "#2E8B57", "#6C757D"]
for k, a in enumerate(ARCH):
    axs[0].plot(j, [r["per_class_recall"][a] for r in sweep], marker="o", ms=3, lw=1.1, color=cols[k], label=a)
axs[0].set_xlabel("Jammer-to-noise ratio (dB)")
axs[0].set_ylabel("Detection recall (%)")
axs[0].set_ylim(0, 102)
axs[0].legend(frameon=False, ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(1.1, -0.2))
axs[1].plot(j, [r["attr_macro"] for r in sweep], marker="o", ms=3, lw=1.3, color="#0B3C5D")
axs[1].set_xlabel("Jammer-to-noise ratio (dB)")
axs[1].set_ylabel("Attribution macro-F1 (%)")
axs[1].set_ylim(0, 102)
fig.tight_layout()
fig.savefig(os.path.join(out, "Fig11_JNR_Sweep.pdf"), bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

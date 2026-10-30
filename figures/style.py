import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "sf": "#0B3C5D",
    "sf_base": "#3E7CB1",
    "trans_s": "#81A4CD",
    "cnn": "#D98E04",
    "lstm": "#A23B72",
    "svm": "#6C757D",
    "rf": "#2E8B57",
}
LABELS = {
    "sf": "SatForensics",
    "sf_base": "SF-Base",
    "trans_s": "Trans.-S",
    "cnn": "CNN",
    "lstm": "LSTM",
    "svm": "SVM",
    "rf": "Random Forest",
}
ARCH = ["BJ", "CW", "SW", "PB", "SP"]


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
    })

"""Plot example wafer maps for each labelled failure type.

Output: results/defect_classes.png — a grid with one row per class, four examples each.
Colour code: white = outside the wafer, grey = die that passed test, red = die that failed.
"""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / "data" / "wm811k.h5"
OUT = ROOT / "results" / "defect_classes.png"

CLASSES = ["none", "Edge-Ring", "Edge-Loc", "Center", "Loc",
           "Scratch", "Random", "Donut", "Near-full"]
N_EXAMPLES = 4
CMAP = ListedColormap(["white", "#d0d0d0", "#c62828"])


def main() -> None:
    with h5py.File(H5, "r") as f:
        labels = f["label"].asstr()[:]
        heights, widths = f["heights"][:], f["widths"][:]
        fig, axes = plt.subplots(len(CLASSES), N_EXAMPLES,
                                 figsize=(2.2 * N_EXAMPLES, 2.2 * len(CLASSES)))
        for r, cls in enumerate(CLASSES):
            # medium-sized maps render most legibly
            idx = np.where((labels == cls) & (heights >= 30) & (widths >= 30))[0]
            if len(idx) < N_EXAMPLES:
                idx = np.where(labels == cls)[0]
            for c, i in enumerate(idx[:N_EXAMPLES]):
                m = f["maps"][i].reshape(heights[i], widths[i])
                ax = axes[r, c]
                ax.imshow(m, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
                ax.set_xticks([]), ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(cls, fontsize=11, rotation=0,
                                  ha="right", va="center", labelpad=10)
    fig.suptitle("WM-811K failure types (grey = pass, red = fail)", y=0.995)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

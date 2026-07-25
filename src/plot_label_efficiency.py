"""Label-efficiency curve: macro-F1 vs label budget, scratch vs AE-pretrained.

Points = mean over 3 subsample seeds; error bars = min-max range.
100% endpoints come from the single-seed reference runs (weighted_sqrt, ae_finetune).
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TXT = ROOT / "results" / "label_efficiency.txt"
FIG = ROOT / "results" / "label_efficiency.png"

N_TRAIN = 120_888
ENDPOINTS = {"scratch": 0.8760, "ae": 0.8708}  # 100%-label reference runs


def main() -> None:
    runs: dict[tuple[str, float], list[float]] = {}
    for line in TXT.read_text().splitlines():
        m = re.search(r"frac=([\d.]+) seed=\d+ init=(\w+) macro_f1=([\d.]+)", line)
        if m:
            frac, init, f1 = float(m.group(1)), m.group(2), float(m.group(3))
            runs.setdefault((init, frac), []).append(f1)
    for init, ref in ENDPOINTS.items():
        runs[(init, 1.0)] = [ref]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for init, color, label in [("scratch", "#666666", "supervised from scratch"),
                               ("ae", "#c62828", "AE-pretrained on 638k unlabeled")]:
        fracs = sorted(f for i, f in runs if i == init)
        means = [np.mean(runs[(init, f)]) for f in fracs]
        lo = [np.min(runs[(init, f)]) for f in fracs]
        hi = [np.max(runs[(init, f)]) for f in fracs]
        x = [f * N_TRAIN for f in fracs]
        ax.errorbar(x, means, yerr=[np.subtract(means, lo), np.subtract(hi, means)],
                    marker="o", color=color, label=label, capsize=3)

    ax.set_xscale("log")
    ax.set_xlabel("labeled training wafers (log scale)")
    ax.set_ylabel("val macro-F1")
    ax.set_title("Value of 638k unlabeled maps vs label budget\n"
                 "(error bars: min–max over 3 subsample seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=120)
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()

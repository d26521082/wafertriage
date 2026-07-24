"""Reliability diagram + ECE for a trained checkpoint on the validation split.

Plumbing (model loading, batching, plot template) is provided.
The statistics — binning, per-bin averages, ECE — are TODO(宸霆).

Run:  uv run python src/calibration.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WaferDataset
from model import WaferCNN

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results" / "weighted_sqrt.pt"
FIG = ROOT / "results" / "reliability_weighted_sqrt.png"

N_BINS = 15  # bins over confidence in (0, 1]


@torch.no_grad()
def get_probs():
    """Run the checkpoint on val. Returns probs (N, 9) and true labels (N,)."""
    device = "cuda"
    model = WaferCNN().to(device)
    model.load_state_dict(torch.load(CKPT, weights_only=True))
    model.eval()
    loader = DataLoader(WaferDataset("val"), batch_size=512, num_workers=4)
    probs, targets = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def main() -> None:
    probs, targets = get_probs()
    conf = probs.max(axis=1)            # (N,) the model's confidence per sample
    pred = probs.argmax(axis=1)         # (N,) the predicted class
    correct = (pred == targets)         # (N,) bool: was the prediction right?
    print(f"val samples: {len(conf)}, overall accuracy: {correct.mean():.4f}")

    # ------------------------------------------------------------------
    # TODO(宸霆) 1 — 分桶統計
    #
    # 把 [0, 1] 切成 N_BINS 個等寬的桶。對每個桶 b:
    #   n_b    = 落在這個桶裡的樣本數      (用布林遮罩: (conf >= lo) & (conf < hi))
    #   conf_b = 桶內樣本的平均信心
    #   acc_b  = 桶內樣本的實際正確率      (correct 在遮罩內的平均)
    # 跳過空桶。把結果收集成三個 list: ns, mean_confs, accs
    #
    # 提示: np.linspace(0, 1, N_BINS + 1) 給你桶的邊界。
    #       最後一個桶記得含右端點 (conf <= 1)。
    # ------------------------------------------------------------------
    ns, mean_confs, accs = [], [], []
    edges = np.linspace(0, 1, N_BINS + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        is_last = hi == edges[-1]
        mask = (conf >= lo) & ((conf <= hi) if is_last else (conf < hi))
        if mask.sum() == 0: continue
        ns.append(mask.sum()); mean_confs.append(conf[mask].mean()); accs.append(correct[mask].mean())

    print(f"[check] binned {sum(ns):,} of {len(conf):,} samples")  # 驗算,暫時的

    # ------------------------------------------------------------------
    # TODO(宸霆) 2 — ECE
    #
    #   ECE = sum_b  (n_b / N) * |acc_b - conf_b|
    #
    # 一行到三行。印出來,格式: ECE: 0.0xxx
    # ------------------------------------------------------------------
    weights = np.array(ns) / len(conf)
    gaps = np.abs(np.array(accs) - np.array(mean_confs))
    ece = np.sum(weights * gaps)
    print(f"ECE: {ece:.4f}")

    # ------------------------------------------------------------------
    # 畫圖 — 樣板給你,TODO(宸霆) 3 是把你的統計量畫上去
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")

    ax.plot(mean_confs, accs, marker="o")

    ax.set_xlabel("confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Reliability — weighted_sqrt (ECE={ece:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG, dpi=120)
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()

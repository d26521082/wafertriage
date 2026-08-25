"""The one-shot test-set evaluation. Pre-registered protocol; run exactly once.

Model: weighted_sqrt (the supervised reference). Outputs: (1) macro-F1, accuracy,
per-class P/R/F1 on the 25,801-wafer test split; (2) the triage policy table at
C_miss in {30, 100, 300, 1000}. Whatever comes out goes in the report.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from dataset import CLASSES, WaferDataset
from model import WaferCNN
from calibration import CKPT
from policy import C_REV, C_INV

C_MISS_GRID = [30, 100, 300, 1000]


@torch.no_grad()
def get_test_probs():
    device = "cuda"
    model = WaferCNN().to(device)
    model.load_state_dict(torch.load(CKPT, weights_only=True))
    model.eval()
    loader = DataLoader(WaferDataset("test"), batch_size=512, num_workers=4)
    probs, targets = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def main() -> None:
    probs, targets = get_test_probs()
    preds = probs.argmax(axis=1)
    n = len(targets)
    print(f"TEST SET — {n:,} wafers, evaluated once, {np.datetime64('today')}")
    print(f"macro-F1:  {f1_score(targets, preds, average='macro'):.4f}")
    print(f"accuracy:  {np.mean(preds == targets):.4f}\n")

    p_, r_, f1_, support = precision_recall_fscore_support(
        targets, preds, labels=range(len(CLASSES)), zero_division=0)
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'F1':>10}{'n':>8}")
    for i, cls in enumerate(CLASSES):
        print(f"{cls:<12}{p_[i]:>10.3f}{r_[i]:>10.3f}{f1_[i]:>10.3f}{support[i]:>8,}")

    cm = confusion_matrix(targets, preds, labels=range(len(CLASSES)))
    np.savetxt("results/final_test_confusion.csv", cm, fmt="%d", delimiter=",",
               header=",".join(CLASSES))

    p_defect = 1.0 - probs[:, 0]
    defect = targets != 0
    print(f"\ntrue defect share: {defect.mean():.3%}\n")
    print(f"{'C_miss':>7}{'p1':>9} | {'clear':>7}{'review':>7}{'flag':>6} | "
          f"{'policy':>8}{'all-rev':>8} | {'missed@clear':>13}")
    for c_miss in C_MISS_GRID:
        p1 = min(1.0, C_REV / (c_miss - C_INV))
        p2 = (C_INV - C_REV) / C_INV
        clear = p_defect < p1
        flag = p_defect >= p2
        review = ~clear & ~flag
        cost = np.where(clear, np.where(defect, c_miss, 0.0),
               np.where(flag, C_INV, C_REV + np.where(defect, C_INV, 0.0)))
        cost_all_review = (C_REV + C_INV * defect).mean()
        missed = int((clear & defect).sum())
        print(f"{c_miss:>7}{p1:>9.4f} | {clear.mean():>6.1%}{review.mean():>7.1%}"
              f"{flag.mean():>6.1%} | {cost.mean():>8.3f}{cost_all_review:>8.3f} | "
              f"{missed:>5d} ({missed / max(clear.sum(), 1):.3%})")


if __name__ == "__main__":
    main()

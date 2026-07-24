"""FixMatch-style semi-supervised training on WM-811K.

Recipe:
  - labeled batch: weighted cross-entropy (sqrt weights), as in the supervised reference
  - unlabeled batch: weak view (identity) -> pseudo-label where max prob >= tau;
    strong view (Cutout occlusion) -> CE against those pseudo-labels
  - total loss = L_sup + lambda_u * L_unsup

Design choices that differ from stock FixMatch, both deliberate:
  - Strong augmentation is Cutout only — no rotations. The sqrt_aug experiment showed
    fab orientation carries real signal; occlusion perturbs without touching it.
  - Warm start from the supervised reference (results/weighted_sqrt.pt), so pseudo-labels
    start from a competent teacher instead of noise.

The known failure mode is pseudo-label collapse toward the majority class; the
per-epoch pseudo-label distribution is printed precisely to watch for it.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from dataset import CLASSES, H5, WaferDataset, to_tensor
from model import WaferCNN
from train_baseline import class_weights, evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

EPOCHS = 20
BATCH = 256
UNLAB_BATCH = 512
LR = 1e-4          # lower than supervised: we start from a trained model
TAU = 0.95
LAMBDA_U = 1.0
SEED = 20260723


class UnlabeledWafers(Dataset):
    """All wafers with no failureType label — the 638k-map pool."""

    def __init__(self):
        with h5py.File(H5, "r") as f:
            labels = f["label"].asstr()[:]
            keep = np.where(labels == "")[0]
            maps = f["maps"][:]         # one bulk read beats 638k fancy reads
            heights, widths = f["heights"][:], f["widths"][:]
        self.maps = maps[keep]
        self.heights, self.widths = heights[keep], widths[keep]

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, i: int) -> torch.Tensor:
        return to_tensor(self.maps[i], self.heights[i], self.widths[i])


def cutout(x: torch.Tensor, n_holes: int = 2, size_range=(8, 16)) -> torch.Tensor:
    """Occlude random squares (both channels). Perturbs without touching orientation."""
    x = x.clone()
    b, _, h, w = x.shape
    for _ in range(n_holes):
        s = torch.randint(size_range[0], size_range[1] + 1, (1,)).item()
        ys = torch.randint(0, h - s, (b,))
        xs = torch.randint(0, w - s, (b,))
        for j in range(b):
            x[j, :, ys[j]:ys[j] + s, xs[j]:xs[j] + s] = 0.0
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="semi_v1")
    ap.add_argument("--init", default=str(OUT / "weighted_sqrt.pt"))
    ap.add_argument("--weighted-unsup", action="store_true",
                    help="apply the sqrt class weights to the unsupervised CE too")
    ap.add_argument("--exclude-none", action="store_true",
                    help="never pseudo-label 'none': labeled none is already abundant, "
                         "and v1 showed none pseudo-labels drive distribution collapse")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda"

    train_ds, val_ds = WaferDataset("train"), WaferDataset("val")
    print("loading unlabeled pool (bulk h5 read, ~1 min) ...")
    unlab_ds = UnlabeledWafers()
    print(f"unlabeled pool: {len(unlab_ds):,} maps")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=6, pin_memory=True, persistent_workers=True,
                              drop_last=True)
    unlab_loader = DataLoader(unlab_ds, batch_size=UNLAB_BATCH, shuffle=True,
                              num_workers=6, pin_memory=True, persistent_workers=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, num_workers=4,
                            pin_memory=True, persistent_workers=True)

    model = WaferCNN().to(device)
    model.load_state_dict(torch.load(args.init, weights_only=True))
    print(f"warm start from {args.init}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    w = class_weights(train_ds.targets, "sqrt").to(device)
    sup_loss_fn = nn.CrossEntropyLoss(weight=w)

    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        unlab_iter = iter(unlab_loader)
        sup_sum = unsup_sum = 0.0
        pseudo_counts = torch.zeros(len(CLASSES), dtype=torch.long)
        n_pseudo = n_unlab = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            try:
                xu = next(unlab_iter)
            except StopIteration:
                unlab_iter = iter(unlab_loader)
                xu = next(unlab_iter)
            xu = xu.to(device, non_blocking=True)

            # pseudo-labels from the weak (identity) view, no gradients
            with torch.no_grad():
                probs = F.softmax(model(xu), dim=1)
                conf, pseudo = probs.max(dim=1)
                mask = conf >= TAU
                if args.exclude_none:
                    mask &= pseudo != CLASSES.index("none")

            logits_sup = model(x)
            loss_sup = sup_loss_fn(logits_sup, y)

            loss_unsup = torch.tensor(0.0, device=device)
            if mask.any():
                logits_strong = model(cutout(xu[mask]))
                loss_unsup = F.cross_entropy(
                    logits_strong, pseudo[mask],
                    weight=w if args.weighted_unsup else None)

            loss = loss_sup + LAMBDA_U * loss_unsup
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            sup_sum += loss_sup.item()
            unsup_sum += float(loss_unsup.detach())
            pseudo_counts += torch.bincount(pseudo[mask].cpu(), minlength=len(CLASSES))
            n_pseudo += int(mask.sum())
            n_unlab += len(xu)

        sched.step()
        preds, targets = evaluate(model, val_loader, device)
        macro_f1 = f1_score(targets, preds, average="macro")
        flag = ""
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), OUT / f"{args.run}.pt")
            flag = "  <- best, saved"

        top3 = ", ".join(f"{CLASSES[i]} {pseudo_counts[i]/max(n_pseudo,1):.0%}"
                         for i in pseudo_counts.argsort(descending=True)[:3])
        print(f"epoch {epoch:2d}  sup {sup_sum/len(train_loader):.4f}  "
              f"unsup {unsup_sum/len(train_loader):.4f}  "
              f"pseudo-rate {n_pseudo/max(n_unlab,1):.1%} [{top3}]  "
              f"val macro-F1 {macro_f1:.4f}  ({time.time()-t0:.0f}s){flag}")

    model.load_state_dict(torch.load(OUT / f"{args.run}.pt", weights_only=True))
    preds, targets = evaluate(model, val_loader, device)
    print(f"\nbest val macro-F1: {f1_score(targets, preds, average='macro'):.4f}")
    print(f"plain accuracy:    {np.mean(preds == targets):.4f}\n")
    p, r, f1, support = precision_recall_fscore_support(
        targets, preds, labels=range(len(CLASSES)), zero_division=0)
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'F1':>10}{'n':>8}")
    for i, cls in enumerate(CLASSES):
        print(f"{cls:<12}{p[i]:>10.3f}{r[i]:>10.3f}{f1[i]:>10.3f}{support[i]:>8,}")
    cm = confusion_matrix(targets, preds, labels=range(len(CLASSES)))
    np.savetxt(OUT / f"{args.run}_confusion.csv", cm, fmt="%d", delimiter=",",
               header=",".join(CLASSES))
    print(f"\nconfusion matrix saved to {OUT / f'{args.run}_confusion.csv'}")


if __name__ == "__main__":
    main()

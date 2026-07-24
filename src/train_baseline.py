"""Train the supervised baseline on labelled data only.

Plain cross-entropy, no class weighting, no augmentation — deliberately. The point
of this run is to document what the standard recipe does under 989:1 imbalance.
Model selection: best macro-F1 on the validation split. The test split is not
touched here or anywhere until the final report.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from dataset import CLASSES, WaferDataset
from model import WaferCNN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

EPOCHS = 15
BATCH = 256
LR = 3e-4
SEED = 20260723


def class_weights(targets: torch.Tensor, mode: str) -> torch.Tensor | None:
    """Per-class loss weights from training-set frequencies.

    'inv'  : w_c ∝ 1/n_c   — textbook inverse frequency; with 989:1 imbalance the
             rarest class weighs ~100x the mean, which can destabilise training.
    'sqrt' : w_c ∝ 1/√n_c  — the gentle version, if 'inv' overshoots.
    Weights are normalised to mean 1 so the loss scale (and LR) stays comparable.
    """
    if mode == "none":
        return None
    counts = torch.bincount(targets, minlength=len(CLASSES)).float()
    w = 1.0 / (counts if mode == "inv" else counts.sqrt())
    return w * len(w) / w.sum()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        preds.append(logits.argmax(1).cpu())
        targets.append(y)
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return preds, targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", choices=["none", "inv", "sqrt"], default="none")
    ap.add_argument("--augment", action="store_true",
                    help="dihedral (rot90/flip) augmentation on the training set")
    ap.add_argument("--run", default="baseline",
                    help="name for output files: results/<run>.pt etc.")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda"

    train_ds = WaferDataset("train", augment=args.augment)
    val_ds = WaferDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=512, num_workers=4,
                            pin_memory=True, persistent_workers=True)

    model = WaferCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    w = class_weights(train_ds.targets, args.weights)
    if w is not None:
        print("class weights: " + ", ".join(
            f"{c}={w[i]:.2f}" for i, c in enumerate(CLASSES)))
    loss_fn = nn.CrossEntropyLoss(weight=None if w is None else w.to(device))

    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        sched.step()

        preds, targets = evaluate(model, val_loader, device)
        macro_f1 = f1_score(targets, preds, average="macro")
        flag = ""
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), OUT / f"{args.run}.pt")
            flag = "  <- best, saved"
        print(f"epoch {epoch:2d}  loss {running / len(train_ds):.4f}  "
              f"val macro-F1 {macro_f1:.4f}  ({time.time() - t0:.0f}s){flag}")

    # Final report from the best checkpoint
    model.load_state_dict(torch.load(OUT / f"{args.run}.pt", weights_only=True))
    preds, targets = evaluate(model, val_loader, device)

    print(f"\nbest val macro-F1: {f1_score(targets, preds, average='macro'):.4f}")
    print(f"plain accuracy:    {np.mean(preds == targets):.4f}  "
          f"(for contrast — see how little it tells you)\n")

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

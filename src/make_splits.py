"""Split the labelled portion of WM-811K into train/val/test — by lot, never by wafer.

Why by lot: wafers from the same lot went through the same tools at the same time and
share defect signatures. Splitting wafers randomly would put near-duplicates on both
sides of the train/test fence and inflate every score we ever report (data leakage).

Output: data/splits.npz with integer index arrays train/val/test into the h5 file.
Only labelled wafers (failureType != '') are assigned; the ~638k unlabelled wafers are
left out here and enter later as the semi-supervised pool.
"""

from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / "data" / "wm811k.h5"
OUT = ROOT / "data" / "splits.npz"

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 20260723  # fixed so the split is identical on every machine, forever


def main() -> None:
    with h5py.File(H5, "r") as f:
        labels = f["label"].asstr()[:]
        lots = f["lot"].asstr()[:]

    labelled = labels != ""
    idx = np.where(labelled)[0]
    lot_of = lots[idx]

    unique_lots = np.unique(lot_of)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_lots)

    n = len(unique_lots)
    n_train = int(n * FRACTIONS["train"])
    n_val = int(n * FRACTIONS["val"])
    lot_split = {
        "train": set(unique_lots[:n_train]),
        "val": set(unique_lots[n_train:n_train + n_val]),
        "test": set(unique_lots[n_train + n_val:]),
    }

    splits = {
        name: idx[np.isin(lot_of, list(lot_set))]
        for name, lot_set in lot_split.items()
    }

    print(f"{len(idx):,} labelled wafers across {n:,} lots\n")
    print(f"{'class':<12}" + "".join(f"{s:>9}" for s in splits))
    all_classes = sorted(set(labels[labelled]))
    for cls in all_classes:
        row = f"{cls:<12}"
        for name, s_idx in splits.items():
            row += f"{np.sum(labels[s_idx] == cls):>9,}"
        print(row)
    print(f"{'total':<12}" + "".join(f"{len(s):>9,}" for s in splits.values()))

    # A split where some class is missing from val or test cannot evaluate that class.
    for name, s_idx in splits.items():
        missing = [c for c in all_classes if np.sum(labels[s_idx] == c) == 0]
        assert not missing, f"{name} split is missing classes {missing}; change SEED"

    np.savez(OUT, **splits)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()

"""PyTorch Dataset for WM-811K wafer maps.

Preprocessing per map:
    1. Decompose the nominal-coded map {0 blank, 1 pass, 2 fail} into two binary
       channels — "die present" (map > 0) and "die failed" (map == 2). Interpolating
       the raw codes would average category labels, which is meaningless; averaging
       indicator channels yields densities, which is not.
    2. Bilinearly resize each channel to 64x64. Most maps are ~30x30, so this is
       usually an upsample (information-preserving); only the largest maps shrink.

Maps are loaded into RAM once per split (a few hundred MB as uint8) because reading
h5 rows one at a time from inside a DataLoader is an order of magnitude slower.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / "data" / "wm811k.h5"
SPLITS = ROOT / "data" / "splits.npz"

SIZE = 64
CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Near-full", "Random", "Scratch"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def to_tensor(flat: np.ndarray, h: int, w: int) -> torch.Tensor:
    """One raw wafer map -> float tensor of shape (2, SIZE, SIZE)."""
    m = torch.from_numpy(flat.reshape(h, w))
    chans = torch.stack([(m > 0), (m == 2)]).float().unsqueeze(0)  # (1, 2, h, w)
    return F.interpolate(chans, size=(SIZE, SIZE), mode="bilinear",
                         align_corners=False, antialias=True).squeeze(0)


class WaferDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        # Dihedral-group augmentation (90-degree rotations + flips): label-preserving
        # because every failure-type definition is rotation/mirror invariant, and
        # lossless because axis-aligned rotations just permute pixels. Train only —
        # the validation yardstick must never move.
        self.augment = augment
        idx = np.load(SPLITS)[split]
        with h5py.File(H5, "r") as f:
            order = np.argsort(idx)  # h5 fancy indexing requires sorted indices
            sorted_idx = idx[order]
            maps = f["maps"][sorted_idx]
            heights = f["heights"][sorted_idx]
            widths = f["widths"][sorted_idx]
            labels = f["label"].asstr()[sorted_idx]
        self.maps, self.heights, self.widths = maps, heights, widths
        self.targets = torch.tensor([CLASS_TO_ID[l] for l in labels])

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, i: int):
        x = to_tensor(self.maps[i], self.heights[i], self.widths[i])
        if self.augment:
            x = torch.rot90(x, k=int(torch.randint(4, ())), dims=(1, 2))
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(2,))
        return x, self.targets[i]


if __name__ == "__main__":
    ds = WaferDataset("val")
    x, y = ds[0]
    print(f"val size: {len(ds)}, sample shape: {tuple(x.shape)}, "
          f"dtype: {x.dtype}, label: {CLASSES[y]}")
    print(f"channel value ranges: die [{x[0].min():.2f},{x[0].max():.2f}], "
          f"fail [{x[1].min():.2f},{x[1].max():.2f}]")

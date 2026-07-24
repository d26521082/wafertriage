"""Convert the raw WM-811K pickle into a compact HDF5 and print a dataset summary.

The raw LSWMD.pkl is a ~2GB pandas DataFrame with columns:
    waferMap      2D uint8 array, values {0: blank, 1: pass die, 2: fail die}
    dieSize       float
    lotName       str
    waferIndex    float
    trianTestLabel  original train/test split (note the upstream typo)
    failureType   one of 8 defect classes, 'none', or [] (unlabelled)

Wafer maps come in many different shapes. We keep every map, store labels as strings
('' for unlabelled), and record shapes so downstream code can decide how to resize.

Usage:
    uv run python src/prepare_wm811k.py
"""

import pickle
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pandas.core.indexes.base

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "LSWMD.pkl"
OUT = Path(__file__).resolve().parents[1] / "data" / "wm811k.h5"


def load_raw(path: Path) -> pd.DataFrame:
    # The upstream pickle predates pandas 0.20 (module paths since renamed) and was
    # written under Python 2 (bytes needing latin1). Shim both.
    sys.modules["pandas.indexes"] = sys.modules["pandas.core.indexes"]
    sys.modules["pandas.indexes.base"] = pandas.core.indexes.base
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def unwrap(cell) -> str:
    """failureType / trianTestLabel cells are nested arrays like [['Edge-Ring']] or []."""
    a = np.asarray(cell)
    return str(a.flat[0]) if a.size else ""


def main() -> None:
    print(f"loading {RAW} ...")
    df = load_raw(RAW)
    n = len(df)
    print(f"{n:,} wafer maps")

    labels = df["failureType"].map(unwrap)
    split = df["trianTestLabel"].map(unwrap)
    shapes = df["waferMap"].map(lambda m: m.shape)

    print("\nlabel distribution:")
    for k, v in sorted(Counter(labels).items(), key=lambda kv: -kv[1]):
        name = k if k else "(unlabelled)"
        print(f"  {name:<12} {v:>8,}  ({v / n:6.2%})")

    shape_counts = Counter(shapes)
    print(f"\n{len(shape_counts)} distinct map shapes; 10 most common:")
    for s, c in shape_counts.most_common(10):
        print(f"  {s}: {c:,}")

    print(f"\nwriting {OUT} ...")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype()
    vlen_dt = h5py.vlen_dtype(np.uint8)
    with h5py.File(OUT, "w") as f:
        maps = f.create_dataset("maps", (n,), dtype=vlen_dt)
        for i, m in enumerate(df["waferMap"]):
            maps[i] = np.asarray(m, dtype=np.uint8).ravel()
            if i % 100_000 == 0:
                print(f"  {i:,}/{n:,}")
        f.create_dataset("heights", data=np.array([s[0] for s in shapes], dtype=np.int16))
        f.create_dataset("widths", data=np.array([s[1] for s in shapes], dtype=np.int16))
        f.create_dataset("label", data=labels.to_numpy(dtype=object), dtype=str_dt)
        f.create_dataset("orig_split", data=split.to_numpy(dtype=object), dtype=str_dt)
        f.create_dataset("lot", data=df["lotName"].astype(str).to_numpy(dtype=object), dtype=str_dt)
        f.create_dataset("wafer_index", data=df["waferIndex"].to_numpy(dtype=np.int16))
        f.create_dataset("die_size", data=df["dieSize"].to_numpy(dtype=np.float32))

    print(f"done: {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()

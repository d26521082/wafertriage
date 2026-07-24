"""Self-supervised pretraining: autoencoder on the 638k unlabeled wafer maps.

v3 of the use-the-unlabeled-data effort. v1/v2 (pseudo-labeling) both failed with
confirmation-bias dynamics; reconstruction has no pseudo-labels and no confidence
judgments, so that failure mode is structurally absent. The encoder (identical to
WaferCNN.features) learns what wafer maps look like from all 638k unlabeled maps;
train_baseline.py can then start classification from those weights (--init-encoder).
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import WaferCNN
from train_semi import UnlabeledWafers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ae_encoder.pt"

EPOCHS = 8
BATCH = 512
LR = 1e-3
SEED = 20260723


class WaferAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WaferCNN().features          # (B, 2, 64, 64) -> (B, 128, 8, 8)
        self.decoder = nn.Sequential(               # mirror it back to (B, 2, 64, 64)
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 2, 2, stride=2), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def main() -> None:
    torch.manual_seed(SEED)
    device = "cuda"

    print("loading unlabeled pool ...")
    ds = UnlabeledWafers()
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=8,
                        pin_memory=True, persistent_workers=True, drop_last=True)
    print(f"{len(ds):,} maps")

    model = WaferAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        t0, running = time.time(), 0.0
        for x in loader:
            x = x.to(device, non_blocking=True)
            loss = loss_fn(model(x), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"epoch {epoch}  recon MSE {running / len(loader):.5f}  "
              f"({time.time() - t0:.0f}s)")

    torch.save(model.encoder.state_dict(), OUT)
    print(f"saved encoder weights to {OUT}")


if __name__ == "__main__":
    main()

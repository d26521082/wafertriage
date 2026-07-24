# Experiment log

One line per run. Command is the full reproduction recipe; never delete rows.
All metrics are on the **validation** split — test stays locked until the final report.

| run | command | val macro-F1 | accuracy | notes |
|---|---|---|---|---|
| baseline | `uv run python src/train_baseline.py` | 0.8675 | 0.9759 | plain CE; misses concentrate on Loc/Scratch/Edge-Loc, errors flow to none (17–20%) |
| weighted_inv | `uv run python src/train_baseline.py --weights inv --run weighted_inv` | 0.8428 | 0.9596 | recalls +8–18pp on weak classes, precision collapses (Scratch 0.81→0.55); trades ~3 false alarms per extra catch — good or bad depends on cost ratio, not F1 |
| weighted_sqrt | `uv run python src/train_baseline.py --weights sqrt --run weighted_sqrt` | **0.8760** | 0.9771 | best of the three; most of inv's recall gains at a fraction of the precision cost (Scratch R 0.86 at P 0.71, Loc R 0.80 at P 0.79). **Supervised reference** — semi-supervised must beat this, not baseline |
| sqrt_aug | `uv run python src/train_baseline.py --weights sqrt --augment --run sqrt_aug` | 0.8725 | 0.9754 | dihedral augmentation ≈ wash (Donut/Near-full up, Loc/Random/Center down). Likely cause: real fabs have consistent wafer orientation, so the invariance assumption discards genuine directional signal. Reference stays weighted_sqrt; the augmentation code returns as FixMatch's weak-augment ingredient |

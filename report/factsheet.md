# Fact sheet — single source of truth

Every number the project may cite, **with its conditions attached**. All deliverables
(memo, write-up, walkthrough, README) quote from here and only here. If a number
isn't on this sheet, it doesn't get cited. Numbers marked (val) are on the 26,261-wafer
validation split; the test split (25,801) has never been opened.

## Data

| Fact | Value |
|---|---|
| Total wafer maps (WM-811K) | 811,457, real fabs, 632 distinct map shapes |
| Unlabeled | 638,507 (78.7%) |
| Labeled | 172,950 — of which 85.2% are "none" (no defect) |
| Defect-labeled | 25,519 = **3.1% of all maps** |
| Class imbalance | none : Near-full = 989 : 1 (147,431 vs 149) |
| Splits (by lot, seed 20260723) | train 120,888 / val 26,261 / test 25,801 |
| True defect share (val) | 14.295% |
| Model | 2-ch 64×64 input CNN, 288,329 params |

## Supervised results (val macro-F1 / accuracy)

| Run | macro-F1 | acc | Note |
|---|---|---|---|
| baseline (plain CE) | 0.8675 | 0.9759 | misses: Loc/Scratch/Edge-Loc → none 17–20% |
| weighted_inv | 0.8428 | 0.9596 | recall +8–18pp, precision collapse; ~3 extra false alarms per extra catch → **break-even miss:FA cost ratio ≈ 3.15** |
| **weighted_sqrt (reference)** | **0.8760** | 0.9771 | supervised reference |
| sqrt_aug | 0.8725 | 0.9754 | dihedral augmentation ≈ wash (orientation-signal hypothesis) |

## Unlabeled-data results

| Run | macro-F1 | Note |
|---|---|---|
| semi_v1 (FixMatch-style) | 0.8539 | collapse: none pseudo-share 93%→97%, F1 decayed to 0.601 by ep19 |
| semi_v2 (weighted+exclude-none) | 0.8203 | collapse re-hosted: Edge-Loc precision 0.52 |
| ae_finetune (AE pretrain, 100% labels) | 0.8708 | wash vs reference (−0.5pp) |

**Label-efficiency sweep** (3 subsample seeds, mean macro-F1, scratch vs AE-pretrained):

| Label budget | Wafers | Scratch | AE | Gap | Seeds |
|---|---|---|---|---|---|
| 100% | 120,888 | 0.876 | 0.871 | −0.5pp | 1 |
| 10% | 12,089 | 0.787 | 0.779 | −0.9pp | 3/3 negative |
| 3% | 3,627 | 0.687 | 0.701 | **+1.4pp** | **3/3 positive** |
| 1% | 1,209 | 0.579 | 0.604 | **+2.5pp** | **3/3 positive** |

Crossover ≈ 5,000–10,000 labels. Horizontal reading: pretraining ≈ several thousand
expert labels saved at the low end. Effect is **consistent but modest** — say so.

## Calibration (val, weighted_sqrt)

| Fact | Value |
|---|---|
| Overall ECE | 0.0074 |
| Defect-predicted subgroup ECE | 0.0117 (3,696 predictions) |
| Top confidence bin | claims 0.992, delivers 0.990 (n = 2,540) |

## Cost model & triage (all costs in units of one human review, C_rev = 1)

| Fact | Value |
|---|---|
| C_inv (engineer investigation) | 10 (range considered 5–20) |
| C_miss (missed systematic defect) | swept 10–1000, **never point-estimated** |
| Thresholds | p1 = C_rev/(C_miss − C_inv); p2 = (C_inv − C_rev)/C_inv = 0.90 |
| p1 at C_miss=100 | 1/90 ≈ 0.011 |

**Policy table (val), by C_miss assumption:**

| C_miss | auto-clear | human | flag | cost/wafer | all-review | argmax | missed among cleared |
|---|---|---|---|---|---|---|---|
| 30 | 80.2% | 6.8% | 12.9% | 1.539 | 2.429 | 1.674 | 23 (0.11%) |
| **100** | **63.3%** | 23.8% | 12.9% | **1.690** | 2.429 | 2.295 | **0** |
| 300 | 32.9% | 54.1% | 12.9% | 1.994 | 2.429 | 4.069 | 0 |
| 1000 | 10.5% | 76.6% | 12.9% | 2.218 | 2.429 | 10.280 | 0 |

Headline (conditions: C_miss=100, val): **auto-clear 63.3% with zero realized misses,
cost −30.4% vs full manual review, −26.4% vs no-triage argmax.** All-flag costs 10.0 always.

## Inspection allocation (station 2.4; C_miss=100, lot-opening cost F=5)

Review band: 6,240 wafers across 986 lots. b_i = min(p·C_miss, C_inv) − (C_rev + p·C_inv).

| Budget K | MIP benefit (wafers/lots) | Greedy (wafers/lots) | MIP edge |
|---|---|---|---|
| 1,000 | 3,351 (580 / 84) | 1,533 (195 / 161) | **+118.5%** |
| 3,000 | 7,607 (1,515 / 297) | 5,469 (755 / 449) | +39.1% |
| 6,000 | 11,422 (2,870 / 626) | 10,716 (2,155 / 769) | +6.6% |
| 10,000 | 13,339 | 13,309 | +0.2% |

Mechanism: at K=1,000 greedy spends 805 of 1,000 units on lot setups (161 lots × 5).
Batching is where the optimisation value lives; edge → 0 as budget loosens.

## Standing assumptions (attach to any claim)

1. Human review assumed error-free (first-version simplification).
2. All costs relative to one human review; no absolute currency anywhere.
3. Single dataset, single model scale, validation only; **test still sealed**.
4. Small-n classes carry wide CIs (val: Near-full 21, Donut 79).
5. The number "47" used earlier in conversation was an illustrative placeholder,
   NOT a computed result — never cite it.

## FINAL TEST RESULTS (unsealed once, 2026-08-26 — these are the definitive numbers)

Model: weighted_sqrt only. Test split: 25,801 wafers, true defect share 15.104%.

| Fact | Test | (Val, for reference) |
|---|---|---|
| macro-F1 | **0.8621** | 0.8760 |
| accuracy | 0.9740 | 0.9771 |
| Triage @ C_miss=100: auto-clear | **62.7%** | 63.3% |
| Triage @ C_miss=100: missed among cleared | **7 (0.043%)** | 0 |
| Triage @ C_miss=100: cost vs all-review | **−28.5%** | −30.4% |
| Triage @ C_miss=300: missed | 1 (0.012%) | 0 |
| Triage @ C_miss=1000: missed | 0 | 0 |

Interpretation (report verbatim): the −1.4pp macro-F1 gap is the price of model
selection on val (two-layer selection-contamination logic). The val "zero misses"
did NOT replicate: test observed 0.043%, exceeding val's rule-of-three 95% upper
bound (0.018%) by ~2.4× (Poisson P(X≥7 | bound) ≈ 3%) — consistent with mild val
contamination of the policy thresholds, exactly the risk pre-declared in the
reviewer-criticism note. The economic conclusion is robust; the zero-miss claim
must be stated as "zero at the most conservative setting; ~0.04% at C_miss=100."
Per-class test table in results/final_test_confusion.csv.

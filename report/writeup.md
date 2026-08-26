# Wafer-Map Defect Triage under Label Scarcity — Technical Report

Chen-Ting Lin · August 2026 · code and data recipes at `github.com/d26521082/wafertriage`

## 1. Problem and data

After fabrication, every die on a wafer is electrically probed and marked pass or
fail. Plotted by position, the failures form a *wafer map*, and the spatial pattern
of failures carries diagnostic information: a ring at the edge points to an
edge-uniformity problem, a thin line to a mechanical scratch, a central blob to a
polishing or deposition issue. Reading these maps is how yield engineers locate
misbehaving process steps. It is also slow, expensive, and asymmetric in its
errors: a pattern that goes unnoticed means a faulty tool keeps producing scrap
until someone else catches it, while an unnecessary second look costs minutes.

The dataset is WM-811K: 811,457 real wafer maps in 632 different sizes. Only
172,950 carry an expert label, and 85% of those are "no pattern," so the maps
with an actual defect label number 25,519 — 3.1% of the data — spread across
eight pattern classes with a 989:1 imbalance between the most and least common.
This is the working condition of a real fab: data is abundant, expert labels are
scarce, and the classes that matter most are the rarest.

Two preprocessing decisions matter. First, the raw maps encode {blank, pass,
fail} as {0, 1, 2}; these are category codes, not quantities, so resizing them
directly would average labels into nonsense. Each map is instead decomposed
into two binary channels — die present, die failed — and only then resized to
64×64, so that interpolation produces meaningful failure densities. Second,
wafers from the same manufacturing lot share tools and defect signatures, so
the data is split 70/15/15 **by lot**, never by wafer. Random splits let a
model recognize lots rather than patterns, which is why published accuracies
on this dataset (96–98%) are not comparable to the numbers below.

## 2. Supervised foundation, and what unlabeled data is worth

A 288K-parameter CNN trained with plain cross-entropy reaches macro-F1 0.868
on validation. Its 97.6% accuracy is a mirage of the majority class; the
per-class view shows the real problem: Loc, Scratch, and Edge-Loc patterns are
missed 17–20% of the time, and those misses flow to "no pattern" — the
expensive direction. Class-reweighting the loss fixes part of this. Full
inverse-frequency weighting overshoots (recall up 8–18 points, precision
collapses, macro-F1 0.843); square-root weighting keeps most of the recall gain
at a fraction of the precision cost and becomes the reference model at
macro-F1 0.876. Rotation and flip augmentation, a textbook move for images,
was a wash here: wafers sit in a consistent orientation in real fabs, so the
invariance the augmentation imposes discards genuine signal.

The 638K unlabeled maps were then tested three ways. Two pseudo-labeling
variants (FixMatch-style) failed instructively: the first collapsed into
labeling everything "no pattern" (its pseudo-labels went from 93% to 97%
majority class while validation macro-F1 decayed to 0.60), and the second,
after excluding the majority class, simply re-hosted the same confirmation
bias in the next most confident class. Autoencoder pretraining, which has no
self-reinforcing loop, did not collapse but did not help either — a wash
against the reference with all labels available.

The resolution came from asking the question properly. A label-efficiency
sweep retrained the model on 10%, 3%, and 1% of the labels, with and without
pretraining, three random subsamples each. Pretraining is neutral or slightly
harmful with abundant labels, and becomes a consistent advantage below a
crossover of roughly 5–10K labels: +1.4 points at 3% and +2.5 points at 1% of
the labels, positive in six of six seeded comparisons below the crossover.
Unlabeled data pays exactly when labels are scarce, and the earlier failures
had simply been fought on the wrong side of that line.

## 3. From predictions to decisions

A classifier outputs probabilities; a fab needs dispositions. Three steps
connect them.

**Calibration.** Before probabilities can be multiplied by costs, they must be
honest. Expected calibration error on validation is 0.0074 overall and 0.0117
on the subset predicted as defects, so the reference model's probabilities can
be used at face value on this distribution.

**A cost model with no invented numbers.** Every outcome is priced in units of
one manual review: an engineer investigation costs 10 such units, and the cost
of a missed systematic defect — the one number nobody has — is never fixed but
swept from 10 to 1,000. With three actions available (auto-clear, auto-flag to
an engineer, send to human review), the expected cost of each is a straight
line in the defect probability p, and the optimal policy is whichever line is
lowest. The two crossing points give the triage thresholds in closed form:
clear below p₁ = C_rev/(C_miss − C_inv), flag above p₂ = (C_inv − C_rev)/C_inv,
review in between. The three-way structure is not designed in; it falls out of
the cost matrix.

**Allocation when reviewers are scarce.** If human capacity is capped, which
wafers get the reviewer? Reviewing a wafer requires opening its lot context at
a fixed cost, so the problem is a facility-location-style integer program:
maximize captured benefit subject to a time budget, with lot-opening variables
linking wafers. Against a naive "review the highest-value wafers first" rule,
the exact solution captures 118% more benefit at a tight budget — the greedy
rule spends 80% of its budget opening lots — and the advantage vanishes as the
budget loosens. Knowing when optimization is worth its complexity is part of
the result.

## 4. Final evaluation on the sealed test set

The test split (25,801 wafers) was never read during development and was
evaluated once, under a protocol written down before the run.

Macro-F1 came in at 0.862 against 0.876 on validation. The gap is the expected
price of model selection: the checkpoint and recipe were chosen by their
validation scores, and a winner chosen by a noisy score is always slightly
flattered by it.

The triage economics held. At a missed-defect cost of 100 reviews, 62.7% of
wafers are auto-cleared (validation: 63.3%), 23.7% go to human review, 13.6%
are flagged directly, and total cost falls 28.5% against full manual review
(validation: 30.4%). What did not hold was the validation set's most
attractive statistic: zero missed defects among cleared wafers. On the test
set, seven defective wafers were cleared — a miss rate of 0.043%, above the
statistical upper bound that the validation zero had implied. A zero count is
the most fragile statistic there is, and validation had been used to choose
the model that produced it. The most conservative setting (missed-defect cost
1,000×) does reach zero observed misses on test, at 9.8% automation. The
headline is therefore stated as it was measured: about 63% automation, a miss
rate of a few per ten thousand, and a cost reduction near 30%, with zero
misses only at the most conservative setting.

## 5. Conclusion

The central finding of this project is not a higher defect-classification
score. It is that wafer-map recognition can be carried one step further, into
an inspection process that routes wafers by risk and is judged by actual
cost. Under label scarcity, the value of pretraining on unlabeled data is not
a fixed property of the method but depends on how many labels are available:
with ample labels the benefit is marginal, while in the low-label regime it
improves performance clearly.

The results also show that a model score cannot be equated with production
value. Severe class imbalance, lot-to-lot variation, and model selection all
affect what a test set reports. This project therefore split the data by lot,
kept the test set sealed until the end, and disclosed the gap between
validation and test in the final evaluation rather than reporting only the
best validation number.

Once model outputs are converted into risk probabilities, triage thresholds
can be set from the different costs of a missed defect and a manual review.
Under the assumption that a miss costs 100 times a review, on 25,801 sealed
test wafers that never took part in development, the system auto-clears 62.7%
of wafer maps with an observed miss rate of 0.043% (7 wafers), at a total cost
28.5% below a fully manual process.

The value of this work, then, lies not in claiming that the model can replace
engineers, but in establishing a pilot framework that can be tuned to real
costs, real risk tolerances, and real data conditions. The most important
next step is not to chase a higher single-model score. It is to obtain, on a
real production line, the actual cost of a missed defect, the actual cost of
manual inspection, and the lot-level distribution of incoming wafers — and
then to run a small pilot that tests whether the cost and risk gains of this
triage strategy reproduce in a real environment.
